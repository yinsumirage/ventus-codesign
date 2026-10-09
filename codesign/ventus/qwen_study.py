"""Qwen A/B/C co-design study: layer-sliced instruction simulation and receipts.

Every number here comes from the native instruction executor running real ELF
dispatches. The sliced estimate simulates one representative transformer layer,
the non-layer prologue/epilogue and the LM head, then scales the layer by the
layer count. Complete scenario runs remain the separate `instructions` command.
"""

import contextlib
import hashlib
import io
import json
import re
from pathlib import Path

from .config import Hardware
from .elf import ELFProgram
from .model_program import qwen_manifest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tests/fixtures/compiled-transformer/bundle.json"
# Same frozen ELFs plus the compiled column-order GEMM kernels.
NCOL_BUNDLE = ROOT / "tests/fixtures/compiled-transformer-ncol/bundle.json"
BASELINE_HARDWARE = ROOT / "examples/baseline-hardware-v1.json"
SCENARIOS = {
    "Q-P128": ("prefill", 128),
    "Q-P512": ("prefill", 512),
    "Q-D128": ("decode", 128),
    "Q-D512": ("decode", 512),
}
# Frozen default H0+S0 values: 128 complete simulations, 512 labelled extrapolations.
BASELINE_T0 = {
    "Q-P128": 812589857,
    "Q-P512": 3787836632,
    "Q-D128": 2481135702 / 16,
    "Q-D512": 2894095806 / 16,
}
TASKS = {
    "A": ("Q-P128", "Q-P512"),
    "B": ("Q-D128", "Q-D512"),
    "C": ("Q-P128", "Q-P512", "Q-D128", "Q-D512"),
}
LAYERS = 24
DECODE_STEP = 8  # middle of 16 steps; per-step cycles vary by <0.4% in the baseline
SAMPLE_LAYER = 0
# Estimator v2: decode slices first replay the previous step's tail (last layer, final
# norm, LM head, cast) so caches carry the state a continuous decode run has; the
# sampled layers then reproduced complete-run layer cycles. Prefill starts cold by
# definition. Sampled layers are scaled by 24 / count.
ESTIMATOR = __import__("os").environ.get("QWEN_STUDY_ESTIMATOR", "v2")
SAMPLE_LAYERS = {"Q-P128": (0, 1, 2), "Q-P512": (0,), "Q-D128": (0, 1, 2), "Q-D512": (0, 1, 2)}
FIXED = dict(threads=32, line_bytes=128, memory_channels=1, memory_bytes_per_cycle=64)
FIXED["memory_latency"] = 100


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def full_hardware(fields):
    base = json.loads(BASELINE_HARDWARE.read_text())
    values = {**base, **fields}
    if any(values[k] != v for k, v in FIXED.items()):
        raise ValueError("threads, line width and external memory are frozen")
    return Hardware(**values).validate().to_dict()


def program(cache, scenario, software):
    """Build (once) the full scenario program for a software choice."""
    phase, context = SCENARIOS[scenario]
    out = Path(cache) / digest(software)[:16] / scenario
    if not (out / "program.json").is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / "software.json").write_text(json.dumps(software, indent=2) + "\n")
        import os
        import shutil

        tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            bundle = NCOL_BUNDLE if software.get("block_order") == "col" else BUNDLE
            qwen_manifest(
                bundle, tmp, phase=phase, context=context, mapping="packed64", software=software
            )
        try:
            tmp.rename(out)
        except OSError:
            # Another process published the same deterministic program first.
            shutil.rmtree(tmp)
            if not (out / "program.json").is_file():
                raise
    return out / "program.json"


def classify(name, layers=(SAMPLE_LAYER,)):
    """Dispatch name -> (part, generic class, step). Parts: layer, lm, other, skip."""
    step = name.split(".")[0]
    rest = name[len(step) + 1 :]
    m = re.match(r"layer(\d+)\.(.*)", rest)
    if m:
        cls = re.sub(r"(head\d+|kv\d+)\.", "", m.group(2))
        return ("layer" if int(m.group(1)) in layers else "skip"), cls, step
    if rest.startswith("lm_head"):
        return "lm", rest, step
    return "other", rest, step


def slices(raw, scenario):
    """Sliced dispatch lists: one sampled layer plus prologue/epilogue, and the LM head."""
    phase = SCENARIOS[scenario][0]
    want = "prefill" if phase == "prefill" else f"decode{DECODE_STEP:02}"
    parts = {"body": [], "lm": []}
    for d in raw["dispatches"]:
        part, _, step = classify(d["name"])
        if step != want or part == "skip":
            continue
        parts["lm" if part == "lm" else "body"].append(d)
    return parts


def slices_v2(raw, scenario):
    """Estimator v2 slices: decode bodies include their warm-up and their LM head."""
    phase = SCENARIOS[scenario][0]
    layers = SAMPLE_LAYERS[scenario]
    if phase == "prefill":
        parts = {"body": [], "lm": []}
        for d in raw["dispatches"]:
            part, _, _ = classify(d["name"], layers)
            if part != "skip":
                parts["lm" if part == "lm" else "body"].append(d)
        return parts
    want, previous = f"decode{DECODE_STEP:02}", f"decode{DECODE_STEP - 1:02}"
    warm, body = [], []
    for d in raw["dispatches"]:
        part, _, step = classify(d["name"], layers)
        if step == previous and (part != "skip" or d["name"].startswith(previous + ".layer23.")):
            if not d["name"].startswith(previous + ".layer0") and "embedding" not in d["name"]:
                warm.append(d)
        elif step == want and part != "skip":
            body.append(d)
    return {"body": warm + body}


def launch_resources(raw, root, dispatches):
    elves, out = {}, {}
    for d in dispatches:
        path = root / d["elf"]
        if path not in elves:
            elves[path] = ELFProgram.read(path)
        res = elves[path].resources[d["kernel"]]
        lds = (res["lds_static"] + res["lds_stack"] + 127) // 128 * 128
        out[d["kernel"]] = dict(vgpr=res["vgpr"], sgpr=res["sgpr"], lds=lds)
    return out


def residency(hw, resources):
    """Resident one-warp blocks per SM, exactly as the executor computes it."""
    r = min(hw["blocks_per_sm"], hw["warps_per_sm"])
    for cap, use in (
        (hw["vgpr_slots"], resources["vgpr"]),
        (hw["sgpr_slots"], resources["sgpr"]),
        (hw["lds_bytes"], resources["lds"]),
    ):
        if use:
            r = min(r, cap // use)
    return r


def simulate(raw, root, dispatches, hw):
    """Run dispatches in order in one cold session; return per-dispatch rows."""
    from .instruction_cli import prepare
    from .rust import RustSession

    sub = dict(raw, dispatches=dispatches)
    h = Hardware(**hw).validate()
    rows = []
    with RustSession(h, instruction_target=raw.get("instruction_target")) as session:
        for d, (name, prepared) in zip(dispatches, prepare(sub, root, h, session), strict=True):
            row = session.dispatch(prepared)
            expected = d.get("work", {}).get("mma_instructions")
            mma = sum(row["counters"].get(k, 0) for k in ("mma_bf16", "mma_f16", "mma_tf32"))
            if expected is not None and mma != expected:
                raise ValueError(f"{name}: dynamic MMA count {mma} differs from shape check")
            rows.append(
                dict(
                    name=name,
                    cycles=row["cycles"],
                    instructions=row["instructions"],
                    requested_bytes=row["requested_bytes"],
                    counters=row["counters"],
                )
            )
    return rows


class Study:
    """Cached sliced evaluations in an append-only JSONL database."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        import os
        import time

        # One append-only file per writer process; readers merge all of them.
        self.db_path = self.dir / f"slices-{int(time.time())}-{os.getpid()}.jsonl"
        self.db = {}
        for path in sorted(self.dir.glob("slices*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a concurrent writer's unfinished last line
                self.db[row["key"]] = row

    def part_key(self, hw, scenario, software, part):
        if ESTIMATOR == "v2" and part == "body":
            return digest(
                dict(hw=hw, scenario=scenario, software=software, part=part, estimator="v2")
            )
        # LM-head work is identical for every scenario (one row, same weights).
        lm_tile = software.get("tiles", {}).get("lm_head")
        if part == "lm":
            key = dict(hw=hw, part="lm", lm_tile=lm_tile)
            if software.get("block_order", "row") != "row":
                key["block_order"] = software["block_order"]
            if software.get("epilogue", "direct") != "direct":
                key["epilogue"] = software["epilogue"]
            return digest(key)
        return digest(dict(hw=hw, scenario=scenario, software=software, part=part))

    def jobs(self, hw, software, scenarios):
        """Pending (key, scenario, part) units for this design."""
        pending = {}
        for scenario in scenarios:
            decode_v2 = ESTIMATOR == "v2" and SCENARIOS[scenario][0] == "decode"
            for part in ("body",) if decode_v2 else ("body", "lm"):
                key = self.part_key(hw, scenario, software, part)
                if key not in self.db and key not in pending:
                    pending[key] = (hw, scenario, software, part)
        return pending

    def record(self, key, hw, scenario, software, part, rows, host):
        row = dict(
            key=key,
            hardware=hw,
            scenario=scenario if part != "lm" else None,
            software=software if part != "lm" else None,
            lm_tile=software.get("tiles", {}).get("lm_head") if part == "lm" else None,
            part=part,
            rows=rows,
            host_seconds=host,
        )
        with self.db_path.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.db[key] = row

    def estimate(self, hw, software, scenario):
        """Scenario cycles from cached slices; None while a slice is missing."""
        body = self.db.get(self.part_key(hw, scenario, software, "body"))
        decode_v2 = ESTIMATOR == "v2" and SCENARIOS[scenario][0] == "decode"
        lm = None if decode_v2 else self.db.get(self.part_key(hw, scenario, software, "lm"))
        if body is None or (lm is None and not decode_v2):
            return None
        layers = SAMPLE_LAYERS[scenario] if ESTIMATOR == "v2" else (SAMPLE_LAYER,)
        want = "prefill" if SCENARIOS[scenario][0] == "prefill" else f"decode{DECODE_STEP:02}"
        classes = {}
        total = 0
        lm_rows = lm["rows"] if lm else []
        for r in body["rows"] + lm_rows:
            part, cls, step = classify(r["name"], layers)
            if decode_v2 and step != want:
                continue  # warm-up dispatches of the previous decode step
            scale = LAYERS / len(layers) if part == "layer" else 1
            total += r["cycles"] * scale
            classes[cls] = classes.get(cls, 0) + r["cycles"] * scale
        return dict(cycles=total, classes=classes, estimator=ESTIMATOR)


def run_job(args):
    """Worker: build/slice a program and simulate one part (picklable)."""
    from time import perf_counter

    cache, key, hw, scenario, software, part = args
    path = program(cache, scenario, software)
    raw = json.loads(path.read_text())
    if part == "lm":
        # Reuse the prefill-128 LM head; its work equals every scenario's LM head.
        path = program(cache, "Q-P128", software)
        raw = json.loads(path.read_text())
        dispatches = slices(raw, "Q-P128")["lm"]
    elif ESTIMATOR == "v2":
        dispatches = slices_v2(raw, scenario)["body"]
    else:
        dispatches = slices(raw, scenario)["body"]
    begin = perf_counter()
    rows = simulate(raw, path.parent, dispatches, hw)
    return key, hw, scenario, software, part, rows, perf_counter() - begin


def check_legal(cache, hw, software, scenarios, area_check=True):
    """Area budget and per-kernel residency; returns (ok, reasons, area).

    Exploration samples may exceed the area budget (area_check=False) to teach
    the surrogate; they are never reported as designs.
    """
    from codesign.ventus_costs.unified import evaluate_unified_area

    reasons = []
    cost = evaluate_unified_area(hw)
    if area_check and not cost["area_feasible"]:
        reasons.append(f"area {cost['total_area_mm2']:.6f} mm2 exceeds budget")
    for scenario in scenarios:
        path = program(cache, scenario, software)
        raw = json.loads(path.read_text())
        parts = slices(raw, scenario)
        res = launch_resources(raw, path.parent, parts["body"] + parts["lm"])
        for kernel, r in res.items():
            if residency(hw, r) < 1:
                reasons.append(f"{kernel} ({r}) cannot reside")
    return not reasons, sorted(set(reasons)), cost["total_area_mm2"]


def run_pending(study, cache, pending, workers):
    import multiprocessing as mp

    if not pending:
        return
    jobs = [(str(cache), k, *v) for k, v in pending.items()]
    # Longest first: prefill-512 bodies dominate.
    weight = {"Q-P512": 4, "Q-P128": 2}
    jobs.sort(key=lambda j: -(weight.get(j[3], 1) if j[5] == "body" else 2))
    with mp.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for result in pool.imap_unordered(run_job, jobs):
            study.record(*result)
            print(
                json.dumps(dict(done=result[2] or "lm", part=result[4], host=round(result[6], 1))),
                flush=True,
            )


def ensure_programs(cache, designs, scenarios):
    for software in {digest(sw): sw for _, sw in designs}.values():
        for scenario in set(scenarios) | {"Q-P128"}:
            program(cache, scenario, software)


def evaluate_designs(study_dir, designs, scenarios, workers=4, area_check=True):
    """designs: [(hardware fields, software)] -> list of per-design results."""
    study = Study(study_dir)
    cache = study.dir / "programs"
    full = [(full_hardware(hw), sw) for hw, sw in designs]
    ensure_programs(cache, full, scenarios)
    pending, results = {}, []
    for hw, sw in full:
        ok, reasons, area = check_legal(cache, hw, sw, scenarios, area_check)
        results.append(
            dict(
                hardware=hw,
                software=sw,
                legal=ok,
                reasons=reasons,
                area_mm2=area,
                area_feasible=area <= 1.11 + 1e-12,
            )
        )
        if ok:
            pending.update(study.jobs(hw, sw, scenarios))
    run_pending(study, cache, pending, workers)
    for res in results:
        if not res["legal"]:
            continue
        res["scenarios"] = {}
        for scenario in scenarios:
            est = study.estimate(res["hardware"], res["software"], scenario)
            est["r_vs_frozen_T0"] = est["cycles"] / BASELINE_T0[scenario]
            res["scenarios"][scenario] = est
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--designs", type=Path, required=True, help="JSON [[hardware, software]]")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--explore", action="store_true", help="allow over-budget samples")
    args = parser.parse_args()
    designs = json.loads(args.designs.read_text())
    results = evaluate_designs(
        args.study, designs, args.scenarios, args.workers, area_check=not args.explore
    )
    for res in results:
        line = dict(area=round(res["area_mm2"], 6), legal=res["legal"], reasons=res["reasons"])
        for s, e in res.get("scenarios", {}).items():
            line[s] = round(e["r_vs_frozen_T0"], 4)
        print(json.dumps(line))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
