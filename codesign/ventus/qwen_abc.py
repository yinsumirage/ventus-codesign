"""Rounds of fit -> MIP -> independent simulation for tasks A/B/C.

Usage (inside the configured execution environment):
  python -m codesign.ventus.qwen_abc --study results/abc-study/db --task A --rounds 2

Each round fits per-scenario surrogates from every cached simulation, solves one
MIP per phase-tile combination, decodes the solutions, checks area and compiled
residency independently, and simulates the new designs. Round receipts (models'
fit errors, solver status, predictions and simulated ratios) are appended to
`<study>/rounds-<task>-<policy>.jsonl`; nothing is overwritten.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

from . import qwen_search as qs
from .qwen_study import BASELINE_T0, TASKS, Study, evaluate_designs


def samples(study, scenario):
    """All complete cached estimates for one scenario as surrogate training rows."""
    phase = qs.phase_of(scenario)
    out = []
    for row in study.db.values():
        if row["part"] != "body" or row["scenario"] != scenario:
            continue
        est = study.estimate(row["hardware"], row["software"], scenario)
        tile = qs.tile_of(phase, row["software"])
        if est is None or tile == "mixed":
            continue
        out.append(
            dict(
                hardware=row["hardware"],
                tile=tile,
                variant=qs.variant_of(row["software"], phase),
                cycles=est["cycles"],
                software=row["software"],
            )
        )
    return out


def fit_models(study, scenarios):
    models = {}
    for scenario in sorted(scenarios, key=lambda s: "512" in s):
        rows = samples(study, scenario)
        prior = None
        if "512" in scenario:
            prior = models.get(scenario.replace("512", "128"))
            if prior is None:
                prior = qs.fit(samples(study, scenario.replace("512", "128")), scenario)
        models[scenario] = qs.fit(rows, scenario, prior=prior)
    return models


def qs_first(scenarios, phase):
    return next(s for s in scenarios if qs.phase_of(s) == phase)


def best_known(study, scenarios, area_ok=True, allowed=None):
    """Best simulated design for a scenario set (max r), using only area-feasible rows.

    allowed: optional list of phase -> software dicts the design may use.
    """
    from codesign.ventus_costs.unified import evaluate_unified_area

    by_design = {}
    for row in study.db.values():
        if row["part"] != "body" or row["scenario"] not in scenarios:
            continue
        if allowed is not None and row["software"] not in [
            a[qs.phase_of(row["scenario"])] for a in allowed
        ]:
            continue
        key = json.dumps(row["hardware"], sort_keys=True)
        by_design.setdefault(key, {}).setdefault(row["scenario"], []).append(row["software"])
    best = None
    for key, per in by_design.items():
        hw = json.loads(key)
        if set(per) != set(scenarios):
            continue
        if area_ok and not evaluate_unified_area(hw)["area_feasible"]:
            continue
        ratios = {}
        choice = {}
        for scenario, softwares in per.items():
            options = []
            for sw in softwares:
                est = study.estimate(hw, sw, scenario)
                if est:
                    options.append((est["cycles"] / BASELINE_T0[scenario], sw))
            if not options:
                break
            ratios[scenario], choice[scenario] = min(options, key=lambda x: x[0])
        else:
            score = max(ratios.values())
            if best is None or score < best["max_r"]:
                best = dict(max_r=score, hardware=hw, ratios=ratios, software=choice)
    return best


def design_scores(study, scenarios):
    """hardware key -> {scenario: best simulated ratio over cached software choices}."""
    per = {}
    for row in study.db.values():
        if row["part"] != "body" or row["scenario"] not in scenarios:
            continue
        est = study.estimate(row["hardware"], row["software"], row["scenario"])
        if est is None:
            continue
        key = json.dumps(row["hardware"], sort_keys=True)
        ratio = est["cycles"] / BASELINE_T0[row["scenario"]]
        cur = per.setdefault(key, {})
        cur[row["scenario"]] = min(ratio, cur.get(row["scenario"], ratio))
    return per


def feasible_starts(study, scenarios):
    """Area-feasible simulated designs, best (partial) max ratio first."""
    from codesign.ventus_costs.unified import evaluate_unified_area

    from .qwen_study import full_hardware

    ranked = []
    for key, ratios in design_scores(study, scenarios).items():
        hw = json.loads(key)
        if evaluate_unified_area(hw)["area_feasible"]:
            missing = len(set(scenarios) - set(ratios))
            ranked.append((missing, max(ratios.values()), key))
    ranked.sort()
    return [json.loads(k) for _, _, k in ranked] + [full_hardware({})]


def policy_softwares(policy, phase, fixed=None):
    """Software menu per control: joint enumerates tiles, others fix one program."""
    if policy == "hardware-only":
        return [{}]
    if policy == "sequential":
        return [fixed[phase]]
    return qs.phase_menu(phase)


def simulated_with(study, scenarios, software_by_phase):
    """Hardware already simulated with exactly this software for every scenario."""
    keys = {
        (json.dumps(r["hardware"], sort_keys=True), r["scenario"])
        for r in study.db.values()
        if r["part"] == "body"
        and r["scenario"] in scenarios
        and r["software"] == software_by_phase[qs.phase_of(r["scenario"])]
    }
    hws = {k for k, _ in keys}
    return [json.loads(h) for h in hws if all((h, s) in keys for s in scenarios)]


def run_round(
    study_dir,
    task,
    *,
    workers,
    time_limit,
    top,
    log,
    trust_radius=4,
    per_radius=2,
    policy="joint",
    fixed=None,
):
    study = Study(study_dir)
    scenarios = TASKS[task]
    phases = sorted({qs.phase_of(s) for s in scenarios})
    begin = time.perf_counter()
    models = fit_models(study, scenarios)
    starts = feasible_starts(study, scenarios)
    menus = [policy_softwares(policy, p, fixed) for p in phases]
    solutions = []

    def solve_for(software_by_phase, trust=None, extra_exclude=()):
        tiles = {p: qs.tile_of(p, software_by_phase[p]) for p in phases}
        variants = {p: qs.variant_of(software_by_phase[p], p) for p in phases}
        sol = qs.solve(
            models,
            scenarios,
            tiles,
            variants,
            exclude=simulated_with(study, scenarios, software_by_phase) + list(extra_exclude),
            starts=starts,
            trust=trust,
            time_limit=time_limit if trust is None else min(time_limit, 30),
        )
        sol["software"] = software_by_phase
        if trust:
            sol["trust_radius"] = trust[1]
        solutions.append(sol)

    for combo in itertools.product(*menus):
        solve_for(dict(zip(phases, combo)))
    allowed = [dict(zip(phases, combo)) for combo in itertools.product(*menus)]
    incumbent = best_known(study, scenarios, allowed=allowed)
    if incumbent and trust_radius:
        # Local refinement around the incumbent with its own software.
        software = {p: incumbent["software"][qs_first(scenarios, p)] for p in phases}
        # Top-k per radius: each solution is cut off before the next solve.
        for radius in range(1, trust_radius + 1):
            found = []
            for _ in range(per_radius):
                solve_for(software, (incumbent["hardware"], radius), found)
                if not solutions[-1].get("hardware"):
                    break
                found.append(solutions[-1]["hardware"])
    ranked = sorted((s for s in solutions if s.get("hardware")), key=lambda s: s["predicted_max_r"])
    feasible, keys = [], set()
    for sol in ranked:
        key = json.dumps([sol["hardware"], sol["software"]], sort_keys=True)
        if key not in keys:
            keys.add(key)
            feasible.append(sol)
    global_top = [s for s in feasible if "trust_radius" not in s][:top]
    local = [s for s in feasible if "trust_radius" in s]
    feasible = global_top + [s for s in local if s not in global_top]
    scen_by_phase = {p: [s for s in scenarios if qs.phase_of(s) == p] for p in phases}
    simulated = [
        dict(
            predicted=sol["predicted"],
            tiles=sol["tiles"],
            software=sol["software"],
            trust_radius=sol.get("trust_radius"),
            hardware={k: sol["hardware"][k] for k in qs.FIELDS},
            ratios={},
            legal=True,
            reasons=[],
        )
        for sol in feasible
    ]
    for phase in phases:
        batch = [[row["hardware"], row["software"][phase]] for row in simulated]
        results = evaluate_designs(study_dir, batch, scen_by_phase[phase], workers)
        for row, res in zip(simulated, results, strict=True):
            row["legal"] = row["legal"] and res["legal"]
            row["reasons"] += res["reasons"]
            row["area_mm2"] = res["area_mm2"]
            for s, e in res.get("scenarios", {}).items():
                row["ratios"][s] = e["cycles"] / BASELINE_T0[s]
    for row in simulated:
        if len(row["ratios"]) == len(scenarios):
            row["max_r"] = max(row["ratios"].values())
    study = Study(study_dir)
    receipt = dict(
        task=task,
        models={
            s: {k: m[k] for k in ("samples", "rms_log_error", "loo_rms_log_error")}
            for s, m in models.items()
        },
        solves=[{k: v for k, v in s.items() if k != "hardware"} for s in solutions],
        simulated=simulated,
        policy=policy,
        best_known=best_known(study, scenarios, allowed=allowed),
        host_seconds=time.perf_counter() - begin,
    )
    with open(log, "a") as f:
        f.write(json.dumps(receipt) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--trust-radius", type=int, default=4)
    parser.add_argument("--per-radius", type=int, default=2)
    parser.add_argument(
        "--policy", choices=["joint", "hardware-only", "sequential"], default="joint"
    )
    parser.add_argument("--fixed-software", type=Path, help="phase -> software JSON")
    args = parser.parse_args()
    fixed = json.loads(args.fixed_software.read_text()) if args.fixed_software else None
    log = args.study / f"rounds-{args.task}-{args.policy}.jsonl"
    for _ in range(args.rounds):
        r = run_round(
            args.study,
            args.task,
            workers=args.workers,
            time_limit=args.time_limit,
            top=args.top,
            log=log,
            policy=args.policy,
            fixed=fixed,
            trust_radius=args.trust_radius,
            per_radius=args.per_radius,
        )
        print(
            json.dumps(
                dict(
                    task=args.task,
                    policy=args.policy,
                    models=r["models"],
                    simulated=[
                        dict(
                            tiles=s["tiles"],
                            pred={k: round(v, 3) for k, v in s["predicted"].items()},
                            sim={k: round(v, 3) for k, v in s["ratios"].items()},
                            area=round(s.get("area_mm2") or 0, 4),
                        )
                        for s in r["simulated"]
                    ],
                    best=r["best_known"] and round(r["best_known"]["max_r"], 4),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
