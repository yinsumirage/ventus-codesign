"""Search effort and solver-status statistics from the study cache and round receipts."""

import collections
import glob
import json
import sys

sys.path.insert(0, ".")
from codesign.ventus.qwen_study import Study  # noqa: E402

study = Study("results/abc-study/db")
slices = list(study.db.values())
designs = {json.dumps(r["hardware"], sort_keys=True) for r in slices}
host = sum(r["host_seconds"] for r in slices)
by_part = collections.Counter(
    (r["part"], r["scenario"]) if r["part"] == "body" else ("lm", None) for r in slices
)
out = dict(
    simulated_slices=len(slices),
    distinct_hardware=len(designs),
    simulation_host_seconds=round(host, 1),
    slices_by_scenario={f"{p}:{s}": n for (p, s), n in sorted(by_part.items(), key=str)},
    rounds={},
)
for path in sorted(glob.glob("results/abc-study/db/rounds-*.jsonl")):
    name = path.split("rounds-")[1].removesuffix(".jsonl")
    rows = [json.loads(line) for line in open(path)]
    statuses = collections.Counter(s["status"] for r in rows for s in r["solves"])
    out["rounds"][name] = dict(
        rounds=len(rows),
        mip_solves=sum(len(r["solves"]) for r in rows),
        statuses=dict(statuses),
        trust_region_solves=sum(1 for r in rows for s in r["solves"] if s.get("trust_radius")),
        simulated_designs=sum(len(r["simulated"]) for r in rows),
        host_seconds=round(sum(r["host_seconds"] for r in rows), 1),
        best_max_r=[round(r["best_known"]["max_r"], 4) if r["best_known"] else None for r in rows],
        solver_seconds=round(sum(s.get("solver_seconds", 0) for r in rows for s in r["solves"]), 1),
        mean_abs_prediction_error=round(
            sum(
                abs(s["predicted"][k] / s["ratios"][k] - 1)
                for r in rows
                for s in r["simulated"]
                for k in s["ratios"]
                if k in s.get("predicted", {})
            )
            / max(1, sum(len(s["ratios"]) for r in rows for s in r["simulated"])),
            4,
        ),
    )
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rounds"}, indent=1))
for k, v in out["rounds"].items():
    print(k, v["rounds"], v["mip_solves"], v["statuses"], v["best_max_r"], v["mean_abs_prediction_error"])
