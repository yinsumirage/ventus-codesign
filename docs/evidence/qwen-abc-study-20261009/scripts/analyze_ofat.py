"""Main effects around each OFAT centre from the cached slices."""
import json, sys
sys.path.insert(0, ".")
from codesign.ventus.qwen_study import Study, BASELINE_T0, full_hardware
from codesign.ventus.qwen_search import RANGES
from codesign.ventus_costs.unified import evaluate_unified_area
study = Study("results/abc-study/db")
designs = json.load(open("results/abc-study/designs/ofat1.json"))
SW = designs[0][1]
def r(hw):
    try:
        h = full_hardware(hw)
    except ValueError:
        return {"Q-P128": None, "Q-D128": None}
    out = {}
    for s in ("Q-P128", "Q-D128"):
        e = study.estimate(h, SW, s)
        out[s] = None if e is None else e["cycles"] / BASELINE_T0[s]
    return out
centres = {"C2": designs[0][0]}
for d in designs:
    pass
# centres are the first design of each block: detect by sms changes in generator order
idx = [0]
prev = designs[0][0]
names = ["C2", "C3", "C4"]
cs = [designs[0][0]]
for d in designs[1:]:
    diff = [k for k in RANGES if d[0][k] != cs[-1][k]]
    if len(diff) > 1:
        cs.append(d[0])
for name, c in zip(names, cs):
    base = r(c)
    a = evaluate_unified_area(full_hardware(c))["total_area_mm2"]
    print(f"== {name} area={a:.4f} P128={base['Q-P128']} D128={base['Q-D128']}")
    if base["Q-P128"] is None: continue
    for k, vals in RANGES.items():
        row = []
        for v in vals:
            if v == c[k]:
                row.append(f"{v}:*"); continue
            x = r(dict(c, **{k: v}))
            fmt = lambda s: "-" if x[s] is None or base[s] is None else f"{x[s]/base[s]:.3f}"
            row.append(f"{v}:{fmt('Q-P128')}/{fmt('Q-D128')}")
        print(f"  {k:17s} " + "  ".join(row))
