"""Cycle-area and prefill-decode trade-off plots from the study cache and report.

Run: uv run --with matplotlib python results/abc-study/plot.py <report-dir> <out-dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from codesign.ventus.qwen_abc import design_scores  # noqa: E402
from codesign.ventus.qwen_study import Study  # noqa: E402
from codesign.ventus_costs.unified import evaluate_unified_area  # noqa: E402

report, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
summary = json.loads((report / "summary.json").read_text())
study = Study("results/abc-study/db")
scores = design_scores(study, ["Q-P128", "Q-P512", "Q-D128", "Q-D512"])
pts = []
for key, r in scores.items():
    hw = json.loads(key)
    area = evaluate_unified_area(hw)["total_area_mm2"]
    pre = [r[s] for s in ("Q-P128", "Q-P512") if s in r]
    dec = [r[s] for s in ("Q-D128", "Q-D512") if s in r]
    pts.append((area, max(pre) if pre else None, max(dec) if dec else None))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
colors = {"A": "#2563EB", "B": "#D97706", "C": "#059669"}
same = {
    t: [u for u in summary if summary[u]["joint"]["hardware"] == summary[t]["joint"]["hardware"]]
    for t in summary
}
label = {t: " = ".join(same[t]) for t in summary}
shown = set()
ax = axes[0]
for area, pre, dec in pts:
    if pre is not None:
        ax.scatter(area, pre, s=10, color="#93C5FD", alpha=0.6, lw=0)
    if dec is not None:
        ax.scatter(area, dec, s=10, color="#FCD34D", alpha=0.6, lw=0)
ax.scatter([], [], s=10, color="#93C5FD", label="simulated: prefill max r")
ax.scatter([], [], s=10, color="#FCD34D", label="simulated: decode max r")
for task, controls in summary.items():
    if label[task] in shown:
        continue
    shown.add(label[task])
    row = controls["joint"]
    ax.scatter(row["area_mm2"], row["task_max_r"], s=80, marker="*", color=colors[task], zorder=4)
    ax.annotate(f"{label[task]} joint", (row["area_mm2"], row["task_max_r"]), xytext=(6, -10),
                textcoords="offset points", fontsize=9, color=colors[task])
ax.scatter(1.103955, 1.0, s=60, marker="s", color="#475467", zorder=4)
ax.annotate("H0+S0", (1.103955, 1.0), xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.axvline(1.11, color="#B42318", ls="--", lw=1, label="1.11 mm² budget")
ax.set_yscale("log")
ax.set_xlabel("Unified area estimate (mm²)")
ax.set_ylabel("Cycle ratio r = T / T0 (log)")
ax.set_title("Cycle–area (sliced instruction simulation)")
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
both = [(p, d) for a, p, d in pts if p is not None and d is not None and a <= 1.11 + 1e-12]
ax.scatter([p for p, _ in both], [d for _, d in both], s=12, color="#98A2B3", alpha=0.7, lw=0,
           label="area-feasible simulated designs")
for task, controls in summary.items():
    for control, marker in (("joint", "*"), ("hardware-only", "^"), ("software-only", "v")):
        if control not in controls:
            continue
        row = controls[control]
        pre = max(row["ratios"]["Q-P128"], row["ratios"]["Q-P512"])
        dec = max(row["ratios"]["Q-D128"], row["ratios"]["Q-D512"])
        ax.scatter(pre, dec, s=90 if marker == "*" else 40, marker=marker, color=colors[task],
                   zorder=4,
                   label=f"{label[task]} joint" if control == "joint" and task == same[task][-1]
                   else None)
lim = [0, 1.05]
ax.plot(lim, lim, color="#D0D5DD", lw=1)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Prefill max r (P128, P512)")
ax.set_ylabel("Decode max r (D128, D512)")
ax.set_title("Prefill–decode trade-off (★ joint, ▲ HW-only, ▼ SW-only)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
for ext in ("png", "svg", "pdf"):
    fig.savefig(out / f"tradeoffs.{ext}", dpi=160)
print("saved", out)
