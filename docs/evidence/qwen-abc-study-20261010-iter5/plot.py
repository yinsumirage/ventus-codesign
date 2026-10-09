"""Task max r per iteration (complete runs) and the sliced design points of iterations 4-5.

Run from the repository root: python docs/evidence/qwen-abc-study-20261010-iter5/plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent

# Complete-simulation task scores; iteration 1 had complete runs of the 128 scenarios only.
HISTORY = {
    "1": {"A": 0.367, "B": 0.313, "C": 0.378},
    "2": {"A": 0.367, "B": 0.279, "C": 0.367},
    "3": {"A": 0.342, "B": 0.279, "C": 0.342},
    "4": {"A": 0.260, "B": 0.269, "C": 0.285},
}
final = json.loads((HERE / "summary.json").read_text())
HISTORY["5"] = {t: final[t]["max_r"] for t in ("A", "B", "C")}

points = {"Q-P128": [], "Q-D128": []}
for folder in (EVIDENCE / "qwen-abc-study-20261010-iter4", HERE):
    for path in sorted((folder / "experiments").glob("*.json")):
        for row in json.loads(path.read_text()):
            if not isinstance(row, dict) or "scenarios" not in row:
                continue
            for s in points:
                if s in row["scenarios"]:
                    r = row["scenarios"][s]["r_vs_frozen_T0"]
                    points[s].append((row["area_mm2"], r, "iter5" in str(folder)))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
colors = {"A": "#2563EB", "B": "#D97706", "C": "#059669"}
iters = list(HISTORY)
for t, c in colors.items():
    ys = [HISTORY[i][t] for i in iters]
    left.plot(iters, ys, marker="o", color=c, label=f"Task {t}")
    dy = {"A": 0, "B": -7, "C": 7}[t]
    left.annotate(
        f"{ys[-1]:.3f}",
        (iters[-1], ys[-1]),
        xytext=(6, dy),
        textcoords="offset points",
        color=c,
        va="center",
    )
left.set_xlabel("Iteration")
left.set_ylabel("Task max r (complete simulation)")
left.set_title("Task scores (lower is better; T0 = 1)")
left.legend(frameon=False)
left.set_ylim(0.2, 0.4)

for s, c, name in (("Q-P128", "#2563EB", "Q-P128"), ("Q-D128", "#D97706", "Q-D128")):
    xs = [a for a, r, _ in points[s] if r < 0.45 and a > 0.9]
    ys = [r for a, r, _ in points[s] if r < 0.45 and a > 0.9]
    right.scatter(xs, ys, s=12, alpha=0.45, color=c, label=f"{name} (sliced)")
offsets = {"A": (-14, -12), "B": (6, -4), "C": (6, 6)}
for t, marker in (("A", "*"), ("B", "^"), ("C", "s")):
    d = final[t]
    for s, c in (("Q-P128", "#2563EB"), ("Q-D128", "#D97706")):
        if s in d["complete"]:
            right.scatter(
                [d["area_mm2"]],
                [d["complete"][s]],
                marker=marker,
                s=90,
                color=c,
                edgecolor="black",
                zorder=3,
            )
    right.annotate(
        t,
        (d["area_mm2"], max(d["complete"].values())),
        xytext=offsets[t],
        textcoords="offset points",
        fontweight="bold",
    )
right.axvline(1.11, color="grey", lw=0.8, ls="--")
right.text(1.1105, 0.43, "1.11 mm²", color="grey", fontsize=8)
right.set_xlim(0.93, 1.125)
right.set_xlabel("Unified-v4 area (mm²)")
right.set_ylabel("r = T / T0")
right.set_title("Iterations 4–5 design points; markers: selected (complete)")
right.legend(frameon=False, loc="upper left")

for ext in ("png", "svg", "pdf"):
    fig.savefig(HERE / f"progress.{ext}", dpi=160)
print("written", file=sys.stderr)
