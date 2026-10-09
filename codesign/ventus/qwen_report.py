"""Cross-evaluate selected A/B/C designs and the five controls from cached slices.

A selection file lists, per task and control, one hardware dict and a software
menu per phase. The task fixes hardware but lets software follow the kernel,
shape and scenario, so each scenario uses the fastest simulated program of its
phase menu on that hardware (the chosen program is recorded). Missing slices are
simulated first; every design is evaluated on all four scenarios. Ratios divide
by the frozen default T0; the 512 T0 values are the baseline's labelled
extrapolations.
"""

import argparse
import json
from pathlib import Path

from .qwen_study import BASELINE_T0, SCENARIOS, TASKS, Study, evaluate_designs, full_hardware

CONTROLS = ("default", "software-only", "hardware-only", "sequential", "joint")


def evaluate_selection(study_dir, selection, workers):
    """selection: {task: {control: {"hardware": {...}, "menu": {phase: [software...]}}}}."""
    from codesign.ventus_costs.unified import evaluate_unified_area

    batches = {"prefill": [], "decode": []}
    for task in selection.values():
        for design in task.values():
            for phase, items in batches.items():
                items.extend([design["hardware"], sw] for sw in design["menu"][phase])
    for phase, items in batches.items():
        scen = [s for s, (p, _) in SCENARIOS.items() if p == phase]
        evaluate_designs(study_dir, items, scen, workers)
    study = Study(study_dir)
    out = {}
    for task, controls in selection.items():
        out[task] = {}
        for control, design in controls.items():
            hw = full_hardware(design["hardware"])
            row = dict(hardware=hw, software={}, cycles={}, ratios={}, classes={}, menu_ratios={})
            for scenario, (phase, _) in SCENARIOS.items():
                options = []
                for sw in design["menu"][phase]:
                    est = study.estimate(hw, sw, scenario)
                    options.append((est["cycles"], sw, est))
                cycles, sw, est = min(options, key=lambda x: x[0])
                row["software"][scenario] = sw
                row["cycles"][scenario] = cycles
                row["ratios"][scenario] = cycles / BASELINE_T0[scenario]
                row["classes"][scenario] = est["classes"]
                row["menu_ratios"][scenario] = [c / BASELINE_T0[scenario] for c, _, _ in options]
            row["task_max_r"] = max(row["ratios"][s] for s in TASKS[task])
            row["all_max_r"] = max(row["ratios"].values())
            row["area_mm2"] = evaluate_unified_area(hw)["total_area_mm2"]
            out[task][control] = row
    return out


def markdown(result):
    lines = [
        "| Design | Area mm² | Q-P128 | Q-P512 | Q-D128 | Q-D512 | Task max r | All-four max r |"
    ]
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for task, controls in result.items():
        row = controls["joint"]
        r = row["ratios"]
        lines.append(
            f"| {task} joint | {row['area_mm2']:.4f} | "
            + " | ".join(f"{r[s]:.4f}" for s in SCENARIOS)
            + f" | **{row['task_max_r']:.4f}** | {row['all_max_r']:.4f} |"
        )
    lines.append("")
    lines.append("| Task (max r) | " + " | ".join(CONTROLS) + " |")
    lines.append("|---|" + "---:|" * len(CONTROLS))
    for task, controls in result.items():
        cells = [f"{controls[c]['task_max_r']:.4f}" if c in controls else "-" for c in CONTROLS]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("output directory exists; results are never overwritten")
    selection = json.loads(args.selection.read_text())
    result = evaluate_selection(args.study, selection, args.workers)
    args.out.mkdir(parents=True)
    (args.out / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    (args.out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    (args.out / "tables.md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
