"""Build the report selection (best design per task and control) from the study cache.

Run inside the execution environment: python results/abc-study/make_selection.py OUT.json
"""

import json
import sys

sys.path.insert(0, ".")
from codesign.ventus import qwen_search as qs  # noqa: E402
from codesign.ventus.qwen_abc import best_known, policy_softwares  # noqa: E402
from codesign.ventus.qwen_study import TASKS, Study  # noqa: E402

FIXED = ("threads", "line_bytes", "memory_channels", "memory_bytes_per_cycle", "memory_latency")
study = Study("results/abc-study/db")
seq = json.load(open("results/abc-study/designs/sequential-software-v2.json"))
full_menu = {p: qs.phase_menu(p, qs.TILES) + [qs.S1] for p in ("prefill", "decode")}
joint_menu = {p: qs.phase_menu(p) + [qs.S1] for p in ("prefill", "decode")}
s0 = {"prefill": [{}], "decode": [{}]}


def best(task, policy, fixed=None):
    phases = sorted({qs.phase_of(s) for s in TASKS[task]})
    menus = {p: policy_softwares(policy, p, fixed) for p in phases}
    if policy == "joint":
        menus = {p: joint_menu[p] for p in phases}
    allowed = [dict(zip(phases, combo)) for combo in __import__("itertools").product(
        *(menus[p] for p in phases))]
    b = best_known(study, TASKS[task], allowed=allowed)
    return {k: v for k, v in b["hardware"].items() if k not in FIXED}, b["max_r"]


selection = {}
for task in ("A", "B", "C"):
    hw_joint, rj = best(task, "joint")
    hw_hw, rh = best(task, "hardware-only")
    seq_fixed = seq if task != "B" else json.load(
        open("results/abc-study/designs/sequential-software.json"))
    hw_seq, rs = best(task, "sequential", seq_fixed)
    selection[task] = {
        "default": {"hardware": {}, "menu": s0},
        "software-only": {"hardware": {}, "menu": full_menu},
        "hardware-only": {"hardware": hw_hw, "menu": s0},
        "sequential": {
            "hardware": hw_seq,
            "menu": {p: [seq_fixed[p]] for p in ("prefill", "decode")},
        },
        "joint": {"hardware": hw_joint, "menu": joint_menu},
    }
    print(task, "joint", round(rj, 4), "hw-only", round(rh, 4), "seq", round(rs, 4))
json.dump(selection, open(sys.argv[1], "w"), indent=1)
