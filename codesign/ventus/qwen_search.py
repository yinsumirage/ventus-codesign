"""MIP-driven hardware/software search for the Qwen A/B/C study.

The MIP owns a fitted surrogate: per scenario, log cycles is an additive function
of one-hot hardware choices, selected pairwise products, and tile-specific terms
on the residency fields (the compiled tile kernels need 33 or 98 VGPRs per warp).
Unified area rows come from `add_unified_area_constraints`; compiled register
requirements become hard rows. The tile choice per phase is enumerated outside
the MIP (3 prefill x 2 decode tiles), so each solve is a pure linear MIP. Every
solution is decoded, checked independently (area, residency) and re-simulated by
`qwen_study`; the surrogate is then refitted with the new samples.
"""

import json
import math
from pathlib import Path

import numpy as np

from .qwen_study import BASELINE_T0, SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
SPACE = json.loads((ROOT / "examples/search-space-v3.json").read_text())
RANGES = dict(SPACE["active_budget_ranges"])
FIELDS = list(RANGES)
TILES = {"prefill": ("16x16", "16x64", "32x64"), "decode": ("16x16", "16x64")}
AUTO_TILE = {"prefill": "32x64", "decode": "16x64"}
GEMM_KERNEL = {"16x16": "mma_packed_bf16", "16x64": "mma_reuse_bf16", "32x64": "mma_reuse_bf16"}
# Compiled per-warp resources from the pinned bundle ELF metadata (see qwen_study).
KERNEL_RESOURCES = {
    "mma_reuse_bf16": dict(vgpr=98, sgpr=14, lds=128),
    "mma_packed_bf16": dict(vgpr=33, sgpr=11, lds=128),
    "addmm_mma_bf16": dict(vgpr=45, sgpr=22, lds=128),
    "pack_a_bf16": dict(vgpr=12, sgpr=12, lds=128),
    "rms_bf16": dict(vgpr=33, sgpr=11, lds=128),
    "causal_softmax_bf16": dict(vgpr=33, sgpr=11, lds=128),
}
# Pairwise products: SM replication multiplies per-SM memory/compute service.
PAIRS = [
    ("sms", "l1_mshrs"),
    ("sms", "tensor_units"),
    ("sms", "lsu_per_warp"),
    ("sms", "tensor_n"),
    ("l1_mshrs", "l2_mshrs"),
    ("l1_mshrs", "lsu_per_warp"),
    ("tensor_n", "tensor_units"),
    # Memory-request limits shift the bottleneck between each other: a factorial
    # around the A/B designs showed large joint effects with weak single effects.
    ("l1_write_entries", "l2_mshrs"),
    ("lsu_per_warp", "l2_mshrs"),
    ("l1_write_entries", "lsu_per_warp"),
]
TILE_FIELDS = ("vgpr_slots", "sgpr_slots", "blocks_per_sm", "warps_per_sm")
# Every compiled kernel reserves at most 128 B of LDS per block, so 16 KiB already
# admits 128 blocks. Simulated LDS capacity/bank/port sweeps around two designs gave
# bit-identical cycles, so these fields carry no surrogate term (only their area).
NO_TIMING_FIELDS = ("lds_bytes", "lds_banks", "lds_ports")
FEATURE_FIELDS = tuple(k for k in FIELDS if k not in NO_TIMING_FIELDS)
GEMM_CLASSES = {
    "prefill": ("q", "k", "v", "o", "gate", "up", "down", "qk", "pv", "lm_head"),
    "decode": ("qkv", "o", "gate_up", "down", "lm_head"),
}
SOFTWARE = dict(
    share_packing=True, gqa_batch=True, fuse_rows1=True, packed_attention=False, packed_pv=True
)


def phase_of(scenario):
    return SCENARIOS[scenario][0]


# Exact software dictionaries of the cached exploration/calibration samples, so equal
# programs share one simulation: S1 (sharing, GQA batching, decode fusion) and, for
# prefill, S3 = S1 + packed PV.
S1 = dict(share_packing=True, gqa_batch=True, fuse_rows1=True)
S3_PREFILL = dict(
    share_packing=True,
    gqa_batch=True,
    fuse_rows1=False,
    packed_attention=False,
    packed_pv=True,
    tiles={},
)
PHASE_IRRELEVANT = {"prefill": ("fuse_rows1",), "decode": ("packed_attention", "packed_pv")}


def software_for(phase, tile, packed_pv=True, order="row"):
    """Uniform phase tile (+ prefill packed PV, block order) -> software options."""
    base = dict(S3_PREFILL) if phase == "prefill" and packed_pv else dict(S1)
    if order != "row":
        base["block_order"] = order
    if tile != AUTO_TILE[phase]:
        base["tiles"] = {
            c: tile for c in GEMM_CLASSES[phase] if not (c == "lm_head" and tile == "16x64")
        }
    return base


# Tiles dominated in every earlier joint round (prefill 16x64, decode 16x16) are
# pruned from the default menu; MENU_TILES = TILES restores the full enumeration.
MENU_TILES = {"prefill": ("16x16", "32x64"), "decode": ("16x64",)}


def phase_menu(phase, tiles=None):
    """Joint-search software menu: tiles x packed PV x block order (prefill)."""
    tiles = (tiles or MENU_TILES)[phase]
    if phase == "decode":
        # One-row GEMMs have a single row tile, so block order cannot matter.
        return [software_for(phase, t) for t in tiles]
    return [
        software_for(phase, t, packed_pv=pv, order=order)
        for order in ("col", "row")
        for pv in (True, False)
        for t in tiles
    ]


def tile_of(phase, software):
    tiles = software.get("tiles", {})
    if not tiles:
        return AUTO_TILE[phase]
    values = {v for k, v in tiles.items() if k != "lm_head"}
    return values.pop() if len(values) == 1 else "mixed"


def variant_of(software, phase=None):
    """Software label without tiles; flags that cannot change the phase are ignored."""
    ignore = PHASE_IRRELEVANT.get(phase, ())
    labels = [k for k in sorted(SOFTWARE) if software.get(k) is True and k not in ignore]
    if software.get("block_order", "row") != "row":
        labels.append("order=" + software["block_order"])
    return ",".join(labels)


def residency(hw, kernel):
    r = KERNEL_RESOURCES[kernel]
    return min(
        hw["blocks_per_sm"],
        hw["warps_per_sm"],
        hw["vgpr_slots"] // r["vgpr"],
        hw["sgpr_slots"] // r["sgpr"],
        hw["lds_bytes"] // r["lds"],
    )


def features(hw, tile, variant):
    f = {("f", k, hw[k]): 1.0 for k in FEATURE_FIELDS}
    for a, b in PAIRS:
        f[("p", a, hw[a], b, hw[b])] = 1.0
    f[("t", tile)] = 1.0
    f[("s", variant)] = 1.0
    for k in TILE_FIELDS:
        f[("tf", tile, k, hw[k])] = 1.0
    return f


def fit(samples, scenario, ridge="auto", prior=None, prior_ridge=1.0):
    """Ridge regression of log cycles; with a prior, shrink toward its coefficients.

    ridge="auto" picks the strength with the smallest leave-one-out log error.
    """
    if ridge == "auto":
        fits = [
            fit(samples, scenario, r, prior, prior_ridge if prior is None else r * 10)
            for r in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
        ]
        return min(fits, key=lambda m: m["loo_rms_log_error"])
    rows = [
        (features(s["hardware"], s["tile"], s["variant"]), math.log(s["cycles"])) for s in samples
    ]
    keys = sorted({k for f, _ in rows for k in f} | set(prior["coef"] if prior else ()), key=str)
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys) + 1
    X = np.zeros((len(rows), n))
    y = np.zeros(len(rows))
    for i, (f, v) in enumerate(rows):
        X[i, -1] = 1.0
        for k, x in f.items():
            X[i, index[k]] = x
        y[i] = v
    b0 = np.zeros(n)
    lam = np.full(n, ridge)
    lam[-1] = 0.0
    if prior:
        for k, c in prior["coef"].items():
            b0[index[k]] = c
        b0[-1] = prior["intercept"]
        lam[:-1] = prior_ridge
        lam[-1] = 1e-6
    A = X.T @ X + np.diag(lam)
    coef = np.linalg.solve(A, X.T @ y + lam * b0)
    resid = y - X @ coef
    hat = np.einsum("ij,ji->i", X, np.linalg.solve(A, X.T))
    loo = resid / np.maximum(1 - hat, 1e-6)
    return dict(
        scenario=scenario,
        coef={k: float(coef[index[k]]) for k in keys},
        intercept=float(coef[-1]),
        samples=len(rows),
        rms_log_error=float(np.sqrt(np.mean(resid**2))),
        loo_rms_log_error=float(np.sqrt(np.mean(loo**2))),
        loo_max_abs_log_error=float(np.max(np.abs(loo))),
        ridge=ridge,
    )


def predict(model, hw, tile, variant):
    total = model["intercept"]
    for k in features(hw, tile, variant):
        total += model["coef"].get(k, 0.0)
    return math.exp(total)


def new_highs(time_limit):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-4)
    # Same convention as codesign.solver: HiGHS fixes its thread pool process-wide
    # at first use, so every solver in the process must request the same count.
    h.setOptionValue("random_seed", 0)
    h.setOptionValue("threads", 1)
    return h


def area_model(highs, fixed=None):
    from codesign.ventus_costs.unified import add_unified_area_constraints

    binding = add_unified_area_constraints(highs, fixed=fixed)
    for key, cols in binding["fields"].items():
        for value, col in cols.items():
            if key in RANGES and value not in RANGES[key]:
                highs.changeColBounds(col, 0, 0)
    area = {}
    for columns in binding["groups"].values():
        for col, choice in columns:
            area[col] = (choice["logic_area"] + choice["storage_area_um2"]) / 1e6
    return binding, area


def min_area_completion(fixed, time_limit=30):
    """Smallest-area legal hardware with the given fields fixed."""
    from codesign.ventus_costs.unified import decode_unified_solution

    h = new_highs(time_limit)
    binding, area = area_model(h, fixed=fixed)
    for col, a in area.items():
        h.changeColCost(col, a)
    h.run()
    if h.getInfo().primal_solution_status != 2:
        return None
    sol = decode_unified_solution(binding, list(h.getSolution().col_value))
    return sol["hardware"], sol["cost"]["total_area_mm2"]


def solve(
    models,
    scenarios,
    tiles,
    variants,
    *,
    exclude=(),
    starts=(),
    trust=None,
    time_limit=60,
    unseen_penalty=0.05,
):
    """min t s.t. surrogate log r(w) <= t for each scenario, under area and residency.

    tiles/variants: phase -> fixed uniform tile / software label. exclude: hardware
    assignments cut off by no-good rows. starts: candidate feasible hardware; the
    first one that is not excluded and satisfies the compiled-resource rows becomes
    a complete MIP start (HiGHS otherwise rarely finds an incumbent in this model).
    trust: optional (centre hardware, max changed fields) local-search region.
    """
    from highspy import HighsVarType

    from codesign.ventus_costs.unified import decode_unified_solution

    h = new_highs(time_limit)
    binding, area = area_model(h)
    fields = binding["fields"]
    inf = math.inf

    def binary():
        col = h.getNumCol()
        h.addCol(0, 0, 1, 0, [], [])
        h.changeColIntegrality(col, HighsVarType.kInteger)
        return col

    def row(terms, lo, hi):
        h.addRow(lo, hi, len(terms), list(terms), list(terms.values()))

    # Hard compiled-resource rows: every kernel used must keep at least one block.
    kernels = {"addmm_mma_bf16", "pack_a_bf16", "rms_bf16", "causal_softmax_bf16"}
    kernels |= {GEMM_KERNEL[tiles[phase_of(s)]] for s in scenarios}
    for kernel in kernels:
        r = KERNEL_RESOURCES[kernel]
        for field, need in (("vgpr_slots", r["vgpr"]), ("sgpr_slots", r["sgpr"])):
            for v, c in fields[field].items():
                if v < need:
                    h.changeColBounds(c, 0, 0)
    pair_cols = {}
    for a, b in PAIRS:
        for va, ca in fields[a].items():
            for vb, cb in fields[b].items():
                z = binary()
                pair_cols[(a, va, b, vb)] = z
                row({z: 1.0, ca: -1.0}, -inf, 0)
                row({z: 1.0, cb: -1.0}, -inf, 0)
                row({z: 1.0, ca: -1.0, cb: -1.0}, -1, inf)
    t = h.getNumCol()
    h.addCol(1.0, -inf, inf, 0, [], [])
    scenario_rows = []
    for scenario in scenarios:
        m = models[scenario]
        tile, variant = tiles[phase_of(scenario)], variants[phase_of(scenario)]
        terms = {t: -1.0}
        const = m["intercept"] - math.log(BASELINE_T0[scenario])
        const += m["coef"].get(("t", tile), 0.0) + m["coef"].get(("s", variant), 0.0)
        seen = set()
        for key, c in m["coef"].items():
            col = None
            if key[0] == "f":
                col = fields[key[1]].get(key[2])
                seen.add((key[1], key[2]))
            elif key[0] == "p":
                col = pair_cols.get(key[1:])
            elif key[0] == "tf" and key[1] == tile:
                col = fields[key[2]].get(key[3])
            if col is not None:
                terms[col] = terms.get(col, 0.0) + c
        for k, cols in fields.items():
            for v, col in cols.items():
                if k in FEATURE_FIELDS and v in RANGES[k] and (k, v) not in seen:
                    terms[col] = terms.get(col, 0.0) + unseen_penalty
        row(terms, -inf, -const)
        scenario_rows.append((terms, const))
    for hw in exclude:
        cols = [fields[k][hw[k]] for k in FIELDS if hw[k] in fields[k]]
        row(dict.fromkeys(cols, 1.0), -inf, len(cols) - 1)
    if trust is not None:
        centre, radius = trust
        row({fields[k][centre[k]]: 1.0 for k in FIELDS}, len(FIELDS) - radius, inf)
    excluded = [{k: hw[k] for k in FIELDS} for hw in exclude]
    lower = list(h.getLp().col_lower_)
    upper = list(h.getLp().col_upper_)
    candidates = list(starts)
    if trust is not None:
        from codesign.ventus_costs.unified import evaluate_unified_area

        centre, radius = trust
        near = [c for c in candidates if sum(c[k] != centre[k] for k in FIELDS) <= radius]
        moves = [dict(centre, **{k: v}) for k, v in neighbours(centre)]
        moves = [m for m in moves if _valid(m) and evaluate_unified_area(m)["area_feasible"]]
        candidates = near + moves
    for start in candidates:
        if {k: start[k] for k in FIELDS} in excluded:
            continue
        x = start_vector(binding, pair_cols, start, h.getNumCol())
        if x is None or any(not lower[i] - 1e-9 <= v <= upper[i] + 1e-9 for i, v in enumerate(x)):
            continue
        x[t] = max(
            sum(c * x[col] for col, c in terms.items() if col != t) + const
            for terms, const in scenario_rows
        )
        import highspy

        sol = highspy.HighsSolution()
        sol.col_value = x
        sol.value_valid = True
        h.setSolution(sol)
        break
    h.run()
    info = h.getInfo()
    status = h.getModelStatus()
    result = dict(
        status=h.modelStatusToString(status),
        tiles=dict(tiles),
        variants=dict(variants),
        columns=h.getNumCol(),
        rows=h.getNumRow(),
        mip_gap=info.mip_gap if math.isfinite(info.mip_gap) else None,
        dual_bound_log_r=info.mip_dual_bound if math.isfinite(info.mip_dual_bound) else None,
        solver_seconds=h.getRunTime(),
    )
    if info.primal_solution_status != 2:
        return dict(result, hardware=None)
    values = list(h.getSolution().col_value)
    sol = decode_unified_solution(binding, values)
    hw = sol["hardware"]
    return dict(
        result,
        objective_log_r=values[t],
        predicted_max_r=math.exp(values[t]),
        hardware=hw,
        area_mm2=sol["cost"]["total_area_mm2"],
        predicted={
            s: predict(models[s], hw, tiles[phase_of(s)], variants[phase_of(s)]) / BASELINE_T0[s]
            for s in scenarios
        },
    )


def _valid(hw):
    from .config import Hardware

    try:
        Hardware(**hw).validate()
    except ValueError:
        return False
    return True


def start_vector(binding, pair_cols, hw, ncols):
    """Complete column assignment for one hardware point (fields, groups, pairs)."""
    x = [0.0] * ncols
    for key, cols in binding["fields"].items():
        if hw.get(key) not in cols:
            return None
        x[cols[hw[key]]] = 1.0
    for columns in binding["groups"].values():
        match = [col for col, c in columns if all(hw[k] == v for k, v in c["choice"].items())]
        if len(match) != 1:
            return None
        x[match[0]] = 1.0
    for (a, va, b, vb), z in pair_cols.items():
        x[z] = float(hw[a] == va and hw[b] == vb)
    return x


def neighbours(hw):
    """Single-field moves to adjacent menu levels (local refinement around a design)."""
    for k in FIELDS:
        values = RANGES[k]
        i = values.index(hw[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                yield k, values[j]
