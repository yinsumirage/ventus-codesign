"""Surrogate fitting and MIP selection checks for the Qwen A/B/C search."""

import json
import math
from pathlib import Path

import numpy as np

from codesign.ventus import qwen_search as qs
from codesign.ventus.qwen_study import BASELINE_T0, full_hardware, residency
from codesign.ventus_costs.unified import evaluate_unified_area

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "examples/baseline-hardware-v1.json").read_text())


def synthetic_samples(scenario, n=80, seed=0):
    """log cycles = additive truth over sms, l1_mshrs and tile-specific VGPR terms."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        hw = dict(BASE)
        for k in ("sms", "l1_mshrs", "vgpr_slots", "tensor_n", "l2_mshrs"):
            hw[k] = int(rng.choice(qs.RANGES[k]))
        tile = str(rng.choice(["16x16", "32x64"]))
        log_c = (
            math.log(BASELINE_T0[scenario])
            - 0.5 * math.log(hw["sms"] / 2)
            - 0.2 * math.log(hw["l1_mshrs"] / 4)
            + (0.3 if tile == "32x64" and hw["vgpr_slots"] < 1024 else 0.0)
        )
        out.append(dict(hardware=hw, tile=tile, variant="v", cycles=math.exp(log_c)))
    return out


def test_fit_recovers_additive_and_tile_coupled_effects():
    rows = synthetic_samples("Q-P128")
    model = qs.fit(rows, "Q-P128", ridge=1e-6)
    assert model["rms_log_error"] < 1e-3
    a = dict(BASE, sms=4, l1_mshrs=16, vgpr_slots=512)
    b = dict(a, vgpr_slots=2048)
    # The 98-VGPR tile pays for small register files; the 33-VGPR tile does not.
    assert qs.predict(model, a, "32x64", "v") / qs.predict(model, b, "32x64", "v") > 1.3
    assert abs(qs.predict(model, a, "16x16", "v") / qs.predict(model, b, "16x16", "v") - 1) < 0.02


def test_solve_respects_area_residency_and_no_good_cuts():
    rows = synthetic_samples("Q-P128") + synthetic_samples("Q-D128", seed=1)
    models = {
        "Q-P128": qs.fit([r for r in rows[:80]], "Q-P128"),
        "Q-D128": qs.fit([r for r in rows[80:]], "Q-D128"),
    }
    tiles = {"prefill": "32x64", "decode": "16x64"}
    variants = {"prefill": "v", "decode": "v"}
    starts = [
        full_hardware(change)
        for change in ({}, {"lds_bytes": 16384}, {"l1_mshrs": 8}, {"sgpr_slots": 256})
    ]
    first = qs.solve(models, ["Q-P128", "Q-D128"], tiles, variants, starts=starts, time_limit=30)
    hw = first["hardware"]
    assert hw is not None
    full = full_hardware({k: hw[k] for k in qs.FIELDS})
    assert evaluate_unified_area(full)["area_feasible"]
    assert residency(full, dict(vgpr=98, sgpr=14, lds=128)) >= 1
    assert residency(full, dict(vgpr=45, sgpr=22, lds=128)) >= 1
    second = qs.solve(
        models, ["Q-P128", "Q-D128"], tiles, variants, exclude=[hw], starts=starts, time_limit=30
    )
    assert second["hardware"] is not None
    assert any(second["hardware"][k] != hw[k] for k in qs.FIELDS)
    if first["status"] == "Optimal":
        # A cut can only worsen a proven optimum; time-limited solves prove nothing.
        assert second["objective_log_r"] >= first["objective_log_r"] - 1e-6


def test_software_for_matches_cached_programs_and_labels():
    pre = qs.software_for("prefill", "32x64")
    dec = qs.software_for("decode", "16x64")
    assert pre == qs.S3_PREFILL and dec == qs.S1
    assert qs.software_for("prefill", "32x64", packed_pv=False) == qs.S1
    assert qs.tile_of("prefill", qs.software_for("prefill", "16x16")) == "16x16"
    assert "lm_head" not in qs.software_for("prefill", "16x64")["tiles"]
    # Fusion cannot change prefill programs, packed PV cannot change decode programs.
    assert qs.variant_of(qs.S1, "prefill") == "gqa_batch,share_packing"
    assert qs.variant_of(qs.S1, "decode") == "fuse_rows1,gqa_batch,share_packing"
    assert len(qs.phase_menu("prefill", qs.TILES)) == 16
    assert "lm_head" not in qs.software_for("prefill", "64x64")["tiles"]
    assert len(qs.phase_menu("decode", qs.TILES)) == 3
    trim = qs.software_for("decode", "16x64", order="col", epilogue="trim")
    assert qs.variant_of(trim, "decode").endswith("epilogue=trim")
    col = qs.software_for("prefill", "32x64", order="col")
    assert col["block_order"] == "col" and qs.variant_of(col, "prefill").endswith("order=col")
