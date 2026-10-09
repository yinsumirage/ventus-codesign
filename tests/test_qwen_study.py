"""Sliced-estimate accounting for the Qwen study (no simulation)."""

from codesign.ventus import qwen_study as qst


def row(name, cycles):
    return dict(name=name, cycles=cycles, instructions=0, requested_bytes=0, counters={})


def test_v1_decode_counts_its_separate_prefill_lm_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(qst, "ESTIMATOR", "v1")
    study = qst.Study(tmp_path)
    hw, sw = qst.full_hardware({}), {}
    body = [
        row("decode08.embedding", 5),
        row("decode08.layer00.gate", 10),
        row("decode08.final_norm", 1),
    ]
    lm = [row("prefill.lm_head.pack_activation", 2), row("prefill.lm_head", 100)]
    study.record(study.part_key(hw, "Q-D128", sw, "body"), hw, "Q-D128", sw, "body", body, 0)
    study.record(study.part_key(hw, "Q-D128", sw, "lm"), hw, "Q-D128", sw, "lm", lm, 0)
    est = study.estimate(hw, sw, "Q-D128")
    assert est["cycles"] == 5 + 24 * 10 + 1 + 2 + 100


def test_v2_decode_skips_warm_up_and_scales_sampled_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(qst, "ESTIMATOR", "v2")
    study = qst.Study(tmp_path)
    hw, sw = qst.full_hardware({}), {}
    body = [
        row("decode07.layer23.gate", 1000),  # warm-up only
        row("decode07.lm_head", 1000),
        row("decode08.embedding", 5),
        *[row(f"decode08.layer0{i}.gate", 10 + i) for i in range(3)],
        row("decode08.final_norm", 1),
        row("decode08.lm_head", 100),
    ]
    study.record(study.part_key(hw, "Q-D128", sw, "body"), hw, "Q-D128", sw, "body", body, 0)
    assert study.jobs(hw, sw, ["Q-D128"]) == {}  # no separate LM slice in v2 decode
    est = study.estimate(hw, sw, "Q-D128")
    assert est["cycles"] == 5 + 8 * (10 + 11 + 12) + 1 + 100
