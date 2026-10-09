"""Functional checks for packed64 software options using the NumPy dispatch replay."""

import json
from pathlib import Path

import numpy as np
import pytest

from codesign.ventus.model_program import qwen_manifest
from codesign.ventus.qwen_reference import replay

BUNDLE = Path(__file__).parent / "fixtures/compiled-transformer/bundle.json"
NCOL = Path(__file__).parent / "fixtures/compiled-transformer-ncol/bundle.json"
ALL = dict(share_packing=True, gqa_batch=True, fuse_rows1=True, packed_attention=True)


def build(tmp_path, name, phase, context, software=None, steps=2):
    out = tmp_path / name
    col = (software or {}).get("block_order") == "col"
    qwen_manifest(
        NCOL if col else BUNDLE,
        out,
        phase=phase,
        context=context,
        steps=steps,
        small=True,
        mapping="packed64",
        software=software,
    )
    return out / "program.json"


def dispatch_view(path):
    return json.loads(path.read_text())["dispatches"]


def test_empty_options_keep_packed64_dispatches(tmp_path):
    for phase in ("prefill", "decode"):
        a = build(tmp_path, f"{phase}-none", phase, 32)
        b = build(tmp_path, f"{phase}-empty", phase, 32, software={})
        assert dispatch_view(a) == dispatch_view(b)


@pytest.mark.parametrize(
    "context,software",
    [
        (32, dict(ALL)),
        (32, dict(ALL, tiles={"q": "16x16", "gate": "32x64", "up": "16x64", "down": "16x16"})),
        (64, dict(ALL, gqa_batch=False, tiles={"qk": "32x64", "pv": "16x16"})),
        (64, dict(ALL, tiles={"qk": "16x64"})),
        (32, dict(share_packing=True, tiles={"lm_head": "16x16", "o": "32x64"})),
        (32, dict(share_packing=True, gqa_batch=True, packed_pv=True)),
        (64, dict(packed_pv=True, tiles={"pv": "16x16", "qk": "16x16"})),
        (32, dict(ALL, block_order="col")),
        (64, dict(ALL, block_order="col", tiles={"gate": "16x16", "o": "16x64", "qk": "32x64"})),
    ],
)
def test_prefill_options_preserve_logits_and_kv(tmp_path, context, software):
    base = replay(build(tmp_path, "base", "prefill", context))
    new = replay(build(tmp_path, "new", "prefill", context, software=software))
    assert set(base) == set(new)
    np.testing.assert_allclose(new["logits"], base["logits"], rtol=2e-2, atol=2e-2)
    assert np.corrcoef(new["logits"], base["logits"])[0, 1] > 0.999
    for name in base:
        if name != "logits":
            np.testing.assert_array_equal(new[name], base[name])


@pytest.mark.parametrize(
    "software",
    [
        dict(ALL),
        dict(ALL, tiles={"qkv": "16x16", "gate_up": "16x64", "lm_head": "16x16"}),
        dict(gqa_batch=True),
        dict(ALL, block_order="col", tiles={"qkv": "16x16"}),
    ],
)
def test_decode_options_preserve_logits_and_kv(tmp_path, software):
    base = replay(build(tmp_path, "base", "decode", 5))
    new = replay(build(tmp_path, "new", "decode", 5, software=software))
    np.testing.assert_allclose(new["logits"], base["logits"], rtol=2e-2, atol=2e-2)
    assert np.corrcoef(new["logits"], base["logits"])[0, 1] > 0.999
    for name in base:
        if name != "logits":
            np.testing.assert_array_equal(new[name], base[name])


def test_replay_matches_dense_reference_for_one_gemm(tmp_path):
    """The replay's packed GEMM equals a dense BF16 product (tails included)."""
    from codesign.ventus.qwen_reference import (
        KERNELS,
        Memory,
        from_bf16,
        pack_b_tiles,
        to_bf16,
    )

    rng = np.random.default_rng(3)
    m, n, k = 20, 128, 40
    a = to_bf16(rng.standard_normal((m, k)))
    w = to_bf16(rng.standard_normal((n, k)))
    bufs = [
        dict(address=0x10000000, end=0x10000000 + 2 * m * k),
        dict(address=0x10010000, end=0x10010000 + 2 * 2 * 3 * 256),
        dict(address=0x10020000, end=0x10020000 + 2 * m * n),
        dict(address=0x90000000, end=0x90000000 + 2 * 8 * 3 * 256),
    ]
    mem = Memory(bufs)
    mem.u16(0x10000000, m * k)[:] = a.reshape(-1)
    tiles = pack_b_tiles(w)
    mem.u16(0x90000000, tiles.size)[:] = tiles
    KERNELS["pack_a_bf16"](mem, 2 * 3 * 32, 0x10000000, 0x10010000, m, k)
    expect = from_bf16(a).astype(np.float64) @ from_bf16(w).astype(np.float64).T
    for kernel, args, items in [
        ("mma_packed_bf16", (0, m, n, k, 0), 2 * 8 * 32),
        ("mma_reuse_bf16", (0, m, n, k, 0, 1), 1 * 2 * 32),
        ("mma_reuse_bf16", (0, m, n, k, 0, 0), 2 * 2 * 32),
    ]:
        mem.u16(0x10020000, m * n)[:] = 0
        KERNELS[kernel](mem, items, 0x10010000, 0x90000000, args[0], 0x10020000, *args[1:])
        got = from_bf16(mem.u16(0x10020000, m * n)).reshape(m, n)
        np.testing.assert_allclose(got, expect, rtol=1e-2, atol=1e-2)


def test_column_order_bundle_reuses_frozen_elves_and_compiled_resources():
    bundle = json.loads(NCOL.read_text())
    frozen = json.loads(BUNDLE.read_text())
    for kind in ("generic", "mma", "packed", "reuse"):
        assert bundle[kind + "_sha256"] == frozen[kind + "_sha256"]
    from codesign.ventus.elf import ELFProgram

    elf = ELFProgram.read(NCOL.parent / bundle["ncol"])
    assert elf.sha256 == bundle["ncol_sha256"]
    # Same register/LDS needs as the row-order kernels: only the block index map differs.
    assert (
        elf.resources["mma_reuse_ncol_bf16"]["vgpr"],
        elf.resources["mma_reuse_ncol_bf16"]["sgpr"],
    ) == (98, 14)
    assert (
        elf.resources["mma_packed_ncol_bf16"]["vgpr"],
        elf.resources["mma_packed_ncol_bf16"]["sgpr"],
    ) == (33, 11)
