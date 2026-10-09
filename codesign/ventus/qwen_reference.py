"""NumPy replay of Qwen dispatch manifests for small functional checks.

Each kernel follows the arithmetic, indexing and tail conditions of its OpenCL
source in examples/kernels (approximate SFU builtins use exact math). Only the
official addmm_mma_bf16 is replayed from its argument contract: strided
row/column operands, FP32 accumulation, optional bias and BF16 output. This is a
functional reference for comparing software organizations, not a timing model.
"""

import json
import math
import re
from pathlib import Path

import numpy as np


def to_bf16(x):
    """FP32 -> BF16 bits with round-to-nearest-even."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def from_bf16(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32)


class Memory:
    def __init__(self, buffers):
        self.regions = []
        for base in sorted({b["address"] & 0xF0000000 for b in buffers}):
            end = max(b["end"] for b in buffers if b["address"] & 0xF0000000 == base)
            self.regions.append((base, np.zeros(end - base + 4096, dtype=np.uint8)))

    def _view(self, address, nbytes):
        for base, data in self.regions:
            if base <= address and address + nbytes <= base + len(data):
                return data[address - base : address - base + nbytes]
        raise ValueError(f"address {address:#x} outside replay memory")

    def u16(self, address, count):
        return self._view(address, 2 * count).view(np.uint16)

    def u32(self, address, count):
        return self._view(address, 4 * count).view(np.uint32)

    def f32(self, address, count):
        return self._view(address, 4 * count).view(np.float32)


def pack_b_tiles(w):
    """Logical weight [N, K] -> MMA B tiles [ceil(N/16), ceil(K/16), K16, N16]."""
    n, k = w.shape
    nt, kt = -(-n // 16), -(-k // 16)
    pad = np.zeros((nt * 16, kt * 16), dtype=w.dtype)
    pad[:n, :k] = w
    return pad.reshape(nt, 16, kt, 16).transpose(0, 2, 3, 1).reshape(-1)


def logical_weights(raw, seed=0):
    """Deterministic BF16 values per logical tensor name (layout-independent)."""
    cfg = raw["scenario"]["configuration"]
    width, ff = cfg["hidden_size"], cfg["intermediate_size"]
    dim = width // cfg["num_attention_heads"]
    kvd = cfg["num_key_value_heads"] * dim
    shapes = dict(q=[width, width], k=[kvd, width], v=[kvd, width], o=[width, width])
    shapes.update(gate=[ff, width], up=[ff, width], down=[width, ff])
    shapes.update(qb=[width], kb=[kvd], vb=[kvd], norm1=[width], norm2=[width])
    out = {}

    def value(name, shape, scale):
        h = sum(ord(c) * 131**i for i, c in enumerate(name)) % (2**32)
        rng = np.random.default_rng([seed, h])
        return to_bf16(rng.standard_normal(shape).astype(np.float32) * scale)

    out["embedding=lm_head"] = value("embedding", [cfg["vocab_size"], width], 0.5)
    out["final_norm"] = value("final_norm", [width], 0.2)
    out["rope_inverse_frequency"] = (
        1.0 / (cfg["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    ).astype(np.float32)
    for layer in range(cfg["num_hidden_layers"]):
        for name, shape in shapes.items():
            scale = 1.0 + 0.0 if name.startswith("norm") else 1 / math.sqrt(shape[-1])
            out[f"layer{layer}.{name}"] = value(f"layer{layer}.{name}", shape, scale)
    return out


def initialise(raw, mem, seed=0, kv_seed=1):
    values = logical_weights(raw, seed)
    packed = raw["scenario"]["mapping"] in ("packed", "packed64")
    for buf in raw["scenario"]["weights"]:
        name = buf["name"]
        parts = name.split("|")
        if len(parts) > 1:
            layer = parts[0].split(".")[0]
            arrays = [values[parts[0]]] + [values[f"{layer}.{p}"] for p in parts[1:]]
            data = np.concatenate(arrays, axis=0)
        else:
            data = values[name]
        if name == "rope_inverse_frequency":
            mem.f32(buf["address"], data.size)[:] = data
            continue
        if packed and "layout" in buf:
            data = pack_b_tiles(data)
        mem.u16(buf["address"], data.size)[:] = data.reshape(-1)
    # Decode starts with context tokens of KV already resident.
    scenario = raw["scenario"]
    if scenario["phase"] == "decode":
        rng = np.random.default_rng(kv_seed)
        for buf in scenario["buffers"]:
            if re.fullmatch(r"layer\d+\.[KV]", buf["name"]):
                heads, cap, dim = buf["shape"]
                data = np.zeros((heads, cap, dim), dtype=np.uint16)
                ctx = scenario["context"]
                data[:, :ctx] = to_bf16(rng.standard_normal((heads, ctx, dim)) * 0.5)
                mem.u16(buf["address"], data.size)[:] = data.reshape(-1)


def f32bits(bits):
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def run(raw, mem):
    for d in raw["dispatches"]:
        for address, value in d.get("functional_words", {}).items():
            mem.u32(int(address), 1)[0] = value
        KERNELS[d["kernel"]](mem, d["global_size"][0], *d["arguments"])


def k_embedding_packed(mem, items, w, tokens, out, width, rows):
    kt = -(-width // 16)
    tok = mem.u32(tokens, rows).astype(np.int64)
    i = np.arange(width * rows)
    t, c = tok[i // width], i % width
    src = ((t // 16) * kt + c // 16) * 256 + (c % 16) * 16 + t % 16
    table = mem.u16(w, int(src.max()) + 1)
    mem.u16(out, width * rows)[:] = table[src]


def k_embedding(mem, items, w, tokens, out, width, rows):
    tok = mem.u32(tokens, rows).astype(np.int64)
    i = np.arange(width * rows)
    src = tok[i // width] * width + i % width
    mem.u16(out, width * rows)[:] = mem.u16(w, int(src.max()) + 1)[src]


def k_rms(mem, items, x, w, out, width, rows, eps, inverse_width):
    xv = from_bf16(mem.u16(x, rows * width)).reshape(rows, width)
    s = (xv.astype(np.float32) ** 2).sum(axis=1, dtype=np.float32)
    r = (1 / np.sqrt(s * f32bits(inverse_width) + f32bits(eps))).astype(np.float32)
    y = from_bf16(to_bf16(xv * r[:, None])) * from_bf16(mem.u16(w, width))[None, :]
    mem.u16(out, rows * width)[:] = to_bf16(y).reshape(-1)


def k_rope(mem, items, x, freq, out, heads, dim, rows, capacity, position, append):
    half = dim // 2
    xv = from_bf16(mem.u16(x, rows * heads * dim)).reshape(rows, heads, dim)
    f = mem.f32(freq, half)
    angle = (position + np.arange(rows, dtype=np.float32))[:, None] * f[None, :]
    co, si = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    a, b = xv[..., :half], xv[..., half:]
    view = mem.u16(out, heads * capacity * dim).reshape(heads, capacity, dim)
    view[:, append : append + rows, :half] = to_bf16(a * co - b * si).transpose(1, 0, 2)
    view[:, append : append + rows, half:] = to_bf16(b * co + a * si).transpose(1, 0, 2)


def k_transpose_append(mem, items, x, out, heads, dim, rows, capacity, append):
    xv = mem.u16(x, rows * heads * dim).reshape(rows, heads, dim)
    view = mem.u16(out, heads * capacity * dim).reshape(heads, capacity, dim)
    view[:, append : append + rows] = xv.transpose(1, 0, 2)


def k_softmax(mem, items, x, out, scratch, rows, cols, position, scale):
    xv = from_bf16(mem.u16(x, rows * cols)).reshape(rows, cols) * f32bits(scale)
    mask = np.arange(cols)[None, :] <= position + np.arange(rows)[:, None]
    mx = np.where(mask, xv, -np.inf).max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(xv - mx), 0).astype(np.float32)
    mem.f32(scratch, rows * cols)[:] = e.reshape(-1)
    mem.u16(out, rows * cols)[:] = to_bf16(e / e.sum(axis=1, keepdims=True)).reshape(-1)


def k_heads_to_rows(mem, items, x, out, heads, dim, rows):
    xv = mem.u16(x, heads * rows * dim).reshape(heads, rows, dim)
    mem.u16(out, heads * rows * dim)[:] = xv.transpose(1, 0, 2).reshape(-1)


def k_packed_add(mem, items, a, b, out, pairs):
    s = from_bf16(mem.u16(a, 2 * pairs)) + from_bf16(mem.u16(b, 2 * pairs))
    mem.u16(out, 2 * pairs)[:] = to_bf16(s)


def k_swiglu(mem, items, a, b, out, pairs):
    g = from_bf16(mem.u16(a, 2 * pairs))
    silu = from_bf16(to_bf16(g / (1 + np.exp(-g))))
    mem.u16(out, 2 * pairs)[:] = to_bf16(silu * from_bf16(mem.u16(b, 2 * pairs)))


def k_cast(mem, items, x, out, count):
    mem.f32(out, count)[:] = from_bf16(mem.u16(x, count))


def k_pack_a(mem, items, a, out, m, k):
    kt, mt = -(-k // 16), -(-m // 16)
    src = from_bf16(mem.u16(a, m * k)).reshape(m, k) if m * k else None
    pad = np.zeros((mt * 16, kt * 16), dtype=np.uint16)
    pad[:m, :k] = to_bf16(src)
    tiles = pad.reshape(mt, 16, kt, 16).transpose(0, 2, 1, 3).reshape(-1)
    # One warp per 16x16 tile; launched tiles beyond m*k tail are still written.
    count = (items // 32) * 256
    mem.u16(out, count)[: tiles.size] = tiles[:count]


def _mma_tiles(mem, a, b, m, n, k):
    kt = -(-k // 16)
    nt16 = -(-n // 16)
    mt16 = -(-m // 16)

    def a_tile(mt, kk):
        return from_bf16(mem.u16(a + (mt * kt + kk) * 512, 256)).reshape(16, 16)

    def b_tile(nn, kk):
        return from_bf16(mem.u16(b + (nn * kt + kk) * 512, 256)).reshape(16, 16)

    return kt, nt16, mt16, a_tile, b_tile


def _tile_result(a_tile, b_tile, mt, nn, kt, bias_values):
    acc = np.zeros((16, 16), dtype=np.float64) + bias_values[None, :]
    for kk in range(kt):
        acc += a_tile(mt, kk).astype(np.float64) @ b_tile(nn, kk).astype(np.float64)
    return acc.astype(np.float32)


def _store(mem, out, m, n, acc, mt, nn):
    view = mem.u16(out, m * n).reshape(m, n)
    rows = slice(mt * 16, min(m, mt * 16 + 16))
    cols = slice(nn * 16, min(n, nn * 16 + 16))
    view[rows, cols] = to_bf16(acc[: rows.stop - rows.start, : cols.stop - cols.start])


def _bias(mem, bias, has_bias, n, nn):
    v = np.zeros(16, dtype=np.float32)
    if has_bias:
        cols = np.arange(nn * 16, nn * 16 + 16)
        ok = cols < n
        v[ok] = from_bf16(mem.u16(bias, n))[cols[ok]]
    return v.astype(np.float64)


def k_mma_packed(mem, items, a, b, bias, out, m, n, k, has_bias):
    kt, nt, _, a_tile, b_tile = _mma_tiles(mem, a, b, m, n, k)
    for tile in range(items // 32):
        mt, nn = tile // nt, tile % nt
        acc = _tile_result(a_tile, b_tile, mt, nn, kt, _bias(mem, bias, has_bias, n, nn))
        _store(mem, out, m, n, acc, mt, nn)


def k_mma_packed_ncol(mem, items, a, b, bias, out, m, n, k, has_bias):
    kt, _, mtiles, a_tile, b_tile = _mma_tiles(mem, a, b, m, n, k)
    for tile in range(items // 32):
        mt, nn = tile % mtiles, tile // mtiles
        acc = _tile_result(a_tile, b_tile, mt, nn, kt, _bias(mem, bias, has_bias, n, nn))
        _store(mem, out, m, n, acc, mt, nn)


def k_mma_reuse(mem, items, a, b, bias, out, m, n, k, has_bias, two_m, column_order=False):
    kt, _, _, a_tile, b_tile = _mma_tiles(mem, a, b, m, n, k)
    nt = -(-n // 64)
    mtiles = -(-m // (32 if two_m else 16))
    for tile in range(items // 32):
        if column_order:
            mt, nn = (tile % mtiles) * (2 if two_m else 1), (tile // mtiles) * 4
        else:
            mt, nn = (tile // nt) * (2 if two_m else 1), (tile % nt) * 4
        for dm in range(2 if two_m else 1):
            for dn in range(4):
                bias_values = _bias(mem, bias, has_bias, n, nn + dn)
                acc = _tile_result(a_tile, b_tile, mt + dm, nn + dn, kt, bias_values)
                if (mt + dm) * 16 < m and (nn + dn) * 16 < n:
                    _store(mem, out, m, n, acc, mt + dm, nn + dn)


def k_addmm(mem, items, a, b, bias, o, m, n, k, as0, as1, _a, bs0, bs1, _b, os0, os1, *rest):
    offset, _one, _zero, has_bias = rest
    if offset:
        raise ValueError("replay supports zero output offset")
    i, j, kk = np.arange(m), np.arange(n), np.arange(k)
    av = from_bf16(mem.u16(a, int((m - 1) * as0 + (k - 1) * as1) + 1))
    bv = from_bf16(mem.u16(b, int((k - 1) * bs0 + (n - 1) * bs1) + 1))
    A = av[i[:, None] * as0 + kk[None, :] * as1].astype(np.float64)
    B = bv[kk[:, None] * bs0 + j[None, :] * bs1].astype(np.float64)
    acc = A @ B
    if has_bias:
        acc += from_bf16(mem.u16(bias, n))[None, :]
    view = mem.u16(o, int((m - 1) * os0 + (n - 1) * os1) + 1)
    view[i[:, None] * os0 + j[None, :] * os1] = to_bf16(acc.astype(np.float32))


def k_mma_m64(mem, items, a, b, bias, out, m, n, k, has_bias):
    kt, _, _, a_tile, b_tile = _mma_tiles(mem, a, b, m, n, k)
    mtiles = -(-m // 64)
    for tile in range(items // 32):
        mt, nn = (tile % mtiles) * 4, (tile // mtiles) * 4
        for dm in range(4):
            for dn in range(4):
                acc = _tile_result(
                    a_tile, b_tile, mt + dm, nn + dn, kt, _bias(mem, bias, has_bias, n, nn + dn)
                )
                if (mt + dm) * 16 < m and (nn + dn) * 16 < n:
                    _store(mem, out, m, n, acc, mt + dm, nn + dn)


KERNELS = {
    "embedding_packed_bf16": k_embedding_packed,
    "embedding_bf16": k_embedding,
    "rms_bf16": k_rms,
    "rope_bf16": k_rope,
    "transpose_append_bf16": k_transpose_append,
    "causal_softmax_bf16": k_softmax,
    "heads_to_rows_bf16": k_heads_to_rows,
    "packed_add_bf16": k_packed_add,
    "packed_swiglu_bf16": k_swiglu,
    "cast_bf16_f32": k_cast,
    "pack_a_bf16": k_pack_a,
    "mma_packed_bf16": k_mma_packed,
    "mma_reuse_bf16": k_mma_reuse,
    "mma_packed_ncol_bf16": k_mma_packed_ncol,
    "mma_reuse_ncol_bf16": lambda *args: k_mma_reuse(*args, column_order=True),
    # Same outputs as the column-order reuse kernel; LDS staging changes only how
    # the warp writes them (the compiled store coverage is checked from traces).
    "mma_stage_bf16": lambda *args: k_mma_reuse(*args[:-1], column_order=True),
    "mma_trim_bf16": lambda *args: k_mma_reuse(*args, column_order=True),
    "mma_m64_bf16": lambda mem, items, a, b, bias, out, m, n, k, hb: k_mma_m64(
        mem, items, a, b, bias, out, m, n, k, hb
    ),
    "addmm_mma_bf16": k_addmm,
}


def replay(program_path, seed=0):
    """Return logits and KV caches computed by replaying a manifest."""
    raw = json.loads(Path(program_path).read_text())
    buffers = raw["scenario"]["weights"] + raw["scenario"]["buffers"]
    mem = Memory(buffers)
    initialise(raw, mem, seed)
    run(raw, mem)
    by_name = {b["name"]: b for b in raw["scenario"]["buffers"]}
    logits = by_name["logits_fp32"]
    out = {"logits": mem.f32(logits["address"], logits["shape"][0]).copy()}
    for name, buf in by_name.items():
        if re.fullmatch(r"layer\d+\.[KV]", name):
            out[name] = mem.u16(buf["address"], math.prod(buf["shape"])).copy()
    return out
