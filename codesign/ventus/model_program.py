"""Lightweight kernel launches; timing sees ELF programs, never model/operator names."""

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

from .compiler import compile_kernel
from .elf import ELFProgram

ROOT = Path(__file__).resolve().parents[2]
TARGET = dict(
    packed_latency=3,
    conversion_latency=2,
    sfu_latency=4,
    mma_pipeline_latency=14,
    multiply_packing=2,
    fmul_latency=3,
    compare_latency=2,
    shuffle_latency=1,
)


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def build_bundle(official_kernels, compiler_root, llvm_source, out):
    """Compile once. Hardware searches subsequently reuse both linked programs."""
    out.mkdir()
    compile_kernel(
        ROOT / "examples/kernels/transformer_bf16.cl",
        "embedding_bf16",
        compiler_root,
        llvm_source,
        out / "generic",
    )
    compile_kernel(
        ROOT / "examples/kernels/mma_reuse_bf16.cl",
        "mma_reuse_bf16",
        compiler_root,
        llvm_source,
        out / "reuse",
    )
    compile_kernel(
        ROOT / "examples/kernels/mma_packed_bf16.cl",
        "mma_packed_bf16",
        compiler_root,
        llvm_source,
        out / "packed",
    )
    compile_kernel(
        official_kernels / "addmm_mma_bf16.cl",
        "addmm_mma_bf16",
        compiler_root,
        llvm_source,
        out / "mma",
    )
    bundle = {
        "schema": "ventus-kernel-bundle-v1",
        "generic": "generic/program.elf",
        "mma": "mma/program.elf",
        "packed": "packed/program.elf",
        "reuse": "reuse/program.elf",
        "precision": "BF16",
        "software": "branchless-finite-softmax; warp-shuffle-RMS; compact-GQA-KV; packed-SwiGLU",
        "sources": {
            "pytorch": "2b0e0cffbd4a46add6ec5f89bea665ce233c6b17",
            "llvm": "97df137e687669f9acf4bfd734779f52f2878614",
        },
        "instruction_target": TARGET,
        "target_status": "declared uncalibrated extended-ISA target; no low-precision measured cost",
    }
    for name in ("generic", "mma", "packed", "reuse"):
        bundle[name + "_sha256"] = ELFProgram.read(out / bundle[name]).sha256
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")


class Allocator:
    def __init__(self, base, limit):
        self.next, self.limit, self.buffers = base, limit, []

    def alloc(self, name, shape, dtype_bytes=2):
        size = math.prod(shape) * dtype_bytes
        address = (self.next + 127) // 128 * 128
        if address + size > self.limit:
            raise ValueError("RV32 allocation overflow")
        self.next = address + size
        self.buffers.append(
            dict(
                name=name,
                shape=list(shape),
                bytes=size,
                address=address,
                end=address + size,
                dtype_bytes=dtype_bytes,
            )
        )
        return address


GEMM_TILES = ("16x16", "16x64", "32x64")
# __local staging buffer of mma_stage_bf16, as an offset from the LDS base (the ABI
# passes __local arguments as offsets), above the compiler LDS stack (the
# provider starts the stack at the LDS base; the GEMM kernels use 44 B of it).
STAGE_LDS = 128
GEMM_CLASSES = ("q", "k", "v", "qkv", "o", "gate", "up", "gate_up", "down", "lm_head", "qk", "pv")
SWITCHES = ("share_packing", "gqa_batch", "fuse_rows1", "packed_attention", "packed_pv")
SOFTWARE_DEFAULTS = dict(
    tiles={}, block_order="row", epilogue="direct", **dict.fromkeys(SWITCHES, False)
)


def software_options(software):
    """Explicit packed64 extensions; None keeps the original packed64 organization.

    tiles: GEMM class -> output tile per warp ("16x16" packed, "16x64"/"32x64" reuse).
    share_packing: pack an activation once for consecutive GEMMs reading it (q/k/v, gate/up).
    gqa_batch: one QK and one PV GEMM per KV head over its stacked query heads.
    fuse_rows1: single-row (decode) QKV and gate/up GEMMs over concatenated weights.
    packed_attention: multi-row (prefill) QK/PV use packed MMA kernels; K and V are
        rearranged into MMA B tiles by transpose_append dispatches (timed).
    packed_pv: multi-row PV only uses packed MMA over V B tiles (one chunk-copy
        dispatch per KV head); QK keeps the strided official kernel.
    block_order: "row" (row tiles outer) or "col" (column strips outer; needs the
        bundle's compiled ncol kernels). Tiles, registers and MMA sequence are unchanged.
    epilogue: "direct" (one BF16 store per lane) or "lds" (16x64/32x64 column-order
        GEMMs stage results through 256 B of LDS and store full 128 B row lines;
        m <= 4 loads only the non-zero A fragment register) or "trim" (direct stores
        with the same A-fragment trim). Both need block_order "col".
    """
    options = {**SOFTWARE_DEFAULTS, **(software or {})}
    if set(options) != set(SOFTWARE_DEFAULTS):
        raise ValueError("unknown software option")
    tiles = options["tiles"]
    if not isinstance(tiles, dict) or any(
        k not in GEMM_CLASSES or v not in GEMM_TILES for k, v in tiles.items()
    ):
        raise ValueError("tiles map GEMM classes to 16x16, 16x64 or 32x64")
    if any(type(options[k]) is not bool for k in SWITCHES):
        raise ValueError("software switches must be booleans")
    if options["block_order"] not in ("row", "col"):
        raise ValueError("block_order is row or col")
    if options["epilogue"] not in ("direct", "lds", "trim"):
        raise ValueError("epilogue is direct, lds or trim")
    if options["epilogue"] != "direct" and options["block_order"] != "col":
        raise ValueError("lds/trim epilogues are compiled for column block order")
    return options


def qwen_manifest(
    bundle_path,
    out,
    *,
    phase,
    context,
    steps=16,
    small=False,
    mapping="packed",
    software=None,
):
    if phase not in ("prefill", "decode") or type(context) is not int or context < 1:
        raise ValueError("invalid scenario")
    if type(steps) is not int or steps < 1:
        raise ValueError("invalid decode steps")
    if mapping not in ("packed", "packed64", "scalar"):
        raise ValueError("unknown software mapping")
    if software is not None and mapping != "packed64":
        raise ValueError("software options extend the packed64 mapping")
    sw = software_options(software)
    source = ROOT / "examples/models/qwen2.5-0.5b-instruct.config.json"
    frozen = json.loads(source.read_text())
    cfg = dict(frozen)
    if small:
        cfg.update(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=2,
            vocab_size=256,
        )
    width, ff, heads, kvheads, layers, vocab = (
        cfg[k]
        for k in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "vocab_size",
        )
    )
    dim = width // heads
    if (
        width % heads
        or heads % kvheads
        or dim % 2
        or context + steps > cfg["max_position_embeddings"]
    ):
        raise ValueError("invalid model shape")
    bundle = json.loads(bundle_path.read_text())
    if bundle["schema"] != "ventus-kernel-bundle-v1" or bundle["precision"] != "BF16":
        raise ValueError("incompatible compiled bundle")
    out.mkdir()
    if sw["block_order"] == "col" and "ncol" not in bundle:
        raise ValueError("column block order needs a bundle with compiled ncol kernels")
    if sw["epilogue"] != "direct" and "stage" not in bundle:
        raise ValueError("LDS-staged epilogue needs a bundle with the compiled stage kernel")
    for kind in (
        "generic",
        "mma",
        "packed",
        *[k for k in ("reuse", "ncol", "stage") if k in bundle],
    ):
        elf = ELFProgram.read(bundle_path.parent / bundle[kind])
        if elf.sha256 != bundle[kind + "_sha256"]:
            raise ValueError("compiled bundle hash differs")
        shutil.copyfile(bundle_path.parent / bundle[kind], out / (kind + ".elf"))
    weights, temps = Allocator(0x90000000, 0xF0000000), Allocator(0x10000000, 0x50000000)
    embedding = weights.alloc("embedding=lm_head", [vocab, width])
    freq = weights.alloc("rope_inverse_frequency", [dim // 2], 4)
    per_layer = []
    # Packed B tiles are N-tile major, so concatenating weights along N concatenates buffers.
    fused = sw["fuse_rows1"] and phase == "decode"
    kvd = kvheads * dim
    for layer in range(layers):
        if fused:
            w = {
                "norm1": weights.alloc(f"layer{layer}.norm1", [width]),
                "norm2": weights.alloc(f"layer{layer}.norm2", [width]),
                "qkv": weights.alloc(f"layer{layer}.q|k|v", [width + 2 * kvd, width]),
                "qkvb": weights.alloc(f"layer{layer}.qb|kb|vb", [width + 2 * kvd]),
                "o": weights.alloc(f"layer{layer}.o", [width, width]),
                "gate_up": weights.alloc(f"layer{layer}.gate|up", [2 * ff, width]),
                "down": weights.alloc(f"layer{layer}.down", [width, ff]),
            }
            w["q"], w["qb"] = w["qkv"], w["qkvb"]
            w["k"], w["kb"] = w["qkv"] + width * width * 2, w["qkvb"] + width * 2
            w["v"], w["vb"] = w["k"] + kvd * width * 2, w["kb"] + kvd * 2
            w["gate"], w["up"] = w["gate_up"], w["gate_up"] + ff * width * 2
            per_layer.append(w)
            continue
        shapes = {
            "norm1": [width],
            "norm2": [width],
            "q": [width, width],
            "k": [kvheads * dim, width],
            "v": [kvheads * dim, width],
            "qb": [width],
            "kb": [kvheads * dim],
            "vb": [kvheads * dim],
            "o": [width, width],
            "gate": [ff, width],
            "up": [ff, width],
            "down": [width, ff],
        }
        per_layer.append(
            {k: weights.alloc(f"layer{layer}.{k}", shape) for k, shape in shapes.items()}
        )
    weight_addresses = {
        embedding,
        *[w[k] for w in per_layer for k in ("q", "k", "v", "o", "gate", "up", "down")],
        *[w[k] for w in per_layer for k in ("qkv", "gate_up") if k in w],
    }
    for buffer in weights.buffers:
        if mapping in ("packed", "packed64") and buffer["address"] in weight_addresses:
            buffer["layout"] = "MMA B tiles: [ceil(N/16),ceil(K/16),K16,N16]; logical shape [N,K]"
    finalnorm = weights.alloc("final_norm", [width])
    rows = context if phase == "prefill" else 1
    cap = context if phase == "prefill" else context + steps
    group = heads // kvheads
    srows = rows * group if sw["gqa_batch"] else rows
    attention_tiles = sw["packed_attention"] and phase == "prefill"
    pv_tiles = (sw["packed_attention"] or sw["packed_pv"]) and phase == "prefill"
    if pv_tiles and (cap % 16 or dim % 16):
        raise ValueError("packed attention needs 16-multiple context and head width")
    if pv_tiles:
        tiles_needed = max(-(-rows // 16) * -(-ff // 16), -(-srows // 16) * -(-max(cap, dim) // 16))
        packed_a = temps.alloc("packed_activation_scratch", [tiles_needed, 16, 16])
    else:
        packed_a = temps.alloc(
            "packed_activation_scratch", [(rows + 15) // 16, (ff + 15) // 16, 16, 16]
        )
    tokens = temps.alloc("tokens", [rows], 4)
    x = temps.alloc("hidden", [rows, width])
    y = temps.alloc("normalized", [rows, width])
    if fused:
        qkv_out = temps.alloc("q|k|v_projection", [rows, width + 2 * kvd])
        q, k, v = qkv_out, qkv_out + width * 2, qkv_out + (width + kvd) * 2
    else:
        q = temps.alloc("q_projection", [rows, width])
        k = temps.alloc("k_projection", [rows, kvheads * dim])
        v = temps.alloc("v_projection", [rows, kvheads * dim])
    qr = temps.alloc("rotated_q", [heads, rows, dim])
    att = temps.alloc("head_attention", [heads, rows, dim])
    joined = temps.alloc("joined_attention", [rows, width])
    proj = temps.alloc("output_projection", [rows, width])
    if fused:
        gate_up_out = temps.alloc("gate|up", [rows, 2 * ff])
        gate, up = gate_up_out, gate_up_out + ff * 2
    else:
        gate = temps.alloc("gate", [rows, ff])
        up = temps.alloc("up", [rows, ff])
    glu = temps.alloc("swiglu", [rows, ff])
    scores = temps.alloc("scores", [srows, cap])
    prob = temps.alloc("probabilities", [srows, cap])
    logits = temps.alloc("logits_bf16", [vocab])
    logits32 = temps.alloc("logits_fp32", [vocab], 4)
    softscratch = temps.alloc("softmax_fp32_exponentials", [srows, cap], 4)
    caches = [
        (
            temps.alloc(f"layer{layer}.K", [kvheads, cap, dim]),
            temps.alloc(f"layer{layer}.V", [kvheads, cap, dim]),
        )
        for layer in range(layers)
    ]
    if attention_tiles:
        k_transposed = temps.alloc("K_transposed", [dim, kvheads * cap])
        k_tiles = temps.alloc("K_mma_B_tiles", [kvheads, cap, dim])
    if pv_tiles:
        v_tiles = temps.alloc("V_mma_B_tiles", [kvheads, cap, dim])
    dispatches = []
    packed_source = [None]

    def emit(name, kernel, args, items, kind="generic", known=None, check=None):
        if kind not in ("packed", "reuse", "ncol", "stage"):
            # Any other kernel may rewrite a packed source; never reuse across it.
            packed_source[0] = None
        elf_hash = bundle[kind + "_sha256"]
        dispatches.append(
            dict(
                name=name,
                kernel=kernel,
                elf=kind + ".elf",
                elf_sha256=elf_hash,
                arguments=args,
                global_size=[(items + 31) // 32 * 32, 1, 1],
                local_size=[32, 1, 1],
                functional_words=known or {},
                previous_dispatch=dispatches[-1]["name"] if dispatches else None,
                work=check or {},
            )
        )

    def mm(name, a, b, bias, o, m, n, kdim, astride, bstride, ostride, offset=0, b_tiles=False):
        if mapping in ("packed", "packed64") and (b in weight_addresses or b_tiles):
            if astride != (kdim, 1) or ostride != (n, 1) or offset:
                raise ValueError("packed software mapping requires row-major activation/output")
            if not (sw["share_packing"] and packed_source[0] == (a, m, kdim)):
                emit(
                    name + ".pack_activation",
                    "pack_a_bf16",
                    [a, packed_a, m, kdim],
                    ((m + 15) // 16) * ((kdim + 15) // 16) * 32,
                    "packed",
                )
            packed_source[0] = (a, m, kdim) if o != a else None
            tile = sw["tiles"].get(name.rsplit(".", 1)[-1])
            if tile is None and mapping == "packed64" and n % 64 == 0:
                tile = "32x64" if m >= 32 else "16x64"
            if tile in ("16x64", "32x64"):
                if n % 64 or (tile == "32x64" and m <= 16):
                    raise ValueError(f"{tile} tile does not apply to {name}")
                two_m = int(tile == "32x64")
                mt = 32 if two_m else 16
                col = sw["block_order"] == "col"
                staged = sw["epilogue"] == "lds"
                kernel, kind = (
                    ("mma_stage_bf16", "stage")
                    if staged
                    else ("mma_trim_bf16", "stage")
                    if sw["epilogue"] == "trim"
                    else ("mma_reuse_ncol_bf16", "ncol")
                    if col
                    else ("mma_reuse_bf16", "reuse")
                )
                emit(
                    name,
                    kernel,
                    [packed_a, b, bias, o, m, n, kdim, int(bool(bias)), two_m]
                    + ([STAGE_LDS] if staged else []),
                    ((m + mt - 1) // mt) * (n // 64) * 32,
                    kind,
                    check=dict(
                        effective_multiplies=m * n * kdim,
                        mma_instructions=((m + mt - 1) // mt)
                        * (n // 64)
                        * (8 if two_m else 4)
                        * ((kdim + 15) // 16),
                    ),
                )
                return
            col = sw["block_order"] == "col"
            emit(
                name,
                "mma_packed_ncol_bf16" if col else "mma_packed_bf16",
                [packed_a, b, bias, o, m, n, kdim, int(bool(bias))],
                ((m + 15) // 16) * ((n + 15) // 16) * 32,
                "ncol" if col else "packed",
                check=dict(
                    effective_multiplies=m * n * kdim,
                    mma_instructions=((m + 15) // 16) * ((n + 15) // 16) * ((kdim + 15) // 16),
                ),
            )
            return
        args = [
            a,
            b,
            bias,
            o,
            m,
            n,
            kdim,
            *astride,
            0,
            *bstride,
            0,
            *ostride,
            offset,
            1,
            0,
            int(bool(bias)),
        ]
        emit(
            name,
            "addmm_mma_bf16",
            args,
            ((m + 15) // 16) * ((n + 15) // 16) * 32,
            "mma",
            check=dict(
                effective_multiplies=m * n * kdim,
                mma_instructions=((m + 15) // 16) * ((n + 15) // 16) * ((kdim + 15) // 16),
            ),
        )

    for step in range(1 if phase == "prefill" else steps):
        pos = 0 if phase == "prefill" else context + step
        length = rows if phase == "prefill" else pos + 1
        prefix = "prefill" if phase == "prefill" else f"decode{step:02}"
        known = {tokens + 4 * t: (pos + t + 17) % vocab for t in range(rows)}
        emit(
            prefix + ".embedding",
            "embedding_packed_bf16" if mapping in ("packed", "packed64") else "embedding_bf16",
            [embedding, tokens, x, width, rows],
            rows * width,
            known=known,
        )
        for layer, w in enumerate(per_layer):
            stem = f"{prefix}.layer{layer:02}"
            kc, vc = caches[layer]
            for norm, inp, output in [("norm1", x, y)]:
                emit(
                    stem + "." + norm,
                    "rms_bf16",
                    [
                        inp,
                        w[norm],
                        output,
                        width,
                        rows,
                        fbits(cfg["rms_norm_eps"]),
                        fbits(1 / width),
                    ],
                    rows * 32,
                )
            projections = (
                [("qkv", q, width + 2 * kvd)]
                if fused
                else [("q", q, width), ("k", k, kvd), ("v", v, kvd)]
            )
            for name, outbuf, n in projections:
                mm(
                    stem + "." + name,
                    y,
                    w[name],
                    w[name + "b"],
                    outbuf,
                    rows,
                    n,
                    width,
                    (width, 1),
                    (1, width),
                    (n, 1),
                )
            emit(
                stem + ".rope_q",
                "rope_bf16",
                [q, freq, qr, heads, dim, rows, rows, pos, 0],
                rows * heads * dim // 2,
            )
            emit(
                stem + ".rope_k_append",
                "rope_bf16",
                [k, freq, kc, kvheads, dim, rows, cap, pos, pos],
                rows * kvheads * dim // 2,
            )
            emit(
                stem + ".v_append",
                "transpose_append_bf16",
                [v, vc, kvheads, dim, rows, cap, pos],
                rows * kvheads * dim,
            )
            if attention_tiles:
                # K [kv, cap, dim] -> K^T [dim, kv*cap] -> 16-wide chunks in MMA B-tile order.
                emit(
                    stem + ".k_transpose",
                    "transpose_append_bf16",
                    [kc, k_transposed, dim, 1, kvheads * cap, kvheads * cap, 0],
                    kvheads * cap * dim,
                )
                emit(
                    stem + ".k_tiles",
                    "transpose_append_bf16",
                    [k_transposed, k_tiles, kvheads * cap // 16, 16, dim, dim, 0],
                    kvheads * cap * dim,
                )
            if pv_tiles:
                for kh in range(kvheads):
                    emit(
                        stem + f".kv{kh}.v_tiles",
                        "transpose_append_bf16",
                        [
                            vc + kh * cap * dim * 2,
                            v_tiles + kh * cap * dim * 2,
                            dim // 16,
                            16,
                            cap,
                            cap,
                            0,
                        ],
                        cap * dim,
                    )
            # GQA batching stacks the query heads of one KV head: [group*rows, dim] in qr/att.
            units = (
                [(f"kv{kh}", kh, kh * group, group) for kh in range(kvheads)]
                if sw["gqa_batch"]
                else [(f"head{h:02}", h // group, h, 1) for h in range(heads)]
            )
            for label, kh, h0, count in units:
                m_att = rows * count
                if attention_tiles:
                    qk_b, qk_bstride = k_tiles + kh * cap * dim * 2, None
                else:
                    qk_b, qk_bstride = kc + kh * cap * dim * 2, (1, dim)
                if pv_tiles:
                    pv_b, pv_bstride = v_tiles + kh * cap * dim * 2, None
                else:
                    pv_b, pv_bstride = vc + kh * cap * dim * 2, (dim, 1)
                mm(
                    stem + f".{label}.qk",
                    qr + h0 * rows * dim * 2,
                    qk_b,
                    0,
                    scores,
                    m_att,
                    length,
                    dim,
                    (dim, 1),
                    qk_bstride,
                    (length, 1),
                    b_tiles=attention_tiles,
                )
                # Decode rows see every cached column, so one call over stacked heads is
                # exact; prefill keeps one call per head (the causal mask uses the row).
                calls = (
                    [(h0, rows, 0)]
                    if count == 1
                    else (
                        [(h0 + j, rows, j) for j in range(count)]
                        if rows > 1
                        else [(None, count, 0)]
                    )
                )
                for head, srow, j in calls:
                    emit(
                        stem
                        + (f".head{head:02}.softmax" if head is not None else f".{label}.softmax"),
                        "causal_softmax_bf16",
                        [
                            scores + j * rows * length * 2,
                            prob + j * rows * length * 2,
                            softscratch + j * rows * length * 4,
                            srow,
                            length,
                            pos,
                            fbits(dim**-0.5),
                        ],
                        srow * 32,
                    )
                mm(
                    stem + f".{label}.pv",
                    prob,
                    pv_b,
                    0,
                    att + h0 * rows * dim * 2,
                    m_att,
                    dim,
                    length,
                    (length, 1),
                    pv_bstride,
                    (dim, 1),
                    b_tiles=pv_tiles,
                )
            if not fused:
                emit(
                    stem + ".join_heads",
                    "heads_to_rows_bf16",
                    [att, joined, heads, dim, rows],
                    rows * width,
                )
            mm(
                stem + ".o",
                att if fused else joined,
                w["o"],
                0,
                proj,
                rows,
                width,
                width,
                (width, 1),
                (1, width),
                (width, 1),
            )
            emit(
                stem + ".attention_residual",
                "packed_add_bf16",
                [x, proj, x, rows * width // 2],
                rows * width // 2,
            )
            emit(
                stem + ".norm2",
                "rms_bf16",
                [x, w["norm2"], y, width, rows, fbits(cfg["rms_norm_eps"]), fbits(1 / width)],
                rows * 32,
            )
            for name, outbuf, n in (
                [("gate_up", gate, 2 * ff)] if fused else [("gate", gate, ff), ("up", up, ff)]
            ):
                mm(
                    stem + "." + name,
                    y,
                    w[name],
                    0,
                    outbuf,
                    rows,
                    n,
                    width,
                    (width, 1),
                    (1, width),
                    (n, 1),
                )
            emit(
                stem + ".swiglu",
                "packed_swiglu_bf16",
                [gate, up, glu, rows * ff // 2],
                rows * ff // 2,
            )
            mm(
                stem + ".down",
                glu,
                w["down"],
                0,
                proj,
                rows,
                width,
                ff,
                (ff, 1),
                (1, ff),
                (width, 1),
            )
            emit(
                stem + ".ffn_residual",
                "packed_add_bf16",
                [x, proj, x, rows * width // 2],
                rows * width // 2,
            )
        emit(
            prefix + ".final_norm",
            "rms_bf16",
            [x, finalnorm, y, width, rows, fbits(cfg["rms_norm_eps"]), fbits(1 / width)],
            rows * 32,
        )
        mm(
            prefix + ".lm_head",
            y + (rows - 1) * width * 2,
            embedding,
            0,
            logits,
            1,
            vocab,
            width,
            (width, 1),
            (1, width),
            (vocab, 1),
        )
        emit(prefix + ".logits_cast", "cast_bf16_f32", [logits, logits32, vocab], vocab)
    raw = dict(
        schema="ventus-dispatches-v1",
        hardware=dict(memory_channels=1, memory_latency=100, memory_bytes_per_cycle=64),
        instruction_target=bundle["instruction_target"],
        dispatches=dispatches,
        scenario=dict(
            model="Qwen2.5-0.5B-Instruct",
            configuration=cfg,
            configuration_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            small_debug=small,
            phase=phase,
            context=context,
            decode_steps=steps if phase == "decode" else 0,
            precision="BF16 operands and storage; FP32 reductions/MMA accumulator",
            software=bundle["software"],
            mapping=mapping,
            **({"software_options": sw} if software is not None else {}),
            initial_cache="cold",
            initial_kv="empty"
            if phase == "prefill"
            else f"{context} tokens per layer in external memory",
            kv_capacity=cap,
            kv_initial_bytes=0 if phase == "prefill" else layers * 2 * kvheads * context * dim * 2,
            kv_appended_bytes=layers
            * 2
            * kvheads
            * (rows if phase == "prefill" else steps)
            * dim
            * 2,
            weights=weights.buffers,
            buffers=temps.buffers,
            launch_order="fenced sequential dispatches; caches carried throughout",
            excluded=[
                "sampling",
                "tokenizer",
                "initial host-device transfer",
                "compilation",
                "weight loading",
            ],
            official_fallback_policy="explicit device software variants replace unsupported BF16 embedding/advanced indexing and row reductions; no claim to replay unmodified PyTorch CPU fallback",
        ),
    )
    (out / "program.json").write_text(json.dumps(raw, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                dispatches=len(dispatches),
                weights_bytes=sum(b["bytes"] for b in weights.buffers),
                buffers_bytes=sum(b["bytes"] for b in temps.buffers),
                manifest=str(out / "program.json"),
            )
        )
    )
