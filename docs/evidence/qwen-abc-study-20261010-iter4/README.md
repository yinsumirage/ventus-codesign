# Qwen A/B/C study, iteration 4: GEMM epilogue spills

2026-10-10 (night). Follows [iteration 3](../qwen-abc-study-20261009-iter3/README.md);
earlier iterations stay unchanged. Same task, unified-v4 area (1.11 mm²), frozen
memory and T0. Model predictions of the native instruction executor only.

## Result: complete simulations

Same hardware as iteration 3 (N4 for A/C, B for B); only the GEMM kernels changed.

| Design | Area mm² | Q-P128 | Q-P512 | Q-D128 | Q-D512 | Task max r |
|---|---:|---:|---:|---:|---:|---:|
| A (prefill): N4 | 1.0812 | 0.2603 | 0.2366 | – | – | **0.260** |
| B (decode): B | 1.0848 | – | – | 0.2692 | 0.2494 | **0.269** |
| C (all four): N4 | 1.0812 | 0.2603 | 0.2366 | 0.2845 | 0.2626 | **0.285** |

Every entry is a complete `prepare-qwen` + `instructions` run ([validation.json](validation.json)).
Against iteration 3: A 0.342 → 0.260, B 0.279 → 0.269, C 0.342 → 0.285; speedups
A 3.84×, B 3.71×, C 3.51×. Software ([designs/](designs/)) is iteration 3's with
`epilogue: "unrolled"`.

## What changed

**Finding.** On N4 the prefill gate GEMM ran at 44% of the tensor peak while the
down GEMM, with the same MMA count but 5× fewer blocks, ran at 71%. The
difference was a per-block constant: 1,302 store instructions per 64×64 block
for 128 output stores, and 21 MB of DRAM writes per gate GEMM for 1.25 MB of output.
The disassembly shows why: the epilogue `for (reg...) out[...] = BF(acc[reg])`
indexes the float8 accumulator with a loop variable, and at O1 the compiler
writes all eight accumulator registers to the private stack before every output
store and reloads one (`pds_stack` 1080 B for `mma_m64_bf16`, 524 B for the
32×64 kernels). The executor charges each spill as a full-line write-through store.

**Unrolled epilogue** ([mma_unroll_bf16.cl](../../../examples/kernels/mma_unroll_bf16.cl),
same pinned LLVM): each register is named by a constant component
(`acc.s0 … acc.s7`), so the value is converted straight from the register. Tiles,
MMA order, block order, A-fragment trim and the written bytes are unchanged
(trace test: every output byte written exactly once; NumPy replay: logits and KV
caches). The k-loop has no spills. What remains (≈133 stores and reloads per
64×64 warp, ≈69 for the 32×64/trim kernel) is the kernel entry saving its
callee-saved vector registers; O2/O3 and source variants did not remove it.

On N4 (v2 slices, [unr1](experiments/unr1-pre.json)): P128 0.3412 → 0.2613,
P512 0.3307 → 0.2366, D128 0.2957 → 0.2858; complete runs above.

**Neighbourhood after the change** ([nls3-pre](experiments/nls3-pre.json),
[nls3-dec](experiments/nls3-dec.json), [bls-dec](experiments/bls-dec.json)): no single-field move
around N4 or B improves its task by more than 0.6%; several cut area at almost no
cost (L1 128 → 64 sets: −0.034 mm²). C is now bound by decode (D128 0.2845 against
P128 0.2603).

## Limits

As before: model predictions, uncalibrated extended-ISA timing, structural area,
local optimality of a finite menu. The remaining entry spills are an ABI cost of
the compiled kernels, not of the hardware model.
