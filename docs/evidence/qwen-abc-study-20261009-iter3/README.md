# Qwen A/B/C study, iteration 3: 64×64 tile kernel and a 2-SM prefill design

2026-10-09 (evening). Follows [iteration 2](../qwen-abc-study-20261009-iter2/README.md)
and [iteration 1](../qwen-abc-study-20261009/README.md), which stay unchanged. Same
task, unified-v4 area (1.11 mm²), frozen memory and T0. Model predictions of the
native instruction executor only.

## Result: complete simulations

| Design | Area mm² | Q-P128 | Q-P512 | Q-D128 | Q-D512 | Task max r |
|---|---:|---:|---:|---:|---:|---:|
| A (prefill): N3 | 1.0865 | 0.3422 | 0.3339 | – | – | **0.342** |
| B (decode): iteration 2 | 1.0848 | 0.4300 | 0.4382 | 0.2791 | 0.2578 | **0.279** |
| C (all four): N3 | 1.0865 | 0.3422 | 0.3339 | 0.2904 | 0.2677 | **0.342** |

Every entry is a complete `prepare-qwen` + `instructions` run ([validation.json](validation.json)).
N3 is hardware N ([N-hardware.json](designs/N-hardware.json)) with prefill software
[N3-software-Q-P128.json](designs/N3-software-Q-P128.json); its decode entries are
the complete runs of N with the trim kernel (same hardware and decode program).
Against iteration 2: A 0.367 → 0.342, C 0.367 → 0.342, B unchanged; speedups
A 2.92×, B 3.58×, C 2.92×.

N: 2 SMs, 8 warps/blocks, VGPR 1024, SGPR 256, Tensor (4,8,4)×1, LDS 16 KiB/8 banks,
L1 128 sets × 4 ways with 16 MSHRs, 1 subentry and 16 write entries, L2 128 × 16 with
32 MSHRs, LSU 16 entries / 8 per warp, collectors 4, RF 4 banks and single ports.

## What changed

**Traffic analysis.** On A=C, one prefill gate GEMM read 57 MB from DRAM for 8.7 MB
of weights (6.5×): the 32×64 tile re-reads weights per row tile and A=C's 32 KiB L2
holds no strip. GEMM traffic is proportional to 1/tile-rows + 1/tile-columns.

**64×64 tile kernel** ([mma_m64_bf16.cl](../../../examples/kernels/mma_m64_bf16.cl),
compiled with the same pinned LLVM; trace test checks exact output coverage with
row tails; replay test covers 64×64 for all prefill GEMMs). It cuts GEMM traffic by a
third but needs 173 VGPRs, so it only pays with VGPR 1024
([m64eval](experiments/m64eval.json)): on A=C (VGPR 512) P128 0.362 → 0.438; on
A=C with VGPR 1024 0.343, but 0.031 mm² over budget, and no affordable cut
recovered it ([m64cut](experiments/m64cut.json)).

**2-SM region.** A hand-built 2-SM region with the (4,8,4) Tensor, VGPR 1024 and
large caches ([region2sm](experiments/region2sm.json)) reached P128 0.377 with
32×64 tiles and 0.347 with 64×64 tiles at 1.0865 mm² ([r2m64](experiments/r2m64.json)):
design N. Its single- and paired-field neighbours within budget did not improve
P128 ([nls](experiments/nls.json)).

**Packed QK.** With the strided official QK kernel N's Q-P512 was 0.3772 (complete),
because QK's score stores bind on two SMs. Iteration 1 had rejected packed QK
(K rearranged into MMA tiles by an element transpose) as a loss on H0 and the
3-SM design. On N it wins ([nqk](experiments/nqk.json)): P512 0.377 → 0.334,
P128 0.347 → 0.341 (slices), confirmed by complete runs (0.3339, 0.3422). Staging
QK stores through LDS gave no further gain. A paired-field search with this
software ([nls2](experiments/nls2.json)) found L2 128 → 64 sets with 64 MSHRs
(N4) within 0.3% on prefill; its complete decode runs are worse (0.2953/0.2719) and
its Q-P512 was still running.

The MIP did not propose 64×64 by itself: the new tile had too few samples, so the
surrogate's unseen-level penalty kept it away; the 2-SM/64×64 region was found by
targeted simulator exploration and then refined by simulator-driven local search.

## Software of the selected designs

| Scenario | A/C (N3) | B |
|---|---|---|
| prefill | sharing, GQA batching, packed PV and packed QK, column order, 64×64 tiles for all multi-row GEMMs | (cross-evaluation only) column order, packed PV, 32×64 |
| decode | sharing, GQA batching, fusion, column order, trim | same |

## Limits

As before: model predictions, uncalibrated extended-ISA timing, structural area,
local optimality of a finite menu. The software choice is per scenario on fixed
hardware, as the task allows.
