# Qwen A/B/C hardware–software study: workflow

This is the executable workflow for the [project task](project_tasks_cn.md):
one fixed hardware per task (A prefill, B decode, C all four scenarios) under the
unified-v4 1.11 mm² budget, software chosen per kernel/shape/scenario, objective
max r(w) = T(H,S,w)/T0(w). Results are in
[the study evidence](evidence/qwen-abc-study-20261009/README.md).

All cycle numbers come from the native instruction executor running compiled ELF
dispatches. Model timing, RTL timing and hardware measurements remain distinct
claims; nothing here is an RTL or silicon measurement.

## Components

| File | Role |
|---|---|
| `codesign/ventus/model_program.py` | `software=` options extending packed64 (default `None` keeps the original program) |
| `examples/kernels/mma_reuse_ncol_bf16.cl` | column-strip block order variants of the reuse and packed GEMM kernels |
| `examples/kernels/mma_stage_bf16.cl` | `mma_trim_bf16` (A-fragment trim) and `mma_stage_bf16` (trim + LDS-staged epilogue) |
| `examples/kernels/mma_m64_bf16.cl` | 64×64 output tile per warp (VGPR 173) |
| `examples/kernels/mma_unroll_bf16.cl` | `mma_trimu_bf16`, `mma_m64u_bf16`: trim / 64×64 with the store loop unrolled (no per-store accumulator spills) |
| `examples/kernels/mma_multi_bf16.cl` | `mma_trimm_bf16`, `mma_m64m_bf16`: unrolled kernels where each warp computes several tiles in turn |
| `tests/fixtures/compiled-transformer-ncol/` | the frozen four ELFs plus the compiled column-order, stage, m64, unroll and multi ELFs and their receipts |
| `codesign/ventus/qwen_study.py` | sliced evaluator, legality (area, compiled residency), append-only slice cache |
| `codesign/ventus/qwen_search.py` | surrogate features, ridge fit, HiGHS MIP with unified area rows |
| `codesign/ventus/qwen_abc.py` | rounds: fit → MIP per software combination → independent check → simulate |
| `codesign/ventus/qwen_report.py` | per-scenario software selection on fixed hardware, cross table, controls |
| `codesign/ventus/qwen_reference.py` | NumPy replay of manifests for functional comparison of software options |

## Software options

`./run model prepare-qwen ... --mapping packed64 --software options.json` accepts:

| Key | Effect | Phase |
|---|---|---|
| `share_packing` | pack an activation once for consecutive GEMMs that read it | both |
| `gqa_batch` | one QK and one PV GEMM per KV head over its stacked query heads | both |
| `fuse_rows1` | concatenated q/k/v and gate/up weights for single-row GEMMs; no head join | decode |
| `packed_pv` | PV over V rearranged into MMA B tiles (one chunked `transpose_append` per KV head) | prefill |
| `packed_attention` | additionally QK over K tiles (element transpose); kept for the record | prefill |
| `tiles` | per GEMM class: `16x16`, `16x64`, `32x64`, `64x64` (m > 16) | both |
| `block_order` | `col` uses the compiled column-order kernels (needs the ncol bundle) | both |
| `epilogue` | `trim`: column-order kernel that loads only the non-zero A fragment register for m ≤ 4; `lds`: additionally stages outputs through LDS for full-line stores; `unrolled`: trim with constant-index stores, so the compiler no longer writes the whole accumulator to the private stack before each output store (also selects the unrolled 64×64 kernel; all need `col`) | both |
| `tiles_per_warp` | 1, 2, 4 or 8 output tiles per warp (`unrolled` only): the kernel entry's callee-saved register spills are paid once per warp | both |

The network, BF16 storage, FP32 accumulation and semantics are unchanged; every
layout conversion is a timed dispatch. The NumPy replay compares logits and
complete KV caches of each option set with the original program
(`tests/test_qwen_software_options.py`).

New kernels are compiled with the pinned LLVM (see
[instruction programs](instruction-programs.md)). The local toolchain build was
first checked by recompiling `mma_reuse_bf16.cl` and `mma_packed_bf16.cl`: both
reproduce the frozen ELF bytes.

## Sliced evaluation

Estimator v1 simulates, each cold in its own session, (1) the prologue, all
dispatches of layer 0 and the epilogue, scaling layer 0 by 24, and (2) the LM-head
GEMM, whose work is identical in all four scenarios. Decode uses step 8 of 16 and
reports the per-step mean. Estimator v2 (`QWEN_STUDY_ESTIMATOR=v2`, default) samples
layers 0–2 for P128 and decode, and replays the previous decode step's tail (last
layer, final norm, LM head) before step 8 so caches carry continuous-decode state;
its LM head is part of the decode slice. Searches used v1 (largest sample base);
final comparisons use v2 and complete simulations. Completed full simulations check the estimate
(evidence README). Every sliced dispatch checks its dynamic MMA count against the
independent shape count.

## Search

```bash
# Run inside the execution environment selected by local.toml.
python -m codesign.ventus.qwen_study --study results/abc-study/db --designs designs.json --scenarios Q-P128 Q-D128 --explore
python -m codesign.ventus.qwen_abc --study results/abc-study/db --task C --policy joint --rounds 3
python -m codesign.ventus.qwen_abc --study results/abc-study/db --task C --policy hardware-only --rounds 2
python -m codesign.ventus.qwen_abc --study results/abc-study/db --task C --policy sequential --fixed-software seq.json --rounds 2
python -m codesign.ventus.qwen_report --study results/abc-study/db --selection selection.json --out results/abc-report-001
```

`--explore` admits over-budget samples to teach the surrogate; they are never
reported as designs. Round receipts (`rounds-<task>-<policy>.jsonl`) keep fit
errors, every solve's status, gap, bound, prediction and the simulated ratios.

The surrogate is log-additive over one-hot hardware choices with selected pairwise
products, tile-specific residency-field terms and a software-variant offset. LDS
fields have no term: sweeps gave bit-identical cycles, since every kernel uses at
most 128 B of LDS per block. HiGHS solves min t subject to surrogate log r(w) ≤ t,
unified area rows, compiled-resource rows (every kernel keeps one resident block),
no-good cuts and an optional trust region, from a complete feasible start. Global
solves over all 27 fields often stop at the time limit; trust-region solves around
the incumbent reach proven optimality of the surrogate within seconds and supplied
most improvements. Optimality is always with respect to the surrogate on the
finite menu, never the simulator.

## Validation of a selected design

```bash
./run model prepare-qwen --bundle tests/fixtures/compiled-transformer/bundle.json --phase decode --context 128 --steps 16 --mapping packed64 --software sw.json --out results/x-program
./run model instructions --input results/x-program/program.json --hardware hw.json --out results/x-full --budget 14400
```

Column-order software uses `tests/fixtures/compiled-transformer-ncol/bundle.json`.
