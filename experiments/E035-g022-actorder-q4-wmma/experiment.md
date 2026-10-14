# E035-R2 — Down group-parallel ownership candidate

## Question
For Falcon L43 down M4 (then all L0/L22/L43 down × M1/M2/M4), does splitting the 96 independent GPTQ-group WMMA calculations into concurrent one-warp CTAs improve full Linear enough to merit a later Forward qualification, compared with exact R1 and accepted c57?

## Hypothesis
R1 L43/down M4 uses 192 serial group-processing one-warp CTAs, one CTA per N tile; each loops over 96 groups. Kernel-scope NCU records long-scoreboard stalls and low warp activity. Use grid `(N/16, ceil(M/16), G)` with CTA z selecting a group; each CTA computes one group dot product and writes unscaled FP32 partials. A separate reducer reads partials in original ascending group order, applying FP32 `fmaf(partial, scale, accumulator)`, then the ordinary FP16 cast. Hypothesis: more available independent CTAs shorten group dependency/underfill; cost: 4,718,592 B partial scratch at M4/down plus its write/read, reducer, extra launch and synchronization. Performance verdict requires complete Linear versus exact c57, not merely R1.

## Scope, identities, semantics
- Experiment-only source: `grouped_q4_sm70_parallel.cu`; symbol `grouped_q4_sm70_parallel`, kernels `group_wmma_partial` and `reduce_groups_ordered`; separate pybind name `e035_grouped_q4_parallel_sm70`. No production source/dispatcher/extension/default edited.
- R1 binary exactly reused: SHA `f72b764654de643384166f44acf7924fd8199a6512f9573bb921224394b2392d`. Exact c57: `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`. Parallel binary: `/mnt/data/pc01/sglang-v100-migration/experiments/E035/build-parallel/e035_grouped_q4_parallel_sm70.so`, SHA `23ea9c87cbea6d6f6d491a94a5ef2446c363c7367d573d0807ee2358aa26bbcd`, compiled PyTorch 2.9.1+cu128/CUDA12.8 with `nvcc -O3 -lineinfo -gencode=arch=compute_70,code=sm_70`, one job.
- Activation gather, Q4 group-major prepack, g_idx permutation, effective-zero-8 fastpath and scales are unchanged. Within each group, same 8 WMMA K=16 chunks, FP32 WMMA accumulator and its tile output order. Only ownership changes: each (group, M tile, N tile) CTA computes separately; FP32 group point results are stored unscaled, then exact g=0..G−1 ascending FMA consumes them. No FP16 partial, atomics, reassociation/tree reduction, scale-before-dot, or tolerance change.
- Per-call kernels: gather, parallel group WMMA, ordered reducer, then framework final cast. Eager and Graph benchmark includes them all.

## Pre-change evidence and decision criterion
R1 matched exact M4 L43/down NCU report was absent. This run collected it from exact R1 SHA/fixture with one filtered `group_wmma_fused` launch. It confirms R1 geometry grid `[192,1,1]`, block `[32,1,1]` (one warp per CTA), 48 registers/thread, cubin stack 0, 1.879 ms profiled kernel, DRAM read/write 19.590/0.363 MB, active warps 1.477M cumulative-derived `.avg` field, issue active `.avg` 172,101 cycles, long-scoreboard 47.34%, tensor instructions 4,718,592. Source/launch imply 192 CTA each serially traversing 96 groups. Note `cuobjdump` cubin `STACK:0` versus NCU `launch__stack_size=1024` metadata: only the former describes per-thread SASS stack frame here; NCU metric is launch stack limit/default and does not contradict the SASS. Do not attribute NCU stack_size to a local accumulator.

## New candidate parallelism and measured cost
At M4/down the parallel group kernel grid is exactly `[192,1,96]` = 18,432 one-warp CTAs, versus `[192,1,1]` / 192 CTAs R1. NCU filtered one target launch confirms 18,432 grid CTAs, 48 registers/thread, cubin frame to be checked via cuobjdump, 183.68 µs kernel, 18.982 MB DRAM read, 3.718 MB write, 1.777M active warps, issue active 166,939, long scoreboard 26.61%, tensor instructions 4,718,592. Thus group concurrency reduced this kernel's duration by ~10.2× and long-scoreboard fraction by 20.7 percentage points, but added ~3.355 MB DRAM writes relative to R1's group kernel. Logical FP32 partial volume is 4,718,592 B at M4 (G×M×N×4); it is not equal to measured DRAM write transactions. R1 and parallel NCU reports are single-kernel; not complete-call timing.

R2 raw per-kernel reports/logs:
- R1 `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-r1-down-m4.ncu-rep`, SHA `4fab9fa2a98d3d1eee2b03d5ce6c03d6461019b07e15604abb91e88af0c95a33`; log SHA `4787ec22cc3860799b5cefe8c1a922ef2e8c0970b98668befef5c9a2680c16e5`.
- Parallel `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-parallel-down-m4.ncu-rep`, SHA `a85afd9fc28cc1ee52f937b759cf8309509e624dd6ef1da57ba7ebaf345cf253`; log SHA `cbdd26da0d6cdb3ef106e94458a941ff73d19263e541d690ac38d5754cfd208d`.

## Correctness and full-Linear result
All three down fixtures L0/L22/L43 and M1/M2/M4 parallel strict-oracle cells passed. Frozen oracle is FP64 GPTQ matmul → FP32, `abs(error)<=1e-4+5e-5*abs(ref)`, strict-bad=0. FP32/FP16 finite; same-scratch repeat and independent scratch outputs bitwise equal; captured Graph M1/M2/M4 FP32 `go` finite and strict-bad=0, graph output equals `go.half()`, and Eager/Graph cast outputs equal. L43/down M4 input ×1 and ×4 (max 2420/9680) both strict-bad=0 and finite. R2 corrected script SHA is stored per result row; actual loaded binary SHA, dtype, Graph replay identity, and config are recorded. R2 old graph field limitation is documented separately in the control correction.

Three rotated pairs × 20 CUDA-event iterations/arm for L43/down M1/M2/M4; each arm ran in a fresh process, with actual distinct PIDs and binary SHAs. Median-of-three pair medians, µs:

| M | Eager R1 / parallel / c57 | Graph replay R1 / parallel / c57 | parallel vs c57 (Eager) |
|---:|---:|---:|---:|
| 1 | 1580.03 / 190.46 / 135.17 | 1579.01 / 191.49 / 132.10 | 1.41× slower |
| 2 | 1507.33 / 181.25 / 132.10 | 1508.35 / 183.30 / 131.07 | 1.37× slower |
| 4 | 1632.26 / 184.32 / 144.38 | 1628.16 / 187.39 / 138.24 | 1.28× slower |

Raw file `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-paired-down-all.jsonl`. Therefore candidate decisively improves R1, but loses to exact c57 in all three measured M values. As authorized, now extend full parallel correctness across L0/L22/L43 (done) and full R1/parallel/c57 A/B to all three down fixture matrices, not gate_up.
