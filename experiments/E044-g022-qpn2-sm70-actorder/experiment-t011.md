# E044 T011 — QPN2 multiwarp group-aligned execution

Status: **ACTIVE / performance promising at smoke; hardware-counter and paired gate-completeness pending. Not owner-accepted.**

## Donor audit and difference from T010

Pinned donor remains `dnv2003/v100-skinny` commit `5b589c0dc81223e0ba65bcb3e755874723f8b515`. In `skinny_kernels.cu`, `skinny_nvfp4_qpn2<SPLITK,NACC>` maps `G=K/16` fragments among SPLITK warps (`Gq=G/SPLITK`, `g0=warp*Gq`), accumulates each warp's fragment in FP32, writes a small shared FP32 tile, synchronizes once, and deterministically sums split results. Launcher uses split8 for some small-N/gate-up cases, split16 for small/mid work, split32 for N=2048. This donor route is NVFP4/E4M3 group16, not Falcon GPTQ.

T010 had one warp/CTA per N32 tile that visited all 96 GPTQ group128s serially; 96 CTAs at N=3072, no inter-warp CTA reduction, ~3x slower than c57. T011 retains E044 donor prepack/quadpair/HMMA and permutation gather, but maps **whole 128-element GPTQ groups** to each warp (12, 6, or 3 groups/warp for SPLITK 8/16/32 at K=12288), applies that group's independent FP16 scale after its FP32 group dot, and sums the split partials in deterministic warp-index order in an 8×32 shared FP32 tile. No global [G,M,N] partials, atomics, FP16 partials, or FP64 math. The output remains FP32 until final FP16 cast. Gate/up K=3072 only admits split8 with integral complete groups/warp; down K=12288 tested split16 and has split8/16/32 options.

## Correctness and resource evidence

`qpn2_multiwarp.py`, generated CUDA `qpn2_multiwarp.cu`, built with CUDA 12.8 / nvcc, `-gencode=compute_70,sm_70`, PyTorch 2.9.1+cu128, on the explicitly masked target V100 UUID. Six original raw fixture dirs are read-only inputs under `/mnt/data/pc01/.../s4-m3a-fixtures` (each manifest SHA is in raw rows); no RTX2060 visibility during runs. The L43/down stress input is existing E035 artifact `c57_stress_x_fp16.bin`.

- Coverage: all six fixtures × M1/M2/M3/M4; valid splits tested for each shape (51 rows total: 25 split8, 13 split16, 13 split32); plus L43/down M4 ×4 stress at split8/16/32.
- Frozen independent FP64 dequant/GEMM→FP32 oracle; tolerance `1e-4 + 5e-5*abs(reference)`: strict-bad=0 in all 51 rows; Graph FP32 scratch strict-bad=0; repeat equality and independent-scratch equality pass in all rows. Graph output cast equals Eager output cast in all rows. Maximum observed strict tolerance ratio is recorded per row in `t011-correctness.jsonl` (stress max ~0.446).
- Build fix: initial shared scratch was only 4 rows and generated an illegal access; changed to 8 rows (MMA fragment row indices span 0..7), then rebuilt and reran the whole matrix successfully. This is a resolved prototype defect, recorded rather than hidden.
- `cuobjdump --dump-resource-usage`: SPLITK8/16/32 kernel registers=47/thread, shared=8192/16384/32768 bytes, stack/local=0; gather REG12, no shared/stack/local. T010 kernel REG56/shared0. The higher SPLITK shared footprint can reduce residency; verify occupancy/actual issue under NCU when permitted. SASS has repeated `HMMA.884.F32.F32.STEP0..3`, LDG.128, and CTA barrier/reduction; SASS confirms compiled multiwarp design but does not establish measured warp eligibility/utilization.

Binary SHA256 `11ba4fc8bf75a4e38d0eb684e8931fe413c7dd08cad293e42aa0f718e1db689d`; generated CUDA SHA256 `e238cb33072c7be96975b03be66786f75c9cfca4f9f8bc8b387380590481d2e5`; Python harness SHA256 `256306998cd270047f00b71f2a1a13b022bb0a790b0951f57c1633b0a383f39a`; correctness JSONL SHA256 `d53c8aa923c93ea345096b8a39492747696cde9e7a746937686132e3f986c59c`; runlog SHA256 `557f23fbf0ff0275a8560e429378fd9870a69d5eff42f8df089a215a1d62b6e1`. Hashes in `results/SHA256SUMS-T011`. Binary stored externally under `/mnt/data/sglang-V100/t011-build/`; reproducible from harness.

## Preliminary full-Linear paired timing

`t011-paired-linear.py` captures gather + multiwarp HMMA + CTA FP32 split reduction + FP16 output boundary in both graphs (candidate FP32 is explicitly cast, exact-c57 already returns its native output type). c57 is exact binary `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`. Three truly alternating child-process pairs, each with 101 CUDA Event graph replays (timing helper corrected to even median):

- `43.mlp.gate_up_proj`, M4, split8: QPN ~89.088 µs all three; c57 ~252.928 µs all three; relative reduction ~64.8%, delta ~163.84 µs.
- `43.mlp.down_proj`, M4, split16: QPN ~73.728 µs all three; c57 ~155.648–156.672 µs; relative reduction ~52.8%, delta ~82.60 µs.
- Combined sequential Linear component pair: candidate 162.816 µs vs c57 409.6 µs, ~60.3% reduction. This is only two L43 M4 component measurements—not whole model Forward, not sufficient by itself for Owner production adoption.

Raw pair artifacts: `results/t011-gateup-m4-final.d399101e-a246-4e89-b54e-7c6531ac8f3d.jsonl`; `results/t011-down-m4-final.8ee67581-6ed2-4302-8754-706618510100.jsonl`. Earlier smoke logs are exploratory and should not replace the formal corrected-median runs.

### Blocking evidence gap

`ncu --query-metrics` on this machine returns `ERR_NVGPUCTRPERM` on both GPUs; do not bypass the system profiling permission, change host policy, or access RTX2060. Hence no eligible-warps/issue, DRAM/L2, stall counters are available. Static SASS/resource and wall-time A/B show a promising result but do not satisfy the owner's explicit request for actual warp-parallel and targeted NCU confirmation. Also complete Linear still needs the user's requested full gate_up+down results with ≥3 alternating pairs for SPLITK alternatives (especially down 32 and perhaps gate-up/down split variants), plus corrected accounting comparison for E044 old Eager cast omission/Graph copy overhead. Do not yet label performance gate closed, run no service/model matrix before final NCU/measurement disposition.

No production dispatch/default changed; S1/exact-c57 preserved; G023 remains ON_HOLD; E035 protected source untouched; no Owner ACCEPTED designation.
