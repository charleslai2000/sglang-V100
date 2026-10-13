# E032 — G022 Falcon Mamba input projection low-M kernel candidate (rejected at correctness)

## Question
Can an opt-in SM70 FP16 M=2/4 row-reuse GEMM reduce the source-attributed Falcon Mamba `in_proj` CUTLASS owner time for the exact `[M,3072] x [6680,3072]^T` Decode geometry without changing default routing or numerical semantics?

## Hypothesis
The existing WMMA CUTLASS kernel is 44 calls per Forward and about 2.8–2.9ms summed at c2/c4, with grid `[8,53,2]` and a narrow 16x16x64 tile. The existing SGLang SM70 `sm70_small_gemm` scalar-FP32-accumulate row kernel reuses each loaded FP16 weight vector across M=2/4 rows and has 6680 output rows, providing many output CTAs without a separate 2x/4x dequant/materialization. It may reduce low-batch memory dependency and owner time, but does not use Tensor Cores; matched NCU and whole Graph A/B decide. Falsified if correctness fails, kernel duration fails to improve, or service/Forward does not improve.

## Object and identities
Falcon-H1-7B-Instruct-GPTQ-Int4 on V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; production S1 remains default and exact G017 Candidate remains rollback. Only exact FP16 contiguous `model.layers.*.mixer.in_proj` calls at M=2/4, K=3072, N=6680 are routed by an explicit opt-in environment flag. M1 and all other linears stay unchanged. Do not edit GPTQ candidate, numerical tolerances, model structure, or S1/Candidate runtime binaries.

## Procedure and gates
1. Pre-run source/config review and a temporary uncommitted exact-prefix M2/M4 dispatch were used only to test the question; default remained off.
2. First run synthetic random M2/M4 correctness and Graph smoke. Then compare on the exact deployed Falcon checkpoint Mamba `in_proj` layer-0 BF16 weight converted to its runtime FP16 operand, with the frozen unchanged elementwise output contract vs `torch.nn.functional.linear`.
3. Stop at the first correctness gate failure; do not spend NCU, service or performance runs on a rejected kernel. No numerical tolerance relaxation.

## Evidence required
All gates above and exact source/binary SHA/provenance. Isolated operator numbers are diagnostic only and not adoption authority.

## Run constraints
Maintain MemAvailable >=8 GiB and use only the specified V100. Leave S1 default and G017 Candidate route intact. Never stage unrelated dirty worktree changes. CUDA profiler qualification and T001 CUPTI must not be repeated.
