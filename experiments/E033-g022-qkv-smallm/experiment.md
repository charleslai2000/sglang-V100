# E033 — G022 exact-weight Falcon QKV SM70 small-M candidate

## Question
Can the mature local SM70 M2/M4 FP16 small-GEMM reuse kernel safely accelerate Falcon-H1's source-attributed fused attention QKV projection `[M,3072] -> [M,2048]`, which currently launches one cublas GEMV at c1 and CUTLASS WMMA at c2/c4?

## Hypothesis
The existing small-M kernel passes initial exact-checkpoint M2/M4 output screening (per exploratory run documented in E032). Its FP32 accumulation and row-shared vector loads might reduce source-attributed QKV kernel time. This is falsified if exact integrated QKV source dispatch fails frozen output/state/Graph gates, matched NCU shows no improvement, or full service/Forward gains are absent/regressive.

## Object and identity
Falcon-H1-7B-Instruct-GPTQ-Int4, exact frozen S1 runtime extension SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; target V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`. QKV checkpoint source keys are `model.layers.0.self_attn.{q,k,v}_proj.weight`, concatenated in load order into `[2048,3072]`, BF16 stored / FP16 V100 runtime. Existing S1 remains default and exact G017 Candidate rollback remains untouched.

## Procedure and gates
1. Temporary/uncommitted exact source-class route is opt-in false by default and shape/device/alignment/prefix guarded. First run actual checkpoint exact-weight output comparison against the production `torch.nn.functional.linear` FP16 output with the frozen tolerance; then Graph replay and state check. Stop on failure; no tolerance relaxation.
2. If correct, matched-geometry NCU before/after on exact c2/c4 operands and exact projection kernel filter, under V100-only supervisor and host floor.
3. If NCU supports the bottleneck hypothesis, real SGLang production Graph GPU Step c1/c2/c4 A/B, then repeated uninstrumented service Output TPS/ITL/TTFT at 1K/4K.
4. Include prefill throughput, Cold/Warm Cache TTFT, GPU/KV/Mamba capacity and VRAM. No concurrency change. Reject if any required correctness or service/performance gate fails.

## Controlled inputs
Only one feature flag varied. Model/checkpoint and data identities frozen; exact source-aware module site `self_attention.qkv_proj`. No T001 CUPTI repeat. No other code or model paths changed.

## Run constraints
S1 default preserved; Candidate rollback preserved. Target V100 only, MemAvailable >=8GiB, no NCU broad profiling or unrelated process/device use. Any experimental production diff must remain uncommitted until full qualification; unrelated dirty worktree preserved.
