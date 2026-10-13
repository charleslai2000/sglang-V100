# E031 — G022 real-Decode projection source attribution

## Question
Which Falcon-H1 production Linear modules and actual input/output geometries own the observed low-batch FP16 cuBLAS GEMV and c4 CUTLASS GEMM calls, and how many times does each source projection execute per real Decode Forward?

## Prediction
The 44 decoder blocks each invoke one attention QKV and one attention output Linear, plus one Mamba input and one Mamba output Linear. The existing trace's c4 CUTLASS family and c1 GEMV signatures should map to those production projection sites, while GPTQ MLP gate_up/down and FP32 vocabulary lm_head remain separate paths.

## Object and scope
Falcon-H1-7B-Instruct-GPTQ-Int4 local pinned model, exact frozen S1 extension SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`, V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`. This is a bounded source-attribution run, not optimization qualification or service benchmark.

## Controlled inputs and procedure
- Use actual Falcon server and greedy Decode request at 1K prompt with production c1/c2/c4 admitted batch, preserving graph-disabled eager Linear dispatch only for this narrow callsite attribution (record explicit deviation from production CUDA Graph replay).
- Temporary `/tmp` Python startup instrumentation wraps every concrete SGLang `LinearBase` subclass forward and adds both `torch.profiler.record_function` source scopes and NVTX; emits prefix, quant method, input/output and weight shape/dtype. It does not edit tracked production implementation or alter math. Preload via `sitecustomize.py` and run actual server with CUDA Graph disabled solely for attribution; Kineto `/start_profile profile_range=decode` captures real eager Decode forwards. Each source scope contains launch-correlation-linked GPU kernel events. Report eager-dispatch deviation and do not treat its timing as production Graph performance. GPTQ calls bypass unquantized CUTLASS linear but their source module prefixes are attributed. Preserve full trace/script/log hashes and process/resource records.
- Frozen server parameters and prompt/model are those from E029; only batch size and graph-disabled attribution mode vary. No concurrent workload or device other than designated V100.

## Measurements needed
For every matching real Forward, exact module prefix/callsite, class, input/output shape/dtype, count, launch kernels and correlation; per-site duration from CUDA kernels (separate kernel sum and union), graph/eager mode; model/extension identities; MemAvailable and GPU allocation; failed or unmatched modules.

## Run constraints
Host MemAvailable >=8 GiB throughout; target V100 UUID only, idle before/after; no change to server defaults, source implementation, quantization math, or production service concurrency. Do not repeat any CUPTI metric qualification or use NCU. Reuse existing API/range instrumentation. Large traces remain in `/mnt/data/pc01/sglang-v100-migration/experiments/G022/`.
