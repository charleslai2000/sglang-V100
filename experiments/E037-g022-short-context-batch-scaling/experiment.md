# E037 — G022 short-context Falcon batch scaling

## Question
For the exact local Falcon-H1 V100 model and S1 compute route, what are the true complete Decode forward latency/throughput and CUPTI hardware activity at independent actual GPU batches 1/2/4/8/16/32 after a real 128-token prefill? Where does the GPTQ backend/traffic change, what components dominate and which single mature implementation direction offers the most plausible recoverable c1/c2/c4 time?

## Hypotheses to resolve
- H1 small-batch latency may be CTA/issue/dependency/latency limited rather than HBM saturation.
- H2 threshold changes in GPTQ reconstruction/backend may change arithmetic and memory traffic.
- H3 non-GPTQ compute or Mamba state reads/writes may explain some scaling.
- H4 historical utilization labels/ranges/denominators or device identities may not match this full-forward scope.
These are hypotheses only; decide by evidence.

## Experimental object and identities
- Repository current main at task start `3b4393c63d943cf890beba38ac9d7c7bfde47d24`.
- Model local `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`, revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; model config SHA to be recorded.
- S1 common_ops SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; exact-c57 comparison common_ops `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`, native method valid only M<=4. S1 default and c57 source not changed.
- Only V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; host MemAvailable >=8 GiB. Never use RTX2060.

## Conditions
- Real service ModelRunner / ordinary Prefill, short prompt=128 token IDs, fixed input IDs and fixed Mamba/KV length per cell. Each concurrent request gets a distinct Mamba state slot and real KV sequence. Capture fixed continuous Decode window after Prefill, all Mamba slots private; use existing graph/FPM and CUPTI range replay.
- S1 actual b1/2/4/8/16/32, subject to exact free-memory gates. Exact c57 independent b1/2/4 and actual runtime fallback for larger M: record exact selected method, do not label high-M as c57-native.
- Warm identical Graphs per isolated process, capture only required Graph key, uninstrumented Event repeats and CUPTI application replay; interleave cell order and >=3 independent repeats. Separate kernel-level trace/NCU from full forward event/CUPTI timing.
- 128-token prompt is diagnostic and not directly a production 1K/4K measurement; reuse 1K historical anchors with their original scope and identity explicitly marked.

## Procedure
1. Check HEAD/dirty state and source/binary runtime path, inspect exact GPTQ dispatch/reconstruction branch and all call sites. Preserve the unrelated dirty E035 packed-word file and G023 files.
2. Calculate lower-bound resident bytes for weights, KV at short contexts, Mamba slots (conv and temporal SSM for all 44 layers), runtime/Graph and any reconstruction temporaries. Derive max safe b without speculative OOM. If actual b32 cannot fit, record allocator/model memory evidence and safely measure remaining cells.
3. Configure isolated server with cache disabled to avoid confounding prefix reuse; but retain all normal model state, overlap defaults as dictated by compute config, actual batch admission and Graph. Ensure every request receives independent Mamba pool index. Real prefill then Decode; no synthetic/aliased states.
4. Collect uninstrumented CUDA Event complete-forward windows, existing G022 CUPTI CUDA ApplicationReplay full metric vectors with same cell initial state on each pass, and separately selected kernel traces/NCU where safely bounded. Verify no dropped ranges, replay convergence and device UUID. Sample GPU clocks/power/temp/headroom.
5. Analyze curves, full-forward component attribution, backend/method thresholds/call counts and counter units. Locate historical 176 GEMV/GEMM calls by exact source/runtime boundary or mark unresolved only if non-dominant. Compare to G020/021/022 data only by matching identity and boundary.
6. Review mature donor constraints and historic 3090 evidence without accessing that GPU. Rank recoverable complete-forward time with known limits; recommend exactly one implementation direction, file/mechanism, risk and test conditions.

## Required measurements
For each run/cell: actual M/batch, Graph key, prompt/KV/Decode position, selected backend/method per projection shape/dtype, request state uniqueness, number of graph replays and ranges; CUDA Event ms/step and distribution; GPU tokens/s and amortized ms/token; CUPTI SM Active, SM Throughput, active warps/occupancy, Tensor (exact names and active/elapsed), DRAM bytes and throughput, L2 hit, clocks/power/temp and memory; allocator weights/KV/Mamba/Graph/temp use. Keep denominators distinct and never average per-kernel percentages to synthesize forward.
