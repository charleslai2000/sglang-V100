# E037 results

## OBSERVED

- No E037 GPU/service workload has run yet.
- Current platform check (2026-10-17): target V100 was idle at 7 MiB; MemAvailable >16 GiB. Current preflight device: V100 memory 16,384 MiB, 7MiB idle, 135MHz SM/877MHz HBM, 25.13W, 41°C, no active throttle; RTX2060 had unrelated ~2.3GiB use but was never targeted. Model config SHA `ca8c91f8a18a4917f9580ce1cbcb015a6f7cee2b24022cf6c3e8c94d6943a7f7`; exact S1 common_ops SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; exact c57 SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`. Checkpoint config: GPTQ int4/group128, desc_act=true/static_groups=false, 44 Mamba2 layers, 24 heads/layer, d_head128, state256, conv K4, chunk256. Runtime load path still to verify in isolated child processes.
- G023 cache-prefix goal remains separate and BLOCKED; cache work is paused for this G022 goal. Exact c57 ×4 stress defect and E035 production rejections remain unchanged.
- Worktree contains unowned dirty `experiments/E035-g022-actorder-q4-wmma/gptq_candidate_v8_packed_word_reuse.cu`, plus separate G023 task/report/artifacts from the preceding session; preserve them and exclude from G022 commits.
- Prior G022 resources/HBM CUPTI five workload qualification and historical G020/G021 traces exist; never treat them as the requested short-context b1–32 current matrix without identity/scope match.

## INFERRED

- Calculated Mamba state per request from model dimensions and V100 dtypes: conv `[1,12800,3]` FP16 plus temporal `[1,24,128,256]` FP32 ×44 layers = 138,412,800 bytes (132 MiB) per slot. Independent b32 state alone is 4.125GiB before KV, 7.29GiB weights, graph/workspaces and runtime headroom. With Radix disabled, source sets max_mamba_cache_size=max_running_requests; with radix on and overlap extra_buffer, slot ratio is 5. This stage uses `--disable-radix-cache`, so that ratio is not the b32 limiter.
- **No GPU performance workload was run.** The required safe envelope is MemAvailable ≥8GiB on a 16GiB host, which permits ≤8GiB total GPU allocation. The 7.29GiB model plus 132MiB/request state already puts b1 near 7.42GiB before KV, CUDA context and workspaces; b32 requires >11.4GiB before those extras. Thus the full requested matrix, particularly b32, cannot be admitted while retaining the user’s safety margin. No lowering the reserve or using dummy/aliased states was attempted. This is a feasibility stop, not performance or counter evidence.
- S1 GPTQ backend changes at larger M are a strong source-supported hypothesis: unshuffled GPTQ calls the native quantized M=1..8 alternate path through M=8 and reconstructs dense FP16 weights then cuBLAS GEMM for M>8. The exact Falcon `desc_act=true` configuration selects no-shuffle on SM70. Runtime proves this branch only if trace markers/symbols are captured. Do not infer candidate c57 M8+ route; c57 opt-in is guarded to M<=4 and other cases fall back to generic `gptq_gemm`, which itself reconstructs for M>8.

## UNRESOLVED

- Actual b1/2/4/8/16/32 complete-forward latency and throughput under the requested real prefill.
- CUPTI full-forward SM/Tensor/DRAM/L2 vectors with exact metric names and replay validation.
- Actual b4→b8/b16 dispatch transition and GPTQ call counts/methods per projection.
- Full-forward kernel/owner time attribution, request-state uniqueness, allocator/Graph/temp memory, and safe b32 feasibility.
- 1K b1/b4 current anchors and their scope alignment; no 3090 direct run is planned.
- Final single recommended mature implementation direction with c1/c2/c4 budget and falsification criterion.

## Bounded answer

No diagnosis or optimization outcome is established yet. This attempt stopped before GPU execution because the real-model memory budget cannot preserve the required MemAvailable reserve, especially for b32. The S1 GPTQ M>8 fallback remains a static source-supported hypothesis, not a runtime finding. No performance claim until actual-forward evidence is complete.
