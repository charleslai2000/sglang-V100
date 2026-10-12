# G022 — Falcon-H1 V100 c4 Performance Optimization

## Authorized outcome

Optimize Falcon-H1-7B inference on the local Tesla V100-SXM2-16GB under a fixed 16-GiB budget and **c1/c2/c4 working concurrency**, minimizing actual GPU Decode Step execution time first. c1 targets minimum single-token GPU latency and ITL; c2 targets low-batch scaling efficiency; c4 targets maximum actual GPU Decode TPS and service Output TPS. c8/c16 are optional safe-condition references only and are neither production targets nor completion gates. Do not increase max concurrency as a substitute for compute optimization.

Deliver defensible c1/c2/c4 GPU-forward time budgets, implement and correctness-qualify profiler/NCU-evidence-selected optimizations, and show repeated same-load S1 → G017 Candidate → new implementation results for GPU Decode and service, including resource/capacity impacts. Explicitly prioritize GPTQ, Other GEMM/GEMV, memory dependency, SM/Tensor utilization, and CUDA Graph execution efficiency where evidence supports them; investigate order-of-magnitude potential without assuming it is achievable. Preserve Prefill and Cold/Warm Cache TTFT as R17-level performance metrics alongside Decode. Continue through ordinary profiler/build/interface issues; do not stop merely because an individual metric is unavailable. G021/R17 remains its own active metrics-contract goal; use its frozen LLM Inference Performance Metrics V1 and existing evidence without reopening generic profiling infrastructure.

## Frozen authority and constraints

- Preserve the accepted G017 S1 path/default and the exact-identity G017 Candidate as available comparison routes; do not silently change dispatch/default or weaken acceptance.
- Preserve GPTQ-v1 Q4/group128, original `g_idx`, qzero/scales, and Attention/Mamba/Graph numerical semantics. Production correctness authorities and tolerances remain those already frozen for the relevant route; no tolerance relaxation.
- Target only Tesla V100-SXM2-16GB UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`. Never profile/use RTX 2060 UUID `GPU-04ad2dd9-245d-0e7b-c96a-12da83a92b99` or interfere with unrelated tasks.
- Respect 16-GiB VRAM and the existing 8-GiB host MemAvailable abort floor. No OOM probing, host-wide profiler/policy changes, CUDA/driver/PyTorch changes, or global profiling ownership.
- Do not re-establish profiling infrastructure. Reuse G020/G021 reports, NCU workflow, existing bounded forward/request instrumentation, and current artifacts. Keep service TPS, CUDA Event forward time, kernel-duration sums, and GPU active union distinct.
- Distinguish submitted c4 from actual running GPU batch. Measure stable admitted c4 forwards; report if the server does not sustain actual batch 4 and establish the capacity reason rather than mislabeling c4.
- 1K/4K context and 256/512 output are the baseline workload conditions. c8/c16 are reference-only and only if safe; not a required sweep.
- Reuse mature implementations first. Do not revive previously rejected or incorrect candidate routes without fresh source-backed reason and exact-path qualification. Kernel/operator wins alone are insufficient: every adopted optimization must improve real c4 GPU-forward Decode and uninstrumented service output throughput without unacceptable c1/c2 regression.

## Required completion evidence

1. Reproducible c1/c2/c4 baselines and final comparisons at 1K/4K × 256/512 outputs, with S1, exact G017 Candidate, and final route identities; record actual admitted/running GPU batch, queued requests, cache condition, graph replay, errors and repetitions. c4 is the principal throughput point, c1 the single-token latency point, c2 scaling efficiency. Include actual GPU Decode Step time for every point.
2. Per-request Cold/Warm TTFT, ITL/TPOT, E2E and Queue Wait, explicitly separating scheduler lifecycle from GPU time. Record actual prefill work and GPU Prefill TPS; Decode GPU step latency/TPS from stable admitted forwards; client Output TPS separately. Prefill and Cache Cold/Warm TTFT remain R17-level co-equal metrics.
3. Evidence-backed c1/c2/c4 GPU-forward budget across GPTQ gate_up/down, Other GEMM, GEMV, Attention, Mamba, norm/elementwise, graph/launch/scheduling gaps. Resolve or bound unknown time. Pair owner durations and applicable NCU counters with the same measured workload and boundary; do not force a complete sum from overlapping/non-joined data.
4. Before/after SM, Tensor, DRAM/HBM, L2, issue/stall and occupancy counters at supported boundaries; graph replay coverage; KV/Mamba/Graph memory use and safe headroom. Explicitly disposition unsupported counters.
5. For every optimization, jointly report c1/c2/c4 GPU Decode Step Time, GPU Decode TPS, service Output TPS, TTFT, ITL/TPOT, SM/Tensor/HBM, VRAM and KV/Mamba capacity, correctness and rollback. Require true GPU-forward and service A/B improvement; preserve rejected results and provide a safe rollback path.
6. Deliver report, reproducible config/scripts, raw artifact pointers and SHA manifest; commit/push scoped work and verify remote SHA.

Completion requires a real c4 GPU-forward Decode improvement **and** uninstrumented service Output TPS improvement at matching workload, with correctness and resource constraints satisfied. State the measured ceiling/limitation only if the evidence supports it; do not assume 150/200/300/400 TPS is achievable.
