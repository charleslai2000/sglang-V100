# LLM Inference Performance Metrics V1

Status: **FROZEN — authoritative measurement contract for G021 / R17**

Scope: Falcon-H1-7B + SGLang on Tesla V100-SXM2-16GB. This is a performance measurement system, not a warranty system. Performance thresholds cannot substitute for evidence completeness; Output TPS alone is not complete inference performance.

## Frozen boundaries

- G017 remains ACCEPTED. S1 remains default; Candidate is exact-path/SHA opt-in only.
- Reuse G018–G020 and G021 archived data. Historical RTX 3090 same-model observations remain an independent comparator until workload and measurement boundaries are reconciled; never mix different batch conditions as a direct comparison.
- Do not change GPTQ, Attention, Mamba, or CUDA Graph mathematical semantics; do not affect RTX 2060 or unrelated tasks; do not develop a new compute kernel.
- Preserve a usable GV100 NCU toolchain. Codex has sudo authority to resolve ordinary measurement, compile, IPC, and process wiring problems. Use target-only, non-interfering profiling; do not take over global profiling resources or disturb other GPUs/workloads.
- Keep per-kernel, per-Graph, and whole-device boundaries distinct. NVML device GPU Busy/Memory Busy must not be represented as SM or HBM peak utilization.
- Use statuses `MEASURED`, `PARTIAL`, `DISABLED`, `UNSUPPORTED`, `NOT_RUN`, `INVALID`. Unsupported is not zero. Do not simulate cache hits or state restoration.

## A. GPU hardware

Collect Prefill and Decode separately:

1. SM Active and SM Compute Throughput;
2. Tensor Active and Tensor Pipe Utilization;
3. FP16/FP32 pipeline utilization;
4. DRAM read/write GB/s and % theoretical peak;
5. GPU L2 hit rate and L2 throughput;
6. PCIe H→D/D→H GB/s;
7. achieved occupancy;
8. warp issue efficiency and stall reasons;
9. kernel launch overhead and CUDA Graph coverage.

Prefer root NCU aggregate on a real CUDA Graph, target-only and safely bounded; use selected real kernel-level counters when needed. Audit DCGM absence; evaluate safe target-only install/collection only if needed and without impacting another GPU or global profiling ownership. For every unavailable metric, document exact tool, hardware, queried metric, command and failure evidence; no blanket UNKNOWN.

## B. GPU Prefill / Decode

From actual SGLang worker and scheduler capture actual prefill tokens, GPU start/end/active duration/TPS; decode output tokens, per-step GPU latency, active duration/TPS; actual batch distribution and CUDA Graph replay ratio. Repair the existing FPM subscriber zero-record issue by auditing publisher, subscriber, IPC and lifecycle. A second recurrence requires source/log forensics, not a new generic profiling framework. All CPU/GPU events must have cross-checkable time relationships; overlapping paths must not be double-counted.

## C. Scheduler / request lifecycle

For each request ID capture arrival → enqueue → admission → cache lookup/restore → GPU prefill → first token ready → first token delivered → decode steps → completion. Report Queue Wait, Admission Wait, Running/Queued, Scheduling Gap, Preemption, first-token GPU compute latency, and client TTFT separately. Explain the observed 4K/c4 13–14s TTFT tail without merging queue, prefill, and cache waits.

## D. HiCache

Verify actual local-fork support for GPU L1, Host L2, Disk L3, Radix/HiMambaRadixCache, Attention KV plus Mamba convolution/SSM state, and prefix-frontier restoration correctness. Measure Cold, Warm, Pressure/Eviction/Restore only for real implemented paths. Per request capture logical matched tokens, successfully reused tokens, exclusive L1/L2/L3 tokens, recomputed prefill tokens, H→D bytes/time, Disk read/write bytes/time, Cache Wait, effective restore throughput, and TTFT/E2E change. Logical host/disk hits alone do not count as reuse; KV and Mamba state must correspond to one legal prefix frontier. Mark unimplemented features UNSUPPORTED; no synthetic hits or fabricated restore performance. Request architecture decision only if frozen state ownership must change.

## E. Service performance

Use uninstrumented benchmark runs for Input TPS, Output TPS, RPS, TTFT/TPOT/ITL/E2E P50/P95/P99, error rate, and concurrency saturation. Workload families: 1K/4K with 256-token output, Prefill-heavy, Decode-heavy, long context, multi-turn Cache. Scan submitted concurrency c1/4/8/12/16/20/24. For every run record actual running, queued and GPU batch. Determine the V100 runnable upper bound from safe memory budget; never report an unrun point. At least 3 independent runs per point; adequate request sample size for important P99 tails, with request count, estimator and dispersion.

## F. GPU decode owner ledger

Reuse real G020 traces and verify owner of GPTQ gate-up/down, Other GEMM, Attention, Mamba, Elementwise/Norm, unknown GEMV/GEMM, and GPU execution gaps. Investigate c1 ~7.8ms unknown GEMV/GEMM and batch4 ~7.3ms Other GEMM using source/call path and launch geometry. Distinguish kernel-duration sum, interval union, owner-exclusive time and whole-forward elapsed. No owner may be guessed without source and time-boundary evidence.

## G. Data and delivery contract

Every metric row binds GPU UUID, source commit/model revision, S1/Candidate binary SHA, workload/request/run ID, prompt/output tokens, cache mode/cold-warm state, submitted concurrency/actual batch, measurement boundary, CPU timestamps/GPU timing source, tool version, profiler on/off, sample count, value/unit, status, and provenance. Deliver a unified benchmark matrix, raw data/artifact pointers, analysis scripts, reproducible configuration, artifact manifest and complete research report. Each research phase is an independent commit/push with remote SHA verification.

## Required final answers

1. Actual model GPU Prefill and Decode speed?
2. Service Output TPS, TTFT, ITL, E2E?
3. Actual SM, Tensor, HBM utilization?
4. GPU time by major kernel owner?
5. Queue/Admission/Gap time outside GPU compute?
6. Did Cache hit actually reduce GPU Prefill?
7. Did Host/Disk restore overhead erase Cache benefit?
8. First saturation point as concurrency rises?
9. Same-workload, same-boundary V100 vs RTX3090 difference?
10. Most valuable next optimization owner and recoverable budget?

Every performance conclusion must include actual speed, time attribution and hardware resource utilization. Missing one prohibits claiming the corresponding bottleneck analysis complete.
