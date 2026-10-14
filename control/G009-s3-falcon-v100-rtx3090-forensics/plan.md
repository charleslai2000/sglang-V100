# Plan: S3 Falcon-H1 V100-only optimization (RTX 3090 fixed reference)

## Established

- Scope updated by user: V100 only is active. RTX 3090 supplied TPS is fixed external reference; do not access/build/run/profile/requalify Falcon on it. E021 evidence is `/home/ubuntu/workspaces/3rdparty/sglang/.pi/experiments/E021-s3-v100-optimization/results.md`.
- Correctness baseline is frozen at `07343bd165855e9cd35ae63acea6cee5819c8c4e`; S2 is complete and frozen. S2 report: `experiments/E019-s2-v100-performance-memory/results.md`.
- S2 direct V100 results establish standard CUDA Graphs as effective; piecewise graph capture fails on this frozen Falcon path. Do not reopen without new evidence.
- S2 profiler attempt distorted throughput and did not provide usable owner attribution. S3 uses CUDA-only Nsight Systems and targeted NCU/device counters. Both hosts report CPU sampling unavailable due kernel perf restrictions; GPU timeline/device metrics remain to be qualified.
- Active target is remote V100-SXM2-32GB only. RTX 3090 is a fixed historical reference per updated user scope; do not access, build, run, profile, or requalify it. Two unrelated local SGLang services exist; preserve them.
- Checkpoint config/index/tokenizer hashes match between hosts; SHA/revision is frozen as in S2.
- Clean local S1 source archive is at `/home/ubuntu/workspaces/3rdparty/.s3-build/s1-clean` with frozen SHA marker. Existing `sglang-V100` repo is at baseline but has four unrelated dirty scripts/patches; none were modified. The installed local SGLang editable checkout path is missing; do not repair or overwrite its environment in place.
- S1 kernel CMake defaults to general architectures on x86_64 CUDA>=12.8 with SM86/FA3 support; `SGL_KERNEL_V100_ONLY` was the V100 installer option, not general source default. Existing 3090 wheel is newer 0.4.6; its SM86 extension lacks schema(s) required by S1 Python imports (e.g. `moe_fused_gate`). Frozen Python against this binary is not runtime-closed yet. Isolated build outputs stay under `.s3-build`.
- Nsys versions: V100 2024.6.2, 3090 2025.1.3. NCU versions: V100 2025.1.1, 3090 2025.2.1. CPU sampling unavailable on both (perf_event_open restricted); CUDA GPU traces/counters not yet tested.
- Correct S2 Mamba accounting by separating fixed runtime, reserved state pool and incremental live-request bytes; prior 0.256GB/request and ~0.145GB/request statements are unreconciled.

## Decisive frontier

V100-only: obtain clean production-like Nsight Systems CUDA activity (client-only capture missed workers; profiler-injected fresh-server capture segfaulted in `kvcache.store_cache` warmup). NCU returns `ERR_NVGPUCTRPERM`. Try unprofiled JIT prewarm and a safe initialized capture or CUPTI alternative. Then targeted owner counters and separate decode/prefill A/B optimization. RTX 3090 fixed reference only.

## Next action

S2 runner output-path collision was contained: the current altered `s2-measurements.jsonl` and GPU companion are preserved at `/hy-tmp/sglang-logs/forensic-s3-20261005/` with SHA256; independent S2 capacity/graph raw outputs remain intact. E021 has a protected dedicated output path. V100 is idle. Prewarm `kvcache.store_cache` JIT outside nsys before a capture attempt; validate kernel events before long profiling. Keep profiler captures separate from throughput.
