# Plan: S2 Falcon-H1 V100 performance and 16GB feasibility

## Established

- S1 remains frozen PASS at `07343bd165855e9cd35ae63acea6cee5819c8c4e`; no S1 semantics or implementation were changed.
- Clean frozen source was tested on Tesla V100-SXM2-32GB with pinned model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`. Full controlled decode/prefill/prefix matrix, resource samples, config comparison, memory accounting, and inference are in `experiments/E019-s2-v100-performance-memory/results.md`.
- All accepted cases returned full-length nonzero outputs. No-graph 1K aggregate throughput for concurrency 1/2/4/8 was 9.6/19.2/32.9/50.6 tok/s. Regular CUDA Graphs (piecewise disabled) materially improved this; piecewise graph warmup fails with frozen Falcon `KeyError([])` and is not recommended.
- Measured component anchors: 8.09GB weights; ~0.145GB Mamba temporal state per request; ~44KiB FP16 KV/token; ~0.21GB regular graph storage. An idle-device 16K-token/four-state conventional-graph configuration peaked at 11,404MiB on the 32GB V100 and passed 7.68K×1, 4K×2, 1K×4.
- No 16GB device or reliable VRAM cap was available. The 16GB result is explicitly accounting-based, not a physical SKU qualification. Recommended candidate: 24,576 token pool; 4 running requests default; standard graphs for batch 1/2/4; piecewise graphs disabled; keep at least 2GiB actual headroom and reduce pool/disable graphs if startup budget is lower.
- The short profiler run was overhead-contaminated; kernel-level decode attribution remains unresolved and is not a basis for code changes.
- Shared V100 server was stopped; device returned to 1MiB used, 0% utilization.

## Decisive frontier

S2 benchmark and accounting task is complete. Physical V100-SXM2-16GB soak/OOM qualification is a distinct follow-on only if that hardware becomes available; no unfinished action remains in this Goal.

## Result

Conditional accounting-based V100-16GB pool GO for the stated conservative candidate profile; do not describe as direct 16GB hardware acceptance. See `experiments/E019-s2-v100-performance-memory/results.md` for matrix, deploy configuration, per-card/8-card capacity estimates, power, and limitations.
