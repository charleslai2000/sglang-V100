# Goal: S2 Falcon-H1 V100 performance and 16GB pool feasibility

## Authorized outcome

Produce a repeatable Falcon-H1 GPTQ serving benchmark on one Tesla V100-SXM2-32GB at frozen S1 commit `07343bd165855e9cd35ae63acea6cee5819c8c4e`, account for memory from measured deltas, and make an evidence-based deployment GO/NO-GO recommendation for V100-SXM2-16GB.

## Scope and constraints

- Benchmark only the pinned Falcon-H1 checkpoint/revision and frozen S1 semantics, GPTQ route, and serving backends.
- Measure decode concurrency 1/2/4/8 where resources allow at approximately 1K/4K/8K contexts; prefill 1K/4K/8K; and prefix + 256 decode.
- Record request and aggregate token rates, inter-token latency, TTFT, GPU utilization, peak VRAM, and memory components using actual telemetry/deltas where available.
- Evaluate a conservative 16GiB budget, including safe request limits, total-token/cache reservation, graph/workspace headroom, OOM margin, expected throughput, and an 8-card aggregate estimate. Run an actual constrained-memory test if reliable controls permit.
- No kernel/runtime architecture development. S1 boundaries are frozen; only fix a correctness regression if a benchmark concretely exposes one.
- Preserve unrelated worktrees and leave the V100 idle when measurement runs finish.

## Completion condition

A repeatable benchmark matrix and result report exist; no correctness regression is observed; memory accounting explains measured peaks; 16GB feasibility has a clearly labeled conditional accounting recommendation and candidate configuration. Physical 16GB hardware qualification is not claimed because no such device or reliable cap was available.

## Result

Completed on 2026-10-04. Full result: `experiments/E019-s2-v100-performance-memory/results.md`. Outcome: conditional accounting-based pool GO for V100-16GB with conservative token/request caps; not physical SKU qualification. Frozen S1 unchanged and shared V100 released.