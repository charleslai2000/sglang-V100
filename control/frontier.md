# Current frontier

## G022-falcon-v100-c4-performance
- T011 — QPN2 multiwarp group-aligned execution — ACTIVE
  - frontier: Repeat 4K/c4 E2E under matched queues (initial candidate E2E regression conflicts with higher Output TPS); qualify multi-request Mamba/KV state isolation/slot reuse and hidden/logit boundary; record per-request TTFT/ITL plus 4K Prefill. Matched GPU event evidence shows b4 Graph-expected forward −29.5%; sidecar graph_key is null.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T011-qpn2-multiwarp-group-reduction.md`
  - unblock/reopen: Operator gate and bounded NCU passed. Continue owner authorization, target V100 only, MemAvailable≥8GiB, protect E035/unowned changes, S1/default/exact-c57 unchanged; G023 ON_HOLD.

## G023-falcon-gpu-l1-prefix-cache
- T001 — qualify Falcon GPU prefix reuse — BLOCKED
  - frontier: Actual `extra_buffer` radix hit reports 768 tokens; same-input cache hit differs from same-config force-miss at output token 0/1. One chunk=768 diagnostic did not fix it; changing no-cache chunk size also changes one branch at token24. Locate state boundary before any performance matrix.
  - task: `control/G023-falcon-gpu-l1-prefix-cache/tasks/T001-qualify-falcon-gpu-prefix-reuse.md`
  - unblock/reopen: Source-backed explanation/fix then ordinary requests prove KV+conv/temporal same boundary, COW/stream ownership and same-input logits/256 IDs match force-miss while scheduler capability remains. Chunk-boundary hypothesis already tested once; no repeated sweeps absent new predicted fix. Preserve raw/dirty file; target V100 and >=8GiB. E035 stays closed.

## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
