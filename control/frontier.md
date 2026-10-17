# Current frontier

## G023-falcon-gpu-l1-prefix-cache
- T001 — qualify Falcon GPU prefix reuse — BLOCKED
  - frontier: Actual `extra_buffer` radix hit reports 768 tokens; same-input cache hit differs from same-config force-miss at output token 0/1. One chunk=768 diagnostic did not fix it; changing no-cache chunk size also changes one branch at token24. Locate state boundary before any performance matrix.
  - task: `control/G023-falcon-gpu-l1-prefix-cache/tasks/T001-qualify-falcon-gpu-prefix-reuse.md`
  - unblock/reopen: Source-backed explanation/fix then ordinary requests prove KV+conv/temporal same boundary, COW/stream ownership and same-input logits/256 IDs match force-miss while scheduler capability remains. Chunk-boundary hypothesis already tested once; no repeated sweeps absent new predicted fix. Preserve raw/dirty file; target V100 and >=8GiB. E035 stays closed.

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — READY
  - frontier: Resume the independent T002 c4 Decode/Prefill/cache frontier; no continuation of T006 warp-sliced candidate.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: T006 closed PERFORMANCE-NO-GO; any materially different GPTQ architecture requires new explicit authorization. Preserve S1/default/c57, E035 unowned dirty file and G023 ON_HOLD; target only authorized V100.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
