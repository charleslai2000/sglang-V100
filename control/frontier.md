# Current frontier

## G023-falcon-gpu-l1-prefix-cache
- T001 — qualify Falcon GPU prefix reuse — BLOCKED
  - frontier: Actual `extra_buffer` radix hit reports 768 tokens; same-input cache hit differs from same-config force-miss at output token 0/1. One chunk=768 diagnostic did not fix it; changing no-cache chunk size also changes one branch at token24. Locate state boundary before any performance matrix.
  - task: `control/G023-falcon-gpu-l1-prefix-cache/tasks/T001-qualify-falcon-gpu-prefix-reuse.md`
  - unblock/reopen: Source-backed explanation/fix then ordinary requests prove KV+conv/temporal same boundary, COW/stream ownership and same-input logits/256 IDs match force-miss while scheduler capability remains. Chunk-boundary hypothesis already tested once; no repeated sweeps absent new predicted fix. Preserve raw/dirty file; target V100 and >=8GiB. E035 stays closed.

## G022-falcon-v100-c4-performance
- T009 — c57 Q4 vector-load optimization — ACTIVE
  - frontier: Bind exact-c57 binary to source/build identity; establish real load widths/alignment and inspect existing SASS before any candidate edit or timing.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T009-c57-q4-vector-loads.md`
  - unblock/reopen: Explicit T009 authorization; V100-only, MemAvailable>=8GiB; preserve exact-c57/S1/default, protected E035 file and G023 ON_HOLD.

## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
