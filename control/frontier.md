# Current frontier

## G022-falcon-v100-c4-performance
- T003 — short-context batch scaling diagnosis — BLOCKED
  - frontier: No GPU workload admitted: 16GiB host with required ≥8GiB MemAvailable cannot fit 7.29GiB weights plus independent Mamba state and runtime/KV overhead. Await authorized larger-memory host or Owner-approved change to reserve.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T003-short-context-batch-scaling-diagnosis.md`
  - unblock/reopen: Target V100 UUID only, MemAvailable >=8 GiB, independent Mamba state per request, preserve all existing dirty files and E035 dispositions. Cache remains paused.

## G023-falcon-gpu-l1-prefix-cache
- T001 — qualify Falcon GPU prefix reuse — BLOCKED
  - frontier: Actual `extra_buffer` radix hit reports 768 tokens; same-input cache hit differs from same-config force-miss at output token 0/1. One chunk=768 diagnostic did not fix it; changing no-cache chunk size also changes one branch at token24. Locate state boundary before any performance matrix.
  - task: `control/G023-falcon-gpu-l1-prefix-cache/tasks/T001-qualify-falcon-gpu-prefix-reuse.md`
  - unblock/reopen: Source-backed explanation/fix then ordinary requests prove KV+conv/temporal same boundary, COW/stream ownership and same-input logits/256 IDs match force-miss while scheduler capability remains. Chunk-boundary hypothesis already tested once; no repeated sweeps absent new predicted fix. Preserve raw/dirty file; target V100 and >=8GiB. E035 stays closed.

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: Owner rejected FP64 candidate for production after scoped numerical pass and observed service regression; service order was c57 matrix then FP64 matrix, three repeats/cell, not alternating pairs, so retain as descriptive evidence only. FP64 shared experimental wiring archived and withdrawn; no further FP64 GPU/service/cache work. Other G022 objectives remain separate. Preserve S1 default and exact-c57 rollback.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
