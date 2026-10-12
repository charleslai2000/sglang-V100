# Current frontier

## G022-falcon-v100-c4-performance
- T001 — P0 real Falcon hardware-utilization capture — ACTIVE
  - frontier: T001 hardware PASS (commit `50dde9932`) accepted by user. T002 resumed: unify c1/c2/c4 timeline+CUPTI+NCU rank; reconcile HBM wording; then validate one matched-geometry, falsifiable hotspot hypothesis through correctness → NCU → GPU Forward A/B → uninstrumented service A/B.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
