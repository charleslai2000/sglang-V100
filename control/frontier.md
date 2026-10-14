# Current frontier

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: T002 remains ACTIVE for the broader G022 objective (actual-c4 Forward and Prefill/Cold-Warm Cache TTFT still open). E035 down group-parallel is closed by explicit user/Owner decision **REJECTED for Forward/production**: all nine strict-correct down layer×M cells lose exact c57 (~1.28–1.41× slower); UUID-bound L43/down M4 trace shows parallel−c57 kernel-duration sum +51.520µs (partial +58.336, reducer −11.904, gather/cast +5.088); not launch latency. Lock release/manifest newline defects are CPU-tested; original three missing arm_start rows are preserved as provenance gap. Stop this candidate: no tuning, matrix rerun, Forward, Cache, or service work. Any different design needs separate authorization and must first beat c57 under frozen oracle. S1 default/c57 rollback and dispatcher unchanged; broader T002 outcomes remain active.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
