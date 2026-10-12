# Current frontier

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: Exact-SHA Candidate-vs-S1 GPTQ NCU is matched at M1/M2/M4 for gate-up/down and reproduces lower kernel time/long-scoreboard with higher math/not-selected/DRAM-write pressure. This is archived Candidate, not a new change or full-path qualification. Close a concrete owner/source attribution or missing performance context before selecting another implementation; any new change still requires correctness → NCU → GPU Forward A/B → uninstrumented service A/B.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
