# Current frontier

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: T002 packed-word Q4 reuse is DONE as **NUMERICAL-NO-GO**: exact c57 and candidate both fail the corrected frozen pre-cast FP32 oracle on the same eight L43/down ×4 outputs; no service tests followed. Candidate source/binary/evidence are preserved in E035 experiment scope; tracked production edit restored; S1/default/c57 rollback unchanged. Broader G022 objective remains ACTIVE for actual-c4 Forward and Prefill/Cold-Warm Cache TTFT. E035 down group-parallel remains Owner REJECTED and closed.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
