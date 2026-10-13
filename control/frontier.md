# Current frontier

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: Main merge regression in FPM async identity joins repaired/pushed at 0e7b65a97; related registered/manual tests pass. S1 and exact final Candidate extensions load with verified hashes; wrong hash rejected. Single-job V100 CUDA extension rebuild is running, then bounded Falcon S1/Candidate opt-in smoke on Graph1/2/4. T002 continues only after runtime integration gates; no new NCU yet. Mamba candidate rejected and QKV screen was removed pre-NCU; S1 default/G017 fallback intact.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
