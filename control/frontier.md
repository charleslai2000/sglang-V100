# Current frontier

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: Final FP64 continuation is NUMERICALLY-VALID / PERFORMANCE-NO-GO: exact-SHA correctness/state/output qualification passed; matched service Output TPS regressed 7.55–10.15% vs exact c57, exceeding the <=3% gate. Keep opt-in disabled. Broader G022 remains open for actual-c4 Decode Event/counters and Prefill/Cold-Warm Cache TTFT; results/control record the bounded evidence and gaps. Historical packed-word Q4 is separate NUMERICAL-NO-GO; E035 group-parallel remains Owner REJECTED. Preserve S1 default and exact-c57 rollback.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
