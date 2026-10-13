# Current frontier

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: E031 attributes projection sources. E032 Mamba in_proj small-M failed exact-weight correctness and is rejected. E033 QKV small-M passed exact-weight direct dispatch + Graph, but the temporary route was removed before NCU because prior matched replay crossed the host 8-GiB floor; no implementation remains. Next requires a demonstrably safe matched-NCU route, then full Graph Forward c1/c2/c4 and uninstrumented service/Prefill/cache A/B. S1 default/G017 rollback unchanged.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
