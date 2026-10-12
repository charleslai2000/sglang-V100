# Current frontier

## G022-falcon-v100-c4-performance
- T001 — P0 real Falcon hardware-utilization capture — ACTIVE
  - frontier: CUPTI Forward-range metric vector is complete for Prefill EXTEND 1K/4K and Decode Graph1/2/4: three matched seven-metric captures/workload, plus two L2-throughput captures/workload. Effective range-average DRAM bandwidth is derived and cross-checked against CUPTI throughput/nominal peak; temporal peak and unrelated owner/service evidence remain open.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T001-c4-baseline-and-owner-ledger.md`
  - unblock/reopen: Safe V100-only CUPTI replay completed above the 8-GiB MemAvailable floor; RTX2060 untouched. T002/kernel optimization remains explicitly paused; do not modify kernels.

## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
