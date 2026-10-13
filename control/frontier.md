# Current frontier

## G022-falcon-v100-c4-performance
- T002 — evidence-driven low-batch optimization — ACTIVE
  - frontier: Main FPM async-join regression repaired; selected tests pass. Merged-source V100 Candidate extension rebuilt and M4 kernel invoked. Falcon S1 Graph1/2/4 and short-prompt smoke passed. Correction resolved: Candidate helper path fixed; exact-SHA dispatch verified through Graph1/2/4 and two bounded HTTP requests (32/1K prompt). Integration only. Now T002: root NCU works without permission changes; exact accepted c57 Candidate identified and matched QKV M2/M4 counters captured. QKV small-M rejected after symmetric seven-wave exact-c57 vs c57+QKV service A/B: c4 mean Output TPS differences <0.06% (1K 111.085±0.055 vs 111.142±0.079; 4K 50.863±0.013 vs 50.873±0.019), overlapping variability. Source route removed; exact c57 historical rollback/default policy unchanged. Matched GPU events admitted c3 only. Actual-c4 Graph, Prefill/cache peer outcomes and other owner remain T002 frontier. Mamba rejected; QKV removed pre-NCU; S1 default/G017 fallback intact.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: User explicitly lifted the pause. No repeat of the passed five-row CUPTI qualification; target-only V100 and >=8-GiB host floor remain mandatory. Keep S1 default/G017 Candidate fallback and unrelated dirty state.
## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
