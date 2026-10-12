# Current frontier

## G022-falcon-v100-c4-performance
- T001 c1/c2/c4 baseline and owner ledger — ACTIVE
  - frontier: Establish actual admitted GPU batch and stable GPU Decode Step/service baseline for S1 and exact G017 Candidate; reuse G020/G021 without rebuilding profiling infrastructure and preserve Prefill/Cold-Warm Cache metrics.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T001-c4-baseline-and-owner-ledger.md`
  - unblock/reopen: Target V100 UUID only; 8-GiB MemAvailable floor; preserve unrelated dirty worktree.

- T002 evidence-driven low-batch optimization — READY
  - frontier: Select first optimization only after T001 produces a bounded c1/c2/c4 GPU owner/time budget and exact hotspot identity.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: T001 identifies a defensible optimization owner; preserve numerical semantics, S1/Candidate fallback and c1/c2/c4 comparison.

## G022-falcon-v100-c4-performance
- T001 c1/c2/c4 baseline and owner ledger — ACTIVE
  - frontier: Establish actual admitted GPU batch and stable GPU Decode Step/service baseline for S1 and exact G017 Candidate; reuse G020/G021 without rebuilding profiling infrastructure and preserve Prefill/Cold-Warm Cache metrics.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T001-c4-baseline-and-owner-ledger.md`
  - unblock/reopen: Target V100 UUID only; 8-GiB MemAvailable floor; preserve unrelated dirty worktree.
- T002 evidence-driven low-batch optimization — READY
  - frontier: Select first optimization only after T001 produces a bounded c1/c2/c4 GPU owner/time budget and exact hotspot identity.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T002-evidence-driven-low-batch-optimization.md`
  - unblock/reopen: T001 identifies a defensible optimization owner; preserve numerical semantics, S1/Candidate fallback and c1/c2/c4 comparison.

## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
