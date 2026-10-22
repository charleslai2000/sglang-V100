# Current frontier

## G022-falcon-v100-c4-performance
- T011 — QPN2 multiwarp group-aligned execution — ACTIVE (Decode evidence preserved; production qualification incomplete)
  - frontier: T011's independent QPN2 Decode work remains documented but not production ACCEPTED. Its downstream model-state, 16-GiB capacity and service gates were not completed because the prerequisite T012 Prefill route closed PERFORMANCE-NO-GO / NUMERICAL-NOT-FULLY-QUALIFIED; joint service matrix is NOT_RUN. T012's ~13–15% 512 and ~38% 1K/4K regression is route-specific, not a verdict on compressed-Q4 Tensor Core methods as a class.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T011-qpn2-multiwarp-group-reduction.md`
  - unblock/reopen: T012 is DONE and no Falcon optimization continues in this closure. Resume only under future explicit authorization. Preserve E035/unowned changes, S1/default/exact-c57; G023 ON_HOLD.

## G023-falcon-gpu-l1-prefix-cache
- T001 — qualify Falcon GPU prefix reuse — DEFERRED
  - frontier: Actual `extra_buffer` radix hit reports 768 tokens; same-input cache hit differs from same-config force-miss at output token 0/1. One chunk=768 diagnostic did not fix it; changing no-cache chunk size also changes one branch at token24. Locate state boundary before any performance matrix.
  - task: `control/G023-falcon-gpu-l1-prefix-cache/tasks/T001-qualify-falcon-gpu-prefix-reuse.md`
  - unblock/reopen: Owner-directed ON_HOLD; resume only on explicit owner authorization. Preserve raw/dirty file; target V100 and >=8GiB. E035 stays closed.

## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — READY
  - frontier: Keep the frozen V1 contract open: cache restore, safe higher concurrency/P99 dispositions, remaining owner/3090 comparability and full-Graph aggregate remain unresolved. Do not re-open bounded FPM qualification.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.
