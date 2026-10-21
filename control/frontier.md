# Current frontier

## G022-falcon-v100-c4-performance
- T011 — QPN2 multiwarp group-aligned execution — ACTIVE
  - frontier: Optional SM100 MXINT4 import fixed; new SHA `6f3ee573…` loads 88/88 runtime GPTQ linears, captures Graph1/2/3/4 and runs one 512-token Prefill→Decode path. Frozen FP32/logit/greedy oracle, concurrent state/slot lifecycle, NVML peak/capacity and six-point fresh c57/T011 service A/B remain decisive gates; no READY/NO-GO.
  - task: `control/G022-falcon-v100-c4-performance/tasks/T011-qpn2-multiwarp-group-reduction.md`
  - unblock/reopen: Operator gate and bounded NCU passed. Continue owner authorization, target V100 only, MemAvailable≥8GiB, protect E035/unowned changes, S1/default/exact-c57 unchanged; G023 ON_HOLD.

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
