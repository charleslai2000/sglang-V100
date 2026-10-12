# Current frontier

## G021-v100-hardware-utilization
- T001 R17 Inference Performance Metrics V1 — ACTIVE
  - frontier: FPM PASS; standalone speed diagnostic plus root NCU Prefill EXTEND/Decode Graph-replay KERNEL_SCOPE counters are in report/matrix/manifest. Full Graph aggregate is not measured; continue queue-tail/lifecycle/cache/saturation/P99 and owner/3090 dispositions.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: Prior resource gate/resume authorization remains; maintain MemAvailable >=8GiB and target-only V100 ownership. Preserve G017/S1 default; do not touch RTX2060, global profiling policy, or unrelated tasks.

## G018-local-v100-falcon-h1-gpu-audit
- T001 Profile ledger and resource audit — ACTIVE
  - frontier: run separate service, FPM/device-event, and UUID-targeted nvidia-smi windows; establish graph-safe complete c1/c4 owner/count ledger and c1/c2/c4 × 1K/4K prefill/decode matrix; mark NCU hardware counters UNKNOWN due denied permission.
  - task: `control/G018-local-v100-falcon-h1-gpu-audit/tasks/T001-profile-ledger-and-resource-audit.md`
  - unblock/reopen: Clean worktree at G017 accepted commit; exact S1/c57 pins and 44-layer topology confirmed. DCGM unavailable; NCU smoke denied `ERR_NVGPUCTRPERM` with `RmProfilingAdminOnly:1`, do not modify global policy. Use nvidia-smi/FPM/CUDA events; no RTX2060 target, no production kernel change, full-service Nsys remains DEFERRED.
