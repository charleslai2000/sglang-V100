# Current frontier

## G021-v100-hardware-utilization
- T001 Hardware utilization — ACTIVE
  - frontier: S1/Candidate dispatch-verified service matrices, target NVML windows and real Falcon GPTQ M4 NCU fixture measurements are captured; close remaining live Graph/owner attribution, FPM zero-record diagnosis and R17 queue/cache/E2E gaps with explicit unsupported dispositions where needed.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: resource release verified; maintain host MemAvailable >=8GiB and target UUID ownership. Do not alter host policy, G017, or unrelated work.

## G018-local-v100-falcon-h1-gpu-audit
- T001 Profile ledger and resource audit — ACTIVE
  - frontier: run separate service, FPM/device-event, and UUID-targeted nvidia-smi windows; establish graph-safe complete c1/c4 owner/count ledger and c1/c2/c4 × 1K/4K prefill/decode matrix; mark NCU hardware counters UNKNOWN due denied permission.
  - task: `control/G018-local-v100-falcon-h1-gpu-audit/tasks/T001-profile-ledger-and-resource-audit.md`
  - unblock/reopen: Clean worktree at G017 accepted commit; exact S1/c57 pins and 44-layer topology confirmed. DCGM unavailable; NCU smoke denied `ERR_NVGPUCTRPERM` with `RmProfilingAdminOnly:1`, do not modify global policy. Use nvidia-smi/FPM/CUDA events; no RTX2060 target, no production kernel change, full-service Nsys remains DEFERRED.
