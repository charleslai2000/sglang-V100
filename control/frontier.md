# Current frontier

## G021-v100-hardware-utilization
- T001 Hardware utilization — BLOCKED
  - frontier: hold all online GPU work; wait for unrelated PID 3903363 to exit naturally or separately authorized capacity window; require 30-min RAM/PSI/swap/disk gate before exact-SHA S1 Falcon + 1K/4K × c1/c4 NVML and separate bounded NCU Graph runs.
  - task: `control/G021-v100-hardware-utilization/tasks/T001-hardware-utilization.md`
  - unblock/reopen: all auditable resume conditions in `experiments/G021/preflight.md` pass; never kill/throttle unrelated job or alter host policy. R17 TTFT/TPOT/ITL/cache/E2E follows hardware matrix.

## G018-local-v100-falcon-h1-gpu-audit
- T001 Profile ledger and resource audit — ACTIVE
  - frontier: run separate service, FPM/device-event, and UUID-targeted nvidia-smi windows; establish graph-safe complete c1/c4 owner/count ledger and c1/c2/c4 × 1K/4K prefill/decode matrix; mark NCU hardware counters UNKNOWN due denied permission.
  - task: `control/G018-local-v100-falcon-h1-gpu-audit/tasks/T001-profile-ledger-and-resource-audit.md`
  - unblock/reopen: Clean worktree at G017 accepted commit; exact S1/c57 pins and 44-layer topology confirmed. DCGM unavailable; NCU smoke denied `ERR_NVGPUCTRPERM` with `RmProfilingAdminOnly:1`, do not modify global policy. Use nvidia-smi/FPM/CUDA events; no RTX2060 target, no production kernel change, full-service Nsys remains DEFERRED.

## G020-existing-trace-offline
- No active task — DONE PARTIAL
  - result: Offline-only G018/G019 trace analysis is in `experiments/G020/report.md`; artifact and analyzer are in the same directory. No new GPU profiling.
  - control: `control/G020-existing-trace-offline/plan.md`
