# G018 plan

## Established

- G017 ACCEPTED at `149c70f10b49ff1d1fe28ac5a7016526157a04a9`; S1 remains default, G017 candidate is exact-hash explicit opt-in only.
- Current parent worktree is dirty and pinned on `milestone/s1-falcon-h1` at older `66abb118…`; it is not suitable for G018 changes. New worktree must be based on the accepted G017 commit in the shared Git repo, preserving both existing worktrees.
- Existing Falcon model cache is `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`; pinned revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- `nvidia-smi`, Nsight Compute `/usr/local/cuda/bin/ncu`, Nsight Systems are present; `dcgmi` absent and `nvidia-dcgm` service inactive. Do not start system DCGM or enable global profiling mode without proving non-disruption.
- GV100 NCU metric database lists `sm__cycles_active`, issue/Tensor, DRAM bytes and L2 request bytes. A pinned single-kernel smoke failed with `ERR_NVGPUCTRPERM`; read-only driver state is `RmProfilingAdminOnly:1`. Global permission cannot be altered under G018 scope; no NCU counter values can be obtained. Full service Nsys remains deferred.

## Decisive frontier

Create a G018 clean worktree/branch at G017 accepted commit, capture exact base and runtime IDs, then qualify a low-impact supported metric collection plan. Establish c1/c4 complete decode ledger with actual whole-model dispatch/call counts and matched CUDA event/Graph timing. Extend to 1K/4K c1/c2/c4 service matrix and matched targeted NCU only where justified. Report unsupported telemetry as UNKNOWN.

## Instrumentation boundary

Prefer external scripts, Python hooks enabled only in G018 process, CUDA events, targeted NCU and nvidia-smi sampling. Do not commit production model/kernel modifications. Any unavoidable temporary source instrumentation belongs only to the G018 worktree, must be compile/runtime-tested and reverted before checkpoint unless it is a reusable non-production tool explicitly approved by scope. No Nsys full-service capture.

## Result

G018 control Goal/T001 created. Clean worktree `/home/charles/Workspaces/3rdparty/sglang-V100-G018` is at accepted base `149c70f10b49ff1d1fe28ac5a7016526157a04a9`, branch `g018-local-v100-gpu-audit`. No service workload has run yet. Static topology/DeviceTimer/FPM analysis is in `experiments/E027-profile-design.md`; permission/tool evidence in `experiments/E027/resource-preflight.json`. Clean package origin and frozen S1 SHA `fea05241…` verified; checkpoint has 44 hybrid layers. Initial V100 idle sample 7 MiB, 0%, 135/877 MHz, 25.15 W/41C; throttle GPU-idle only. NCU permission denied (`ERR_NVGPUCTRPERM`, RmProfilingAdminOnly=1), DCGM absent/inactive. No global policy change; NCU counters will be UNKNOWN. Next run controlled client/FPM/telemetry observations and G018-only graph-safe owner attribution.