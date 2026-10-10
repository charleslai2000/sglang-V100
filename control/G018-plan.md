# G018 plan

## Established

- G017 ACCEPTED at `149c70f10b49ff1d1fe28ac5a7016526157a04a9`; S1 remains default, G017 candidate is exact-hash explicit opt-in only.
- Current parent worktree is dirty and pinned on `milestone/s1-falcon-h1` at older `66abb118…`; it is not suitable for G018 changes. New worktree must be based on the accepted G017 commit in the shared Git repo, preserving both existing worktrees.
- Existing Falcon model cache is `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`; pinned revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- `nvidia-smi`, Nsight Compute `/usr/local/cuda/bin/ncu`, Nsight Systems are present; `dcgmi` absent and `nvidia-dcgm` service inactive. Do not start system DCGM or enable global profiling mode without proving non-disruption.
- GV100 NCU metric database lists `sm__cycles_active`, issue/Tensor, DRAM bytes and L2 request bytes. A pinned single-kernel smoke failed with `ERR_NVGPUCTRPERM`; read-only driver state is `RmProfilingAdminOnly:1`. Global permission cannot be altered under G018 scope; no NCU counter values can be obtained. Full service Nsys remains deferred.

## Decisive frontier

G018 worktree and exact runtime pins are established. The decisive frontier is now: uninstrumented matched service matrix, valid in-worker allocator stats, complete graph replay/invocation inventory, and graph-safe owner measurements. Existing FPM is bounded whole-forward event evidence, not owner attribution; 4K decode coverage is incomplete. NCU/DCGM unavailable under recorded constraints; hardware counters UNKNOWN. Do not modify production kernels/defaults or imply a recoverable budget until owner/residual evidence supports one.

## Instrumentation boundary

Prefer external scripts, Python hooks enabled only in G018 process, CUDA events, targeted NCU and nvidia-smi sampling. Do not commit production model/kernel modifications. Any unavoidable temporary source instrumentation belongs only to the G018 worktree, must be compile/runtime-tested and reverted before checkpoint unless it is a reusable non-production tool explicitly approved by scope. No Nsys full-service capture.

## Measurement state and evidence

G018 worktree `/home/charles/Workspaces/3rdparty/sglang-V100-G018`, branch `g018-local-v100-gpu-audit`, is based on accepted G017 `149c70f10b49ff1d1fe28ac5a7016526157a04a9`. Model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; 44 hybrid layers; S1 SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; accepted candidate SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9` and runtime dispatch log verified.

Instrumented FPM+service matrices completed, three repeats per cell, 1K/4K × c1/c2/c4, 256 greedy outputs. Artifacts reside under `/mnt/data/pc01/sglang-v100-migration/experiments/E026/` as recorded in T001. Instrumented service candidate TPS is higher across six cells but is not an uninstrumented baseline. FPM per-forward GPU-event decode means (ms) for c1/c2/c4: S1 28.382/31.453/34.177; candidate 26.097/29.040/31.666. FPM captures 2769 S1 and 3080 candidate decode iterations. Coverage is predominantly ~1K tokens per request; matched 4K c2/c4 FPM rows absent, so no matched long-context inference. FPM event and client service timing remain distinct.

UUID-targeted telemetry captured 600 S1 and 650 candidate samples. The original captures have duplicate timestamp columns; parsed by positional fields. Window means: S1 0% sampled GPU util, 43.00W, 1314MHz SM, 45.12C, 11186MiB device memory; candidate 38.25% GPU util, 118.78W, 1393MHz SM, 53C, 11337MiB. Candidate throttle active reason bit 0x4 appeared in 14/650 samples. These are device-wide sampled fields, not allocator usage/occupancy/bandwidth. The sampler CSV bug has since been fixed and the corrected format passed a 3-row target UUID / 13-column smoke; existing captures were not regenerated.

Ordinary-user NCU smoke denied `ERR_NVGPUCTRPERM`; `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly:1`. G016 archive documents existing root authority; G018 verified `sudo -n` root NCU without changing system policy and collected matched single GPTQ fixture captures for S1/c57. Fixture metrics are not model owner timing. Full-forward NCU remains UNKNOWN/not captured; DCGM unavailable. Detached allocator endpoint zeros invalid. G018 server/helper stopped, target V100 returned to 7MiB/0% after settling. Full-service Nsys remains deferred.

## Current disposition and next action

See `experiments/E027/G018-final-audit.md` and `experiments/E027/artifact-manifest.json`. Instrumented service matrix and whole-forward FPM exist; no uninstrumented throughput authority, no matched 4K c2/c4 FPM, no valid worker allocator/graph-pool measurements, and no complete Graph replay owner time/counts. Two bounded 4K S1 startup retries lost harness control before useful rows. GPU owner/residual and recoverable budget remain UNKNOWN. Evidence-backed disposition is PARTIAL; do not claim ACCEPTED unless remaining direct evidence is obtained. Run validation, commit only G018-scoped assets and sampler correction (no temporary serving source changes), push to a dedicated ref, verify remote SHA and clean worktree.