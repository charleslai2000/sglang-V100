# Plan — G016 Local V100-16GB Falcon-H1

## Established facts

- Local device: Tesla V100-SXM2-16GB, driver 580.178.04; another desktop GPU (RTX 2060) is active and must not be used for experiments.
- Isolated runtime `/home/charles/miniconda3/envs/sglang-v100`: Torch 2.9.1+cu128, runtime CUDA 12.8, Triton 3.5.1, Transformers 5.8.1; V100 reports SM70. Installer and SM70 kernel smoke completed successfully; logs at `/mnt/data/pc01/sglang-v100-migration/logs/final-install-success.log` and `smoke-retry3.log`.
- Full pinned Falcon checkpoint is local at `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`; both shard SHA-256s match pinned LFS OIDs. `/mnt/data` now has about 11GB free.
- Runtime import needs repo source on `PYTHONPATH`; expose local V100 as CUDA logical 0 with `CUDA_VISIBLE_DEVICES=0` (host nvidia-smi physical index differs).
- Local OS Ubuntu 24.04.5, Python 3.12.3, CUDA Toolkit 12.9.86.
- The repository at migration start is commit `2eeead7e0ce274e8162a2c4f4e1b64d39c6a1b21`; remote source workspace was `dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3` with local modifications now synchronized by path. These are not yet reconciled/audited as one source state.
- Default local Python has no torch, SGLang, Triton, FlashInfer, Transformers or Safetensors installed.
- `/mnt/data` is nearly full (about 11GB free at last check); root is intentionally not writable. No unrelated data was removed.
- The remote S1 reference environment was V100-SXM2-32GB; those results remain context only and do not qualify local 16GB capacity.
- Remote source tar, untracked files, working patch, logs and traces are preserved at `/mnt/data/pc01/sglang-v100-migration`.

## Decisive frontier

Per the revised acceptance, G016 does not wait on Falcon server Nsight Systems tracing; record it as `DEFERRED / NOT QUALIFIED`. Offline root cause and 4GB staging proof remain in `Nsys-recovery.md`. Remaining work is final context/headroom verification and bounded short soak; then freeze READY and hand off S4-M3C. Targeted NCU remains the profiling authority.

## Local bring-up completed

- Full pinned Falcon GPTQ checkpoint is present and both shards verify against the pinned LFS SHA-256 OIDs.
- Isolated Python 3.12 runtime at `/home/charles/miniconda3/envs/sglang-v100`: Torch 2.9.1+cu128, runtime CUDA 12.8, Triton 3.5.1, Transformers 5.8.1. V100 reports SM70.
- Built/installed the lean SM70 SGLang kernel, FlashInfer SM70 source, TileLang attention route, Marlin dense/MoE, and TurboMind extension. `scripts/install_v100.sh` completed; `scripts/smoke_v100.sh` passed and registered the expected SM70/GPTQ/TurboMind/NCCL paths.
- Installer sparse path handling and shell continuation issues were corrected. Editable namespace imports require `PYTHONPATH=/home/charles/Workspaces/3rdparty/sglang-V100/python`; serving must expose local V100 as CUDA logical device 0 (`CUDA_VISIBLE_DEVICES=0`, not the host's nvidia-smi index).
- Existing root NCU minimal-kernel proof and ordinary-user Nsight Systems CUDA trace are already successful; driver/module remains unchanged, no reboot.
- Evidence/logs: `/mnt/data/pc01/sglang-v100-migration/logs/final-install-success.log`, `smoke-retry3.log`; model under `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`.
- Exact source HEAD at install: `2eeead7e0ce274e8162a2c4f4e1b64d39c6a1b21`; migrated-source classification remains incomplete.

## Profiling environment progress

- Installed tools are Nsight Systems 2025.1.3.140 and Nsight Compute 2025.2.1.0. `nsys` is present; `dcgmi` and CUPTI libraries were not found in the current PATH/loader cache/CUDA tree search.
- The driver currently reports `RmProfilingAdminOnly: 1`; ordinary-user `ncu` reports `ERR_NVGPUCTRPERM` for V100. Do not reload the module or reboot. Use authorized root only for narrowly targeted NCU; serving remains ordinary-user on V100.
- Root NCU has profiled a minimal kernel on the local V100, demonstrating readable SM, tensor, DRAM/L2, occupancy and warp-stall metrics. Do not interpret this as model performance.
- `NVreg_RestrictProfilingToAdminUsers=0` exists in `/etc/modprobe.d/nvidia-profiling.conf` and current initramfs is regenerated; leave the running module state unchanged. This persistent option is for a future convenient reboot only, not a bring-up gate.
- Nsight Systems reports CPU profiling unavailable because kernel `perf_event_paranoid=4`; GPU CUDA tracing remains available. CPU sampling is not required by user policy; do not lower global sysctl without need.
- `nvidia-smi` supports GPU and memory utilization telemetry, memory usage, clocks, power, temperature. DCGM/CUPTI enhanced tensor/DRAM telemetry not yet established.
- Pip HTTP cache was reclaimed; no unrelated user data/model/workspace was deleted.
- Existing shared `/mnt/data/cache/venv` is owned by `dev-codex`; it remains unsuitable (permission restriction and no V100 sm70 kernels). It was not modified; use the isolated local environment above.
- Root NCU minimal kernel passed on the local V100; report `/tmp/g016-root-ncu.ncu-rep`. It demonstrated readable compute(SM), tensor, DRAM/L2, occupancy/resource and warp-stall metric families. Do not interpret this probe as model performance. `RmProfilingAdminOnly: 1` remains unchanged. A harmless local CUDA probe binary/source is in `/tmp/g016-smoke*`.
- Ordinary-user Nsight Systems CUDA trace passed on the same V100 using `CUDA_VISIBLE_DEVICES` UUID pinning; `/tmp/g016-nsys-smoke.nsys-rep` and stats verify one kernel plus CUDA API trace. No NCU whole-serving profile performed.
- Current local server at `127.0.0.1:30003` reports `model_type=falcon_h1`, architecture `FalconH1ForCausalLM`. It loaded 8.09GB GPTQ weights in 7.64s; initialized 16,384-token KV pool (0.68GB), 0.64GB Mamba state, and captured CUDA graphs for batch sizes 1/2/4 (reported +0.20GB, 5.01GB available after init). Current nvidia-smi V100 allocation is approximately 11.2GB. Correctness smoke passed c1/c2/c4 at 1,024 and 4,096 prompt tokens (16 nonzero output tokens/request, deterministic IDs across same-size requests), plus repeat greedy request at 32 prompt tokens; record `/mnt/data/pc01/sglang-v100-migration/logs/s1-correctness-smoke.json`. This is partial correctness, not complete frozen S1 graph-replay/nonfinite/state-contamination qualification.

## Active Task

- `tasks/T001-restore-environment-and-profiling.md` owns restore, qualification, supported-tool/counter discovery and transition to measurement readiness.

## Latest qualification update

- Final S1 matrix at `/mnt/data/pc01/sglang-v100-migration/logs/g016-local-final-qualification.json`: interleaved math/capital/math/prime/capital/math requests matched each prompt's output; graph c2/c4 concurrent distinct prompts stayed distinct; c1/c2/c4 at 1K/4K and c1 at 7,680 passed finite actual generated-token checks. Earlier replay hook confirms keys 1/2/4 (137/16/16 counts). A V100-local `torch.isfinite` check classified finite/NaN/±Inf correctly.
- Root NCU report identifies production GPTQ `gemm_half_q_half_alt_4bit_kernel`, Device 0 CC7.0, with SM/DRAM/L2/resource/warp-stall metrics. No model-performance claim.
- Offline Nsys root-cause evidence is in `Nsys-recovery.md`: server process tree had scheduler and separate CUDA worker; CUDA injection initialized in neither worker report event stream, while OSRT injection did. No CUDA events were recorded; worker session storage grew to 22.3GB over an unbounded ~51-minute collection. No c4 requests occurred during that session. Exact low-level injection cause remains unknown.
- Local CLI help confirms graph-level host-only capture and sampling-off options. `TMPDIR` does not constrain Nsys session staging. A temporary 4GB loop filesystem bind-mounted over the actual `/tmp/nvidia/nsight_systems` path successfully bounded a minimal V100 probe; valid 155,677-byte report with one CUDA kernel and ~36KB staging. Temporary test files and mounts removed. Falcon worker injection not proven by this smoke.
- Source installer changes reviewed and committed as `36b35cb68`; graph hook removed. Runtime/model/dependency hashes recorded in T001/Nsys-recovery. Bounded stability evidence `/mnt/data/pc01/sglang-v100-migration/logs/g016-bounded-soak.json`: 30.3s, c4, 48 waves/192 requests, zero errors, identical greedy outputs; latency median 0.621s, p95 0.649s, max 0.814s. This short smoke is not a performance benchmark or long soak. After stop, V100 16,138 MiB free of 16,384 MiB. During service nvidia-smi used ~11.2GiB/free ~4.55GiB; server reported ~4.98–5.01GB free after init/capture. Allocator snapshot ~10.14GB active, 10.83GB segments, graph pool ~0.153GB reservation. Not a full OOM/long-soak matrix.

## Result / handoff

`G016 LOCAL FALCON + NCU AUTHORITY READY` under the revised acceptance. Serving correctness, interleaved isolation, graph 1/2/4, local capacity/headroom smoke and bounded short stability passed; targeted root NCU is valid and attributable to the production GPTQ kernel. Final code commit is `36b35cb68fe493c202f7d6c64100de29738b2684`; model/runtime/dependency identities are recorded in T001 and Nsys-recovery. Explicit limitation: `NSYS FALCON SERVER CUDA TRACE — DEFERRED / NOT QUALIFIED`; do not treat as PASS. No performance conclusion is made under G016.

Next authorized action: resume S4-M3C small-M GPTQ kernel optimization using existing c4 E023 ledger/E024 rejected shuffled-route evidence, with targeted NCU as needed. Do not repeat the rejected shuffled donor; new donor work follows its own authorization.
