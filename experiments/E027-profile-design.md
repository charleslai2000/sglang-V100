# E027 — G018 profiling design and preflight

Status: initial bounded instrumented FPM/service and target-UUID telemetry windows completed; see control/G018-task-T001.md. Owner timing, valid allocator accounting and uninstrumented service qualification remain open.

## Authority and runtime pins

- Clean G018 worktree base `149c70f10b49ff1d1fe28ac5a7016526157a04a9`; branch `g018-local-v100-gpu-audit`.
- Falcon-H1 model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; local checkpoint `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`.
- S1 exact module SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`, located in `/mnt/data/pc01/sglang-v100-migration/experiments/E026/frozen-s1-runtime/sgl_kernel/sm70/common_ops.abi3.so`.
- Accepted G017 candidate SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`. Important: G017's earlier service A/B/token identity/full state-isolation runs used the pre-final-loader binary `02f923ea…` / final c57 service wrapper qualification inherited from exact c57 loader-hardened build; source's CUDA op implementation unchanged. G018 must use accepted c57 for all new candidate observations, not old candidate runtime SHA.
- Force only `CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; record `nvidia-smi -L` mapping at each process. No target-index ambiguity. RTX2060 UUID must never enter NCU target set.

## NCU permission and bounded capture disposition

The initial ordinary-user NCU smoke failed with `ERR_NVGPUCTRPERM`; read-only driver parameter is `RmProfilingAdminOnly:1`. G016 records document already-authorized targeted root NCU on this V100, with archived report `/tmp/g016-root-ncu.ncu-rep` (SHA `aed3156bdf2e237279711bc10bcb239ece64c5a1b980760398fd6f9233ada835`). G018 verified `sudo -n` root NCU access and UUID mapping (V100 is index 1; RTX2060 index 0). No permission/driver state was changed. A matched one-launch S1/c57 NCU capture was performed on an isolated synthetic SGLang GPTQ M=1,K=N=4096 operator fixture; report SHAs and metric values are in `experiments/E027/G018-final-audit.md` and `hardware-evidence.json`. These are fixture-only results, not Falcon-serving owner values. NCU full-model/Graph/Attention/Mamba capture was not performed; mark those metrics UNKNOWN, not as globally inaccessible to authorized root.

## DCGM and resource telemetry

- `dcgmi` absent, `nvidia-dcgm` inactive; do not start/change a system service.
- `nvidia-smi` target UUID fields available for 1 Hz capture: utilization GPU and memory, SM/memory clocks, P-state, power draw, temperature, memory total/used, supported/active throttle reasons including idle, software power cap, hardware slowdown/thermal and power-brake reasons. Device-global `memory.used` is not process allocator VRAM. For allocator allocated/reserved/peak, read PyTorch allocator stats from the SGLang worker, tagged with capture window and process.
- Idle baseline at 2026-10-10 20:56: V100 7 MiB/0%, P0, SM 135 MHz/memory 877 MHz, 25.61 W, 42C. Throttle reason mask 0x1ff supported, active reason GPU idle only; SW power cap, HW slowdown, thermal, power brake and SW thermal inactive. See `resource-preflight.json`.
- Current `nvidia-smi` shows RTX2060 busy around 2GB/low utilization. That is unrelated; do not target or profile it.

## Workload and timing layers

Use separate, reproducible runs with mode pin (S1 or c57 candidate), same serving args and local model/config:
- 1K and 4K prompts × concurrency 1/2/4; 256 output tokens, greedy temperature 0 and ignore_eos true. At least 3 measured waves/cell; cold/warmup excluded and recorded. Keep same prompt token IDs across modes. Save stream event times and output IDs.
- Client: request wall, TTFT, per-token arrival-derived ITL, per-request TPS, aggregate decoded TPS, full service wall TPS. At c>1 these are not additive because requests overlap.
- Server GPU forward: optionally run separate bounded FPM run with `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 --enable-forward-pass-metrics`, capture PUB before requests, and retain per-iteration counter id, forward duration, scheduled prefill/decode counts and KV lengths. FPM time is CUDA event duration around model forward/Graph replay; not request-ID linked, and this run is instrumented so do not use its service TPS as uninstrumented A/B.
- Host/scheduler: startup, tokenizer client timestamps and logs; FPM scheduled/queued counts and KV lengths; avoid calling service wall minus device timer exact CPU time.
- 1 Hz nvidia-smi sampler over a precisely logged wall-time window (before server, load, graph capture, warmup, steady measured waves, shutdown); use `--id=GPU-UUID` rather than default both-GPU query. Retain raw CSV and mark monitor cadence.

## Model owner inventory

Static model source `python/sglang/srt/models/falcon_h1.py`: every non-idle FalconH1 decoder layer executes self-attention (QKV -> RoPE/attention -> O), Mamba2 linear-attention backend (recurrent + convolution), then MLP gate_up -> SiluAndMul -> down; associated layer norms/communication. GPTQ applies to MLP gate_up/down, attention projection `quant_config=None`. Exact 32-layer/block sequence, layer dimensions, per-layer kernel/event count, and actual graph-replay path must be checked against checkpoint config and runtime graph capture. Do not infer all owner calls by multiplying a fixture timing unless exact shapes and repeated per-layer execution are verified.

## NCU disposition

- `ncu --query-metrics --chip GV100` succeeded and exposes useful metric definitions: `sm__cycles_active`, `sm__issue_active`, `sm__pipe_tensor_cycles_active`, `sm__inst_executed_pipe_tensor`, `sm__ops_path_tensor_src_fp16_dst_fp16`, `dram__cycles_active`, `dram__bytes_read/write`, `lts__t_bytes`, `smsp__warps_active/warps_eligible`, and stall counters. Query confirms theoretical support only, not measurement authorization.
- Due `ERR_NVGPUCTRPERM` + `RmProfilingAdminOnly:1`, no NCU command on a model/server may be run. No host-wide permission change, driver reload or profiling-mode change. Use supported alternatives: server CUDA events/FPM and nvidia-smi telemetry. The request explicitly allows documented unavailable metrics and equivalent observations; issue exact blocker evidence in final ledger.
- Nsight Systems full service run remains DEFERRED. No profiler expansion.

## Ledger accounting

For each workload window, keep separate columns: (A) service wall/TTFT/ITL, (B) FPM CUDA-event whole forward total by scheduled decode KV length, (C) owner measurements and invocation counts where graph-safe, (D) GPU sample telemetry. Compute accountable owner union and residual only if event scopes are non-overlapping and exact calls observed. If only part of model is owner-instrumented, record accounted portion and UNKNOWN remainder; no forced normalization to 100%. GPU utilization sampled by nvidia-smi is a 1-s duty-cycle average, not SM active or occupancy. Power/clocks are system-wide instantaneous samples. Memory util isn't HBM bandwidth. No resource limit claim from one aggregate counter alone.

## Initial measurements and limitations

Bounded instrumented 1K/4K × c1/c2/c4 matrices were run for S1 and accepted c57, 3 repeats/cell, 256 greedy outputs. FPM and client service artifacts are under `/mnt/data/pc01/sglang-v100-migration/experiments/E026/`; exact names and aggregate results are in `control/G018-task-T001.md`. These are instrumentation-window service results, not the required uninstrumented A/B. FPM has 2769 S1 and 3080 candidate decode iterations. Per-forward decode event means by active batch size c1/c2/c4 are S1 28.382/31.453/34.177 ms and candidate 26.097/29.040/31.666 ms. KV-length coverage is mostly 1K/request. Matched 4K/request c2/c4 rows are absent; 4K owner conclusions remain UNKNOWN.

Target UUID nvidia-smi telemetry ran 600 seconds S1 and 650 seconds candidate. Existing CSV rows have an extra timestamp field because both script and nvidia-smi timestamps are present; they were parsed by position, not treated as clean header-aligned CSV. The sampler has been corrected to align 13 columns and passed a 3-row smoke test; original captures remain unchanged. Detached allocator helper snapshots were zero and invalid because they did not run inside the model worker. No process-local allocator conclusion. The G018 server/helper were subsequently stopped; no owned processes remain and the V100 returned to 7 MiB / 0% after settling.

NCU permission smoke failed `ERR_NVGPUCTRPERM`, with `RmProfilingAdminOnly:1`; DCGM tools/service are unavailable. Do not change system permission/service. Hardware counters, process allocator, owner breakdown and residual remain UNKNOWN. Full-service Nsight Systems is still deferred. Uninstrumented service measurements and graph-safe owner attribution remain to be done.

## Artifacts

Retain raw service events/output IDs, server logs, FPM rows, GPU telemetry CSV, config/tokenizer hashes, c57/S1 SHA, launcher command, tool versions, any owner instrumentation hash, model config-derived layer inventory, analysis script/version and result tables. Any injected temporary instrumentation resides only in G018 worktree and must be reverted before final production-neutral commit. No kernel/production code changes.
