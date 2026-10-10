# E027 instrumentation and time-accounting design

Status: design reconnaissance only; no profiler or model run executed.

## Four distinct timing boundaries

1. **Client service**: request wall, streamed TTFT, output token timestamps, per-request ITL, aggregate and request TPS; includes prefill, scheduler/queue and decode.
2. **Scheduler/host**: use server logs/FPM request scheduling fields and Python-level monotonic spans around explicitly instrumented phases; do not infer from service minus GPU time as exact host CPU.
3. **Whole-model forward GPU**: existing `DeviceTimer` uses CUDA start/end events around `ModelRunner.forward_decode` (includes graph replay when CUDA Graph used); FPM reports the accumulated measured device times alongside scheduled request/KV-length statistics. It is intrusive due per-forward events and its association is iteration summary rather than request-ID exact.
4. **Individual op/kernel**: targeted wrapper-level CUDA event scopes/NVTX and NCU kernel-only duration/counters. Neither alone adds up automatically to full-forward time; include omitted activation/normalization/attention/Mamba launches and overlaps. Never equate operator wall with kernel duration.

## Model/topology inventory from pinned model source

Falcon H1 layers run self-attention and Mamba2 for non-idle forward, then dense GPTQ MLP. Query Mamba2 config and exact `layers_block_type` from pinned checkpoint. Instrument each layer owner around the existing `FalconH1HybridAttentionDecoderLayer` method components using a local G018-only source patch or hooks, and count only dispatches from a warmed graph replay; be aware ordinary Python hooks do not execute during CUDA graph replay. For graph mode, record aggregate CUDA events at carefully chosen model regions only if capture-safe and verify graph capture/replay. Safer initial call inventory: model config + offline module parameters and targeted NVTX profiling replay, not Python monkeypatch hooks inside captured regions.

## Instrumentation design (approved scope)

- Exact paired server mode: frozen S1 path versus candidate opt-in; same model revision/config/backend/graph sizes. Separate process per mode and workload. Warm up then issue isolated request(s); record prompt 1K/4K, output 256, c1/c2/c4, requests IDs, streamed token times.
- FPM/DeviceTimer baseline: emits per iteration `wall_time` from CUDA event accumulation and scheduled prefill/decode counts/KV lengths. Use bounded replications. This yields total CUDA-event forward/replay duration but does not include events for other streams unless ordering joins them. Gather profiler logs only while this G018 process owns V100.
- GPU telemetry: sample `nvidia-smi --query-gpu` at 1 Hz for target UUID only: util, util memory, pstate, SM/mem clocks, power, temperature, memory allocated. Capture before/after plus during bounded steady workload. `memory.used` is device-global, allocator allocated/reserved/peak from process can come from PyTorch metrics exported to a local diagnostic endpoint/hook. Throttle counters supported by this driver include GPU idle, app clock, software power cap, hardware slowdown/thermal/power brake; record supported/active and retain the full hex mask.
- NCU: initially run a no-kernel smoke using `ncu --target-processes all --devices <V100-index> --set ...`; do not set `--nvprof`/global profiler mode. Check target UUID maps to index only after confirming both GPUs; use CUDA_VISIBLE_DEVICES UUID. No host configuration changes. For selected kernels collect same set: `sm__cycles_active`, `sm__issue_active`, `sm__pipe_tensor_cycles_active`, `sm__inst_executed_pipe_tensor`, `sm__ops_path_tensor_src_fp16_dst_fp16`, `dram__cycles_active`, `dram__bytes_read/write`, `lts__t_bytes`, registers/thread, achieved occupancy and sampled stall reasons as supported by `ncu --query-metrics --chip GV100`. Separate incompatible sets into matched repeated run windows. Profile targeted Python fixture/operator runs only; no full service NCU if overhead/counter replay alters workload. State NCU duration/clock perturbation.
- DCGM: `dcgmi` absent; system service inactive. Do not start daemon or change global profiling state. `nvidia-smi` is the supported alternative for non-counter telemetry; NCU targeted profiling for kernel counters, subject to permission verification.

## Reconciliation

For each phase/shape/concurrency, retain raw request events, scheduler FPM rows, CUDA forward event rows, GPU telemetry samples and exact layer counts. For c1/c4, sum whole-forward CUDA-event durations over matched decode iterations. Decompose separately measured/captured owner ranges only when event boundaries cover them; report overlaps and unaccounted event time. The identity service wall = prefill/TTFT + decode + queue/host is not exact for overlapping c>1; avoid additive identities unless per-request spans have disjoint intervals. Publish an accounted sum plus UNKNOWN residual and a strict bound (0 <= missing device component <= whole model forward, where applicable), never force-fit to 100%.

## Candidate vs S1

S1 SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; accepted candidate SHA from G017 `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`. For existing paired service evidence/forward measurements preserve their provenance; do not relabel measurements as G018 current-run capture. If no new kernel-level insight is needed, stop at equal-duration matched measurements and owner recommendation rather than exhaustively ncu-profile every op.
