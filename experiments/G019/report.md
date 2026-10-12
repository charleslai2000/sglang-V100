# G019 CUDA Graph decode execution attribution — bounded results

Disposition: **PARTIAL**. Worker-side SGLang profiling yields real Falcon-H1 kernel activity for S1 and exact c57 at 1K/4K × c1/c4. However, its exported kernel events omit CUDA Graph `graphId` and `graphNodeId`. The official CUDA 12.8 CUPTI Graph sample demonstrates those IDs in a standalone CUDA graph, but cannot be attached to the SGLang worker with existing endpoints. No third profiler framework is authorized. Thus the requested node-identified Graph ledger is not available in this phase; do not label time or owner shares as complete.

## Frozen identity and capture conditions

- G019 worktree branch `g019-cuda-graph-attribution`, parent G018 checkpoint `df975fc3a209bca90536d19861c3ee7404eca45d`.
- Model: Falcon-H1-7B GPTQ local revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, visible as CUDA device 0 after UUID pin. RTX 2060 not profiled.
- S1 common_ops SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; c57 SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`. Candidate log confirms exact path/SHA and actual GPTQ dispatch at graph capture.
- PyTorch `2.9.1+cu128`, CUDA runtime 12.8; no upgrades/config changes. Capture used existing `/start_profile`, `profile_by_stage=true`, decode-only, `num_steps=3`, CPU+GPU activities, `with_stack=false`, shapes enabled. Each request wave was greedy 256 output tokens; one 1K or 4K prompt, concurrency 1 or 4. Server logs in `/mnt/data/pc01/sglang-v100-migration/logs/g019-server-s1.log` and `g019-server-c57.log`.
- Eight primary decode traces under `/mnt/data/pc01/sglang-v100-migration/experiments/G019/`; c57/4K/c4 also repeated once. Hashes and sizes in `artifact-manifest.json`. Matching EXTEND traces are adjacent; this report excludes them from decode analysis.

## Route A: fork worker-side profiler inspection and capture

Source audit confirmed the fork has `POST /start_profile` and `/stop_profile` (`entrypoints/http_server.py`), `ProfileReqInput.num_steps`, `profile_by_stage`, `profile_stages` (`managers/io_struct.py`), and `SchedulerProfilerManager` starts/stops `torch.profiler.profile(activities=[CPU, CUDA])` based on actual forward-batch stage/count. For bounded captures, server `SchedulerProfilerManager` did start, auto-stop, and export distinct `-DECODE.trace.json.gz` and `-EXTEND.trace.json.gz` files. CUDA Graph replay was actually exercised: server logs report decode `cuda graph: True`, while capture startup records graph batch sizes `[1,2,4]`.

Each decode file contains 4,438–4,993 CUDA kernel activity events, 9,926–11,254 total trace events, plus CUDA runtime/driver events and some GPU memcpy records. Kernel event fields include kernel name, GPU timestamps/duration, device/context/stream, launch grid/block, registers/thread and correlation. Crucially, kernel event args do **not** include `graphId` or `graphNodeId`; automated audit over every kernel event in all nine traces found zero node-ID fields. Most graph-launched kernels also have no `External id`; only 50–76 out of ~4.4–5.0k events per trace have it. Profiler's exported `ac2g` records are CUDA API-to-GPU correlation bookkeeping and do not identify the CUDA Graph node. Consequently no exact node ID, graph object ID or per-node Graph replay count can be assigned from Route A.

## Trace timing and overlap accounting

The worker trace contains kernel durations from CUPTI. The following are trace-window aggregates over all kernel records captured during the profiler's decode-stage window; `num_steps=3` bounds by forward count, but the actual exported window may include adjacent scheduler activity and decode work for all requests in a concurrent wave. `GPU kernel duration sum` is the sum of concurrent activity durations and **must not be treated as wall time**. `kernel interval union` subtracts temporal overlap among kernel records; `kernel span` is max end - min start among the captured kernel records. These are profiler-window aggregates, not a single-forward CUDA-event measurement; there is no request-ID/forward-to-kernel graph-node mapping. Values are only useful for timeline coverage diagnostics.

| Mode / prompt / c | Kernel records | duration sum (ms; overlap-counting) | kernel interval union (ms) | kernel span (ms) | union/span |
|---|---:|---:|---:|---:|---:|
| S1 / 1K / 1 | 4,447 | 88.983 | 88.906 | 99.039 | 89.77% |
| S1 / 1K / 4 | 4,964 | 107.941 | 107.933 | 111.262 | 97.01% |
| S1 / 4K / 1 | 4,438 | 90.463 | 90.422 | 93.865 | 96.33% |
| S1 / 4K / 4 | 4,980 | 113.453 | 113.371 | 121.403 | 93.38% |
| c57 / 1K / 1 | 4,447 | 81.498 | 81.453 | 86.289 | 94.40% |
| c57 / 1K / 4 | 4,972 | 99.006 | 98.994 | 103.177 | 95.95% |
| c57 / 4K / 1 | 4,445 | 82.719 | 82.701 | 86.467 | 95.64% |
| c57 / 4K / 4 | 4,988 | 103.705 | 103.690 | 109.910 | 94.34% |

Candidate 4K/c4 repeat: 4,993 kernel events, duration sum 103.138 ms, interval union 103.111 ms, span 106.520 ms (96.80% union/span). This is repeatability evidence for trace-window event activity, not a service A/B. The remainder of kernel span includes possible device gaps and activity not represented as kernel records; do not label all remainder as CPU overhead.

A trace window covers 256 output tokens, but sampling trigger is three scheduler decode steps. SGLang decodes by continuous batching; streams/graphs may overlap and event export contains other activity within profiler lifetime. There is no exact per-forward start/end join to the kernel events. No fraction is reported as full-model GPU time explained; owner time residual remains UNKNOWN.

## Source-backed partial kernel owner inventory (not the requested Graph-node ledger)

Source cross-reference:

- GPTQ: `FalconH1MLP` constructs merged `gate_up_proj` and `down_proj` (`python/sglang/srt/models/falcon_h1.py`); `GPTQLinearScheme` dispatches `GPTQLinearKernel`; G017 runtime confirms candidate M=4/K=3072/N=24576 dispatch at graph capture. Trace has exact kernel symbol `sglang::gptq::gemm_half_q_half_alt_4bit_kernel`. For each c1 trace: 264 events, matching 2 GPTQ projections × 44 hybrid layers × 3 captured decode forward steps at the requested-count level. c4 traces also have 264 events; geometry differs by batch, and exact per-replay event grouping is not exposed.
- Attention: `FalconH1HybridAttentionDecoderLayer.self_attention` calls QKV/RoPE, `RadixAttention`, then O projection; source `python/sglang/srt/models/falcon_h1.py`. Trace records `_fwd_grouped_kernel_stage1` and `_fwd_kernel_stage2` 132 each for all c1/c4 windows, consistent with one attention decode implementation pair × 44 layers × 3 requested steps. The event name/geometry is verified from runtime trace; mapping as attention follows the configured `tilelang_fa_v100` backend and its dispatch source, but there is no graph node association.
- Mamba: same layer source calls `Mamba2AttnBackend.forward`/`MambaMixer2`; trace records `_selective_scan_update_kernel` and `causal_conv1d_update_kernel` 132 each for all c1/c4 windows (44×3 event-count consistency). Source verification needed in attached manifest; counts are strongly consistent with one per layer per forward-step but not node-ID proof.
- Other: trace contains c1 GPTQ plus attention, Mamba kernels; elementwise, norms, KV stores, memory sets, cublas GEMV, memcpy post and CPU/GPU scheduler-support events are visible. No comprehensive semantic mapping/count union was made.

Illustrative captured owner-name subsets (sum durations overlap-counting, **not** additive and not whole-owner totals):
- S1 1K/c1 GPTQ 264 events, 48.909 ms sum; attention stage1+stage2 132 each, 6.726 ms sum; Mamba selective-scan + causal-conv + layernorm 132 each, 2.873 ms sum.
- S1 4K/c4 GPTQ 264, 56.457 ms; attention stage1+stage2 132 each, 14.183 ms; Mamba same subset 5.084 ms.
- c57 1K/c1 GPTQ kernel name is absent (candidate path replaces this operator family with separately named candidate kernel; candidate owner must not be assumed missing or zero); attention pair 6.772 ms; selected Mamba subset 2.892 ms.
- c57 4K/c4 selected attention 14.051 ms; selected Mamba 5.074 ms. Candidate GPTQ candidate op is not classified by this old-S1 symbol matcher. See raw traces and analyzer; percentages are intentionally not forced to total 100%.

These are direct kernel activity events, but owner assignment is only partially DERIVED from exact kernel symbol plus source/backend; graph-node provenance and exhaustive operator coverage are UNKNOWN. No numeric GPTQ/Attention/Mamba *owner share of a single forward* is claimed.

## Route B: installed CUDA 12.8 CUPTI graph trace

Reused `/usr/local/cuda-12.8/extras/CUPTI/samples/cuda_graphs_trace/cuda_graphs_trace.cu` and NVIDIA-provided helper headers, copied to private `cupti-sample/`. Built only `sm_70` with CUDA 12.8, linked installed CUPTI, and ran with `CUDA_VISIBLE_DEVICES` pinned to V100 UUID. Sample completed on “Tesla V100-SXM2-16GB” and emitted `graphId=2`, per-node IDs, kernel symbols, launch grids/blocks, nanosecond GPU timestamps and node-creation CUDA APIs; thus CUPTI route is supported and operational on this machine. Sample uses a standalone test graph, not SGLang's captured graph.

CUPTI docs/source show why it cannot be attached using the existing worker profiling endpoint: the sample registers callbacks and activity collection **inside the CUDA process before graph creation/replay**. Kernel activity type `CUpti_ActivityKernel9` has `graphId`/`graphNodeId`; resource callbacks identify graph node creation/cloning and map IDs. SGLang's `/start_profile` worker-side PyTorch profiler exports activity traces after CUPTI collection but its Chrome event conversion drops the graph ID/node ID fields. Attaching CUPTI callbacks to an already-running worker requires an in-process extension/injection or modifying worker profiler/CUPTI collector; no existing fork API was found. The official sample is not an external attach utility. Adding a new injection/sidecar profiler is a third framework and forbidden by finite termination rule. Therefore G019 does not implement it.

## NCU and worker memory/resource status

The S1/c57 top kernel symbols/counts are present in trace, but the Graph node/owner mapping required to identify the “top three Graph owners” is not established. No NCU was run against the serving graph; the G018 fixture NCU results cannot be transferred. NCU per-owner matrix is **NOT COLLECTED** for this phase; this is a route/scope blocker, not ordinary root permission failure (G018 established authorized root NCU). Whole-model SM/Tensor/HBM activity remains UNKNOWN.

The existing profiler supports `activities=["MEM"]` memory history, and the G019 capture did not enable it. No model-worker-local allocated/reserved/peak or Graph pool memory snapshot was captured. After server shutdown a fresh, unrelated Python process had 0 allocated/reserved/peak and 16.60GB free/16.93GB total; this is idle state, not worker usage. Device-global sample last observed after worker shutdown: V100 7 MiB/0%, 1312/877 MHz, 42.81W, 42C, supported throttle mask 0x1ff, active reason 0; unrelated GPU not targeted. Worker clocks/power/thermal over the actual capture windows are not available in a corrected fresh telemetry capture. Those fields remain UNKNOWN.

## Overhead and service authority

PyTorch Profiler is intrusive and source implementation defaults `with_stack=true`; G019 explicitly set `with_stack=false` but left shape recording enabled. The trace itself has profiler events and overhead records; candidate c4 4K repeat kernel union/span differs from first capture (94.34% vs 96.80%) and duration activity is close, but this is not a controlled tracing-on/off experiment. Exact profiling overhead versus normal SGLang service remains UNKNOWN. No profiled TPS/ITL is used as service authority; old G018 instrumented matrices remain distinct and G017's accepted uninstrumented authority is not rerun.

## Final owner status and next conclusion

| Owner/quantity | Status | Result |
|---|---|---|
| Kernel activity names, per-record duration/geometry | DIRECT_MEASURED | Present in worker CUPTI export. |
| CUDA Graph ID/node ID per execution | UNKNOWN | Not exported by SGLang worker profiler; official CUPTI route only standalone sample. |
| GPTQ gate_up/down invocation count | DERIVED / partial | Source says two projections per each of 44 layers; trace has 264 S1 GPTQ events in each 3-step window, but no replay-to-node grouping. c57 kernel uses a different symbol not fully counted here. |
| Attention/Mamba event counts | DERIVED / partial | Runtime symbols and geometry plus source/backend mapping; event count 132 per matching symbol for 44×3. No node IDs; some suboperators not assigned. |
| Complete owner durations/shares | UNKNOWN | Partial symbol subsets overlap in concurrent execution and do not exhaust kernels; no valid single-forward owner sum. |
| copies/memsets and device idle gaps | DIRECT_MEASURED / partial | Some activity and overall span are present; complete Graph-node linkage and true idle decomposition unavailable. |
| Whole-forward wall reconciliation | UNKNOWN | No exact forward-window/kernel graph join. |
| Worker allocator and Graph memory pool | UNKNOWN | No worker-memory capture enabled. |
| Top Graph-owner NCU | NOT COLLECTED | No node-based top-owner authority; do not substitute G018 synthetic GPTQ fixture. |
| Full-model SM/Tensor/HBM | UNKNOWN | NCU activity not collected for full model. |
| Recoverable time budget / next kernel owner | UNKNOWN | Evidence does not support allocation. |

G019 ends **PARTIAL** under finite stop rule: Route A captures kernel activities but drops node identity; Route B official CUPTI sample successfully traces a standalone CUDA graph, but integrating it into a running SGLang worker requires a new in-process collector/attach framework not present in the fork. No third profiling system is built. Preserve all direct traces and source/tool evidence for a future separately authorized instrumentation change.
