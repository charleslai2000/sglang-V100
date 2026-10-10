# G018 bounded V100 Falcon-H1 audit — partial close report

Status: **PARTIAL / ACTIVE**. This report does not claim the requested complete Falcon decode owner ledger or an optimization budget. No kernel/default change is authorized or made.

## Authority and scope

- Base: accepted G017 `149c70f10b49ff1d1fe28ac5a7016526157a04a9`; S1 remains default.
- Target: Tesla V100-SXM2-16GB UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189` only. The RTX 2060 UUID was not profiled. At the last cleanup check, the target had 7 MiB allocated globally and 0% utilization; no G018 server, sampler or subscriber remained.
- Model: Falcon-H1-7B GPTQ, local pinned revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; 44 hybrid decoder layers.
- Frozen S1 module SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; accepted G017 candidate SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`.
- Full-service Nsys remains DEFERRED. No global driver/profiling policy, DCGM service, host or driver state was changed.

## Existing instrumented service evidence (not throughput authority)

The three-repeat 1K/4K × c1/c2/c4 matrices, 256 greedy outputs, are instrumented service measurements. They are not a substitute for uninstrumented A/B and are not used as production throughput authority. Raw artifacts:

- S1 FPM: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g018-s1-fpm-v2.json`
- S1 service: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g018-s1-fpm-v2-service.json`
- Candidate FPM: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g018-candidate-fpm.json`
- Candidate service: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g018-candidate-fpm-v2-service.json`

These service result files report aggregate generated tokens/sec, wave wall, TTFT and client inter-token intervals. A descriptive mean of the three repeats per cell is:

| Prompt | c | Mode | Instrumented aggregate tok/s | Mean TTFT (ms) | Mean client ITL p50 (ms) | Mean client ITL p95 (ms) | Wave wall (s) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1K | 1 | S1 | 33.76 | 312 | 28.48 | 29.27 | 7.58 |
| 1K | 1 | c57 | 36.59 | 309 | 26.21 | 27.07 | 7.00 |
| 1K | 2 | S1 | 59.43 | 567 | 31.50 | 32.47 | 8.62 |
| 1K | 2 | c57 | 64.04 | 564 | 29.11 | 30.00 | 8.00 |
| 1K | 4 | S1 | 103.99 | 960 | 34.16 | 35.36 | 9.85 |
| 1K | 4 | c57 | 111.14 | 961 | 31.75 | 32.69 | 9.21 |
| 4K | 1 | S1 | 29.71 | 1199 | 29.04 | 29.89 | 8.62 |
| 4K | 1 | c57 | 31.93 | 1199 | 26.72 | 27.59 | 8.02 |
| 4K | 2 | S1 | 46.66 | 1921 | 33.62 | 34.42 | 10.97 |
| 4K | 2 | c57 | 49.51 | 1915 | 31.20 | 32.13 | 10.34 |
| 4K | 4 | S1 | 47.78 | 5416 | 34.41 | 35.45 | 21.43 |
| 4K | 4 | c57 | 50.91 | 5229 | 31.77 | 32.76 | 20.11 |

These data descriptively show lower client ITL and higher instrumented aggregate rate for candidate in these captures. They do not establish an uninstrumented service gain. TTFT includes client/server prefilling and scheduler effects; it is not a prefill GPU duration.

## Whole-forward event evidence and missing 4K coverage

The G018-only FPM instrumentation placed CUDA events around the whole `ModelRunner.forward_decode` model call and around CUDA graph replay, and classified scheduler iterations by decode/extend. FPM is not a request-scoped timing. Means of captured decode-event duration by active batch size:

| Active decode requests | S1 events | S1 mean (ms) | c57 events | c57 mean (ms) |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 28.382 | 1034 | 26.097 |
| 2 | 1023 | 31.453 | 1023 | 29.040 |
| 4 | 722 | 34.177 | 1023 | 31.666 |

KV coverage is aggregate sum over scheduled requests. S1 observed total decode KV ranges c1 1025–1280, c2 2050–2560, c4 4100–5120. Candidate observed c1 1025–4106 (only ten long-context rows near 4K), c2 2050–2560, c4 4100–5120. Therefore these event means are predominantly ~1K/request and **do not answer steady decode at 4K/request**. The candidate's ten c1 long-context events averaged approximately 26.94 ms; there is no matched S1 slice and no 4K c2/c4 coverage. A later bounded attempt to restart S1 did initialize the service once but lost the client/control process at the harness timeout; a second retry also timed out before usable FPM records. No partial or inferred values are reported as 4K data.

FPM retained 14 S1 and 18 candidate prefill/extend iteration records. Per-iteration extend event values are whole-forward interval evidence, not client TTFT. Event timing includes the captured graph's replay on the CUDA stream; it excludes Python scheduling outside the event boundary and does not create owner timing.

## Graph and owner ledger

Static source audit: each non-idle Falcon-H1 hybrid decoder layer executes self-attention, Mamba2 linear-attention, then feed-forward MLP; GPTQ applies to MLP gate_up and down projections. Runtime graph capture logs show batch sizes 1, 2 and 4 captured; `CudaGraphRunner.replay()` invokes the captured graph by batch key. FPM records observe scheduler batch counts, not per-kernel invocation counts inside graph replay. No measurement node/hook was inserted inside graph capture.

| Owner | Status | Evidence and disposition |
|---|---|---|
| Whole model decode/Graph forward event | DIRECT_MEASURED | FPM CUDA event; counts/means above, only matched for observed KV ranges. |
| GPTQ gate_up and down owner time in full Falcon forward | UNKNOWN | No graph-safe per-owner events or invocation-count trace. |
| Attention owner time | UNKNOWN | No owner timing; cannot derive from whole forward. |
| Mamba owner time | UNKNOWN | No owner timing; cannot derive from whole forward. |
| Other GEMM/elementwise/norm/communication | UNKNOWN | Not instrumented separately. |
| Graph/host/device-forward gap | UNKNOWN | CUDA-event forward and host/service wall have different scopes; no additive subtraction. |
| Residual | UNKNOWN | No non-overlapping complete owner measurement set; do not normalize to whole-forward. |

Only the total forward CUDA-event measurements are DIRECT_MEASURED. No full-model owner is DERIVED, ESTIMATED, or assigned a numeric time. Consequently no defensible recoverable-time budget or next kernel optimization owner is established.

## Targeted NCU: prior permission blocker resolved for root, but only one real GPTQ op fixture collected

G016 records already authorized narrowly scoped root NCU on V100 (`RmProfilingAdminOnly:1`). The archived `/tmp/g016-root-ncu.ncu-rep` is present, SHA-256 `aed3156bdf2e237279711bc10bcb239ece64c5a1b980760398fd6f9233ada835`; report metadata identifies Tesla V100 CC7.0 and a standalone `g016_probe`, not model performance. G018 rechecked device mapping: index 1 is V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; index 0 is the unrelated RTX 2060. `sudo -n` succeeds and root `ncu --query-metrics --chip GV100` works. Ordinary-user NCU denial remains expected; no permission setting changed.

Using root NCU, CUDA visibility pinned by UUID, and the G018 environment, one real SGLang `gemm_half_q_half_alt_4bit_kernel` from `gptq_gemm` was profiled for the isolated synthetic Q4/G128 operator fixture M=1,K=N=4096. It is **not** a Falcon-serving graph replay and must not be extrapolated to per-layer/full-model owner time. S1 and c57 module paths were switched to the exact SHA-pinned artifacts, and the same fixture and command shape were used. One launch was captured per mode (NCU used three replay passes for the requested counters):

| Counter | S1 | c57 |
|---|---:|---:|
| NCU `gpu__time_duration.avg` | 68.768 us | 66.016 us |
| `sm__issue_active` (% peak sustained elapsed) | 35.60% | 37.13% |
| `sm__pipe_tensor_cycles_active` | 0% | 0% |
| FP16→FP16 tensor-path ops | 0 | 0 |
| DRAM bytes read | 8,441,952 | 8,441,184 |
| DRAM bytes written | 1,457,920 | 1,476,512 |
| DRAM cycles active (% peak sustained elapsed) | 16.51% | 16.25% |
| L2 sector throughput (% peak sustained elapsed) | 7.96% | 8.39% |
| L2 sector hit rate | 40.01% | 40.72% |
| long scoreboard stalled warps / issue-active | 8.12 | 8.17 |
| registers/thread | 40 | 40 |
| NCU occupancy-per-block-size field | 2369 (raw NCU occupancy field; not percent) | 2369 (raw NCU occupancy field; not percent) |

The captured kernel is an integer unpack/reorder Q4 GPTQ op with FP16 output, so zero FP16 Tensor-path operations is consistent with that op's instruction path; it does not characterize Attention/Mamba or other forward kernels. Counter evidence is collected for this isolated GPTQ fixture only. NCU report paths and SHA-256:

- S1 `/tmp/g018-s1-gptq-full.ncu-rep`, `432675f3506e6c38d90cd1bece8721c27aa5e84ddb9cb87457bc6f070c9d047b`.
- c57 `/tmp/g018-c57-gptq-full.ncu-rep`, `718e9562f17190c2cd2544ac548dce40031f5706fc6e7abe5b30328bb278a04e`.

The first broad `--set basic` exploratory report was not used in the matched table because it omitted several desired counters. A subsequent identical explicit metric set acquired three passes per mode. No Falcon-serving process was profiled by NCU. The counters can show that this fixture is not sustaining peak issue/DRAM activity, but do not alone prove the whole model's limiting factor or motivate a kernel change.

## Worker memory and device telemetry

Existing target-only nvidia-smi windows are 600 seconds S1 and 650 seconds candidate. Positional parsing was required because the old files contain script timestamp + nvidia-smi timestamp under the old header. For those old windows, parsed means were S1: sampled GPU utilization 0%, SM clock 1314 MHz, memory clock 877 MHz, power 43.00 W, temperature 45.12 C, global memory used 11186 MiB; candidate: 38.25%, 1393 MHz, 877 MHz, 118.78 W, 53 C, 11336.53 MiB. Candidate had 14/650 samples with active reason bit 0x4. These are device-wide samples, and do not say which process owns the memory. S1's zero sampled utilization indicates that its telemetry interval did not overlap the measured request window; it is not proof the service used zero GPU.

The corrected v1 target sampler produces 13 header-aligned columns and passed a short 3-row target UUID smoke. Existing captures were not rewritten. A versioned schema/parser and artifact manifest are not yet committed.

The attempted detached allocator helper returned zeros because it ran in a separate process; those observations are invalid for model-worker allocator state. A separate no-model CUDA process reported 0 allocated/reserved/peak and about 16.60 GB free of 16.93 GB total, also not worker data. Thus worker `allocated`, `reserved`, `peak`, CUDA Graph pool bytes and worker-local free memory remain **UNKNOWN**. Global nvidia-smi used memory is explicitly not represented as worker allocator usage.

DCGM/DCGMI unavailable and service inactive. No DCGM metric was collected. Global profiling policy was left unchanged.

## Evidence-ranked limitations and next direction

1. The Falcon whole-forward decode event is ~28.4/31.5/34.2 ms for the measured S1 ~1K/request batch sizes 1/2/4, but the requested matched 4K c2/c4 event evidence is still absent. Candidate corresponding observed-range means are ~26.1/29.0/31.7 ms. These are the only direct steady-forward figures available; they are not service ITL nor full 4K results.
2. GPTQ synthetic fixture NCU shows 36–37% issue-active, 16% DRAM cycles-active, zero Tensor-path instructions, and ~8 long-scoreboard warps/issue-active; no isolated metric establishes whole-model bandwidth/compute limitation. Tensor and DRAM/SM activity of the full Falcon forward remain UNKNOWN.
3. Full owner attribution and residual are UNKNOWN. No recoverable budget is defensible; the next work should be a bounded, capture-safe diagnostic that obtains matched 4K FPM and worker-local allocator data, then a graph-safe means to count or time owners without changing replay semantics. If this cannot be done within authorized facilities, close G018 as PARTIAL/BLOCKED with uncertainty rather than infer owner shares.

Not collected: uninstrumented matched service throughput; matched S1/candidate 4K c2/c4 forward events; actual per-owner graph replay call counts/times; worker-local allocator and graph pool bytes; whole-forward NCU SM/Tensor/DRAM/L2/stalls; DCGM. These remain UNKNOWN, not zero.
