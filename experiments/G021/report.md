# G021 — Falcon-H1 V100 Inference Performance Metrics V1

**Status: G021/R17 ACTIVE — partial metrics matrix.** **FPM production capture PASS for the bounded accepted Falcon S1 path.** This is a measurement report, not a warranty or full bottleneck acceptance. G017 remains ACCEPTED; S1 is default and Candidate remains exact-path/SHA opt-in. No compute math, CUDA/driver policy, or unrelated GPU was changed by this work.

## Executive findings

- Tested Falcon-H1-7B-Instruct-GPTQ-Int4 on Tesla V100-SXM2-16GB UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`. The accepted S1 extension SHA is `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; G017 Candidate SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9` remains opt-in.
- In the existing uninstrumented 256-output service matrix, S1 mean aggregate throughput was 33.70 tokens/s at 1K/c1, 103.78 at 1K/c4, 29.68 at 4K/c1, and 47.73 at 4K/c4. Candidate observed 36.55, 111.08, 31.88, 50.87 respectively. These are three wave repeats/cell, not a statistically robust production interval. At 4K/c4 the S1 request TTFT p95 was 14.02 s among 12 requests; existing client measurements do not identify the per-request queue/admission cause.
- Target-only whole-device NVML in S1's six-cell matrix reported GPU busy mean 95.25% (median 99%, p95 100%) and memory-controller busy mean 38.43% (p95 43%). Those are sampled whole-device busy percentages, **not** SM compute throughput, Tensor utilization, HBM bandwidth, or per-process attribution. VRAM was 11,478 MiB; SM clock mean/p95 1,513/1,530 MHz, memory clock 877 MHz, power mean/p95/max 225.85/289.30/307.75 W, temperature max 72 C.
- Targeted root NCU evidence is for actual Falcon-derived L43 M4 GPTQ operator fixtures, not live whole-model Graph: S1 gate-up duration 0.290496 ms, DRAM read+write 42.217 MB, effective boundary ratio 145.3 GB/s; S1 down 0.186112 ms, 21.502 MB, 115.5 GB/s. Tensor instruction counters were zero for those two GPTQ fixture kernels. These values do not answer whole-forward HBM/SM/Tensor utilization.
- G020 offline G019 trace owner windows show for stable S1 1K/bs1 annotation: GPTQ gate_up/down 9.91/6.40 ms, Attention 2.46 ms, identified Mamba 0.69 ms, UNKNOWN 7.69 ms; interval-union kernel span 29.58 ms within 30.55 ms annotated step. At bs4 the corresponding owner sums are 11.41/7.30/2.91/1.79/7.20 ms (Other GEMM), UNKNOWN 2.19 ms, union 35.90 ms within 36.98 ms. These are intrusive profiled trace-window activity sums (some overlap), not normal service wall time or exclusive owner budgets. Graph node IDs and full hardware counters are absent.
- The 96-output FPM capture provides request-owned scheduled GPU-work snapshots and CUDA Event timing. Across 43 EXTEND forwards, `sum(req.extend_input_len)=75,021`; summed `PrefillStats.log_input_tokens=75,376`, a 355-token discrepancy to retain, not reconcile by guess. Cached-token snapshots summed to zero with radix disabled. Dividing the per-request scheduled token sum by the sum of CUDA Event forward intervals (21.6828 s) gives a diagnostic 3,459.9 scheduled tokens/s, **not** complete request prompt TPS or GPU active union. Decode yielded 2,888 token steps over 68.0653 Event-seconds = 42.43 steps/s; again diagnostic interval-sum ratio, not qualified Decode GPU TPS. Step Event p50/p95 is 28.79/33.65 ms; scheduler wall interval sum was 140.00 s across decode Forwards, a different boundary.
- A bounded `bench_one_batch` S1 1K/c1 synthetic smoke on 2026-10-12, with valid vocabulary IDs and accepted S1 runtime binaries, measured 3,595.75 input tokens/s and 32.48 output tokens/s median Decode. This is single-run standalone diagnostic, not production service authority. Follow-up root NCU measured real Falcon Prefill EXTEND and Decode Graph replay **selected kernel records** (KERNEL_SCOPE), including SM throughput, Tensor/FP16 and SASS arithmetic activity, DRAM bytes/peak throughput, L2 sector hit, occupancy/issue and long-scoreboard stall. It is not a full-Graph aggregate or GPU active-union measurement; NCU replay slows its instrumented benchmark substantially. Full details and boundaries are below.
- Cache, request lifecycle wait, c8+ saturation, statistically useful p99, matched 3090 comparison, and complete model-Graph hardware counters remain not measured or incomplete. No single optimization-owner recommendation is supportable because unknown GEMV/GEMM and hardware-level throughput are unresolved.

## Unified matrix and provenance

The row-level evidence matrix is `artifacts/r17-evidence-matrix.json`, regenerated offline by `build_metric_matrix.py`; hash/provenance manifest is `artifacts/manifest.json`. Each available S1/Candidate service condition (1K/4K × c1/c2/c4) records binary SHA, model revision, request counts, repeat counts, output TPS, TTFT/ITL/E2E sample summaries, explicit unavailable fields and boundaries. FPM and G020/NCU/NVML data remain distinct records by capture and boundary. The script does not initialize CUDA or load the model.

**Canonical repository / consolidation:** This G021 report is maintained in the single `sglang-V100` checkout. Prior G017/G018/G019/G020 checkpoints remain in Git branch history, not as separate source checkouts: G017 `origin/g017-close-prod-checkpoint` (`149c70f10b49ff1d1fe28ac5a7016526157a04a9`); G018 `origin/g018-local-v100-gpu-audit` (`631cfc89c2644d398a4555fcfbcb509919275bef`); G019 `origin/g019-partial-graph-attribution` (`db08f603033ccebb252ed8f8cb209b2d0203eda1`); G020 `origin/g020-existing-trace-offline` (`572bcd751`). G018's source implementation was measured at its pinned commit; it is not overlaid on the later mainline source because the SGLang source evolved independently afterward. Its report and scripts are archived at `experiments/E027/`; G019 report, manifest, and standalone CUPTI sample are archived at `experiments/G019/` and `cupti-sample/`; G020 report, analysis, script, and manifest are at `experiments/G020/`. Raw large profiler/client captures remain in the external store and are SHA-identified in manifests; no duplicate raw traces were copied into Git.

Statuses are per-metric: `MEASURED`, `PARTIAL`, `NOT_MEASURED`, `NOT_RUN`, `UNSUPPORTED`, or `UNRESOLVED_COMPARABILITY`. No absent counter is zero-filled. For every small-n c1/c4 service sample, p99 is deliberately not reported as qualified.

## Service performance — uninstrumented; observed

The archived S1 and Candidate service JSON each contain 18 waves covering 1K/4K × c1/c2/c4, three repeats per cell, 256 greedy output tokens/request, target V100, model revision above. Candidate dispatch/path identity is verified in its server evidence. The primary c1/c4 summary (mean of three wave aggregate rates; request quantiles pooled over 3 c1 or 12 c4 requests) is:

| Route | Prompt | c | Output tok/s | TTFT p50 / p95 ms | ITL p50 / p95 ms | E2E p50 / p95 ms | request n |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1K | 1 | 33.70 | 312.7 / 313.7 | 28.55 / 32.54 | 7,596 / 7,599 | 3 |
| Candidate | 1K | 1 | 36.55 | 307.7 / 308.7 | 26.25 / 27.08 | 7,005 / 7,007 | 3 |
| S1 | 1K | 4 | 103.78 | 815.7 / 1,104.6 | 34.25 / 36.71 | 9,863 / 9,867 | 12 |
| Candidate | 1K | 4 | 111.08 | 804.7 / 1,099.7 | 31.74 / 35.27 | 9,213 / 9,220 | 12 |
| S1 | 4K | 1 | 29.68 | 1,198.3 / 1,199.8 | 29.11 / 30.03 | 8,625 / 8,627 | 3 |
| Candidate | 4K | 1 | 31.88 | 1,194.4 / 1,196.0 | 26.79 / 27.50 | 8,030 / 8,033 | 3 |
| S1 | 4K | 4 | 47.73 | 2,623.7 / 14,021.2 | 36.31 / 38.15 | 12,821 / 21,452 | 12 |
| Candidate | 4K | 4 | 50.87 | 2,622.6 / 13,295.2 | 33.48 / 34.60 | 12,093 / 20,132 | 12 |

The service harness reports output throughput, not independent request/s or input TPS. Request IDs exist within run JSON, but there is no per-request server arrival→enqueue→admission→prefill timestamp chain. TTFT is client request-start to first streamed token and includes any queue/admission wait. In particular, 4K/c4 S1 has a 14.02 s observed p95 TTFT; existing data cannot separate queue from prefill or cache wait. No p99 claim is justified by 3/12 requests. c2 data are in the matrix and raw JSON; not repeated here.

## Whole-device NVML — observed, window-scoped

S1 and Candidate each have target-UUID-only 1 Hz NVML CSVs with 279 samples (~278 s) spanning six cells, not tagged per request/cell. S1 GPU busy mean/median/p95 95.25/99/100%; memory-controller busy 38.43% mean, 43% p95; VRAM 11,478 MiB; SM clock 1,513 MHz mean / 1,530 p95; memory clock 877 MHz; power 225.85 W mean / 289.30 p95 / 307.75 max; temperature max 72 C. Candidate matrix values are preserved in the matrix/report source: GPU busy mean 95.33%, memory busy 38.46%, SM clock mean 1,520.65 MHz, power mean 227.03 W, max temperature 72 C. Throttle mask `0x4` occurred in some prior samples; without mapping the exact raw active mask and sample window, no causal thermal/power claim is made. NVML percentages are not compute or HBM counters.

## NCU hardware — actual Falcon-derived operator fixtures, not live Graph

Root NCU 2025.1.1.0, kernel replay, target V100 UUID isolation profiled Falcon L43 M4 gate-up/down operands for S1 and Candidate. Per-fixture results in `artifacts/ncu-*.csv`; bytes/duration share the replayed kernel boundary. S1: gate-up 0.290496 ms, DRAM read/write 39.875/2.342 MB, effective ratio 145.3 GB/s; down 0.186112 ms, 19.793/1.709 MB, 115.5 GB/s. Candidate gate-up 0.234624 ms, 39.799/14.303 MB, 230.6 GB/s; down 0.125248 ms, 19.771/5.391 MB, 200.9 GB/s. Candidate fixture elapsed is 19.2% / 32.7% lower for these isolated gates, not whole-forward savings. Tensor pipe counters are zero in these four GPTQ kernels.

GV100 reports `sm__throughput.avg`, `dram__throughput.avg`, and `lts__throughput.avg` as `n/a` for these fixture kernel replay aggregates; this is unsupported/evaluation failure for those specific fixture aggregations, not zero. A previous `--graph-profiling graph` attempt was rejected: it yielded only an initialization kernel after 81 replay passes. The corrected NCU node capture and actual Falcon Prefill/Decode KERNEL_SCOPE measurements are reported in the next section; they do not make fixture reports whole-model evidence. PCIe aggregate, full Graph-level aggregate counters, launch overhead and complete Forward L2 rate remain NOT COLLECTED.

## Falcon S1 NCU real-call hot kernels — measured, KERNEL_SCOPE

A follow-up replaced the earlier invalid profiler-window attempt. The accepted S1 standalone harness ran in the actual ModelRunner process with Mamba selective-state-update initialization, legal synthetic Falcon vocabulary IDs and the scheduler's existing deferred H2D resolve boundary. The clean G017 path used the exact accepted S1 extension SHA and SHA-matched prebuilt SM70 dense/MoE runtime shared objects; no tracked inference source or math changed. NCU root 2025.1.1.0 targeted only V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, tracked all process descendants, used `--profile-from-start off`, an explicit start/end NVTX marker around the warm post-initialization measured phase, and `--graph-profiling node --replay-mode kernel`. NCU reports include the actual worker PID and `R17_MEASURED_FORWARD` range. The range also contains framework work; a kernel-name allowlist then restricted results to observed Falcon call-path hot kernels. Prefill's EXTEND is a regular launch path (radix/piecewise disabled); Decode is a completed, captured CUDA Graph replay (graph keys [1,2,4], runtime batch 1). Kernel identities/grids are also present in the uninstrumented standalone Decode trace. The exact benchmark/profile command and artifacts are in `metric-differential-notes.md` and matrix; normalization script `parse_ncu_hotspots.py` produces the external `s1-hotspots-normalized.csv`.

The table reports arithmetic means across NCU replay records for an identical kernel/grid, **not sum over all operators or whole-Forward time**. Prefill NCU duration units are ms; Decode are µs. DRAM/L2 byte counters use NCU's displayed Mbyte/Gbyte units, while throughput percentages use NCU's supported sustained-peak elapsed metrics. No rows are summed as a budget; overlapping and sequential operator timings are not an active-union measurement.

| Phase / observed Falcon kernel | Grid | NCU replay rows | GPU duration | SM throughput | Tensor inst / FP16 inst | FP32 SASS FADD/FMUL/FFMA inst | DRAM read / write | DRAM / L2 throughput | L2 sector hit | active warps / issue active / long-scoreboard stall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill CUTLASS MLP gate-up projection | 64×7×1 | 4 | 0.5432 ms | 73.82% | 260,506 / 339 | 0 / 0 / 21,709 | 74.379 / 14.341 Mbyte | 18.21% / 48.46% | 88.98% | 11.66% / 52.19% / 3.29% |
| Prefill Mamba chunk scan | 128×4×24 | 4 | 0.9662 ms | 29.97% | 0 / 154 | 83,558 / 312,925 / 3,863,347 | 27.009 / 12.845 Mbyte | 4.60% / 25.14% | 95.63% | 55.08% / 30.42% / 11.87% |
| Prefill CUTLASS attention projection | 64×24×1 | 4 | 1.6724 ms | 86.60% | 471,859 / 2,784 | 0 / 0 / 0 | 252.113 / 49.608 Mbyte | 20.10% / 55.00% | 89.78% | 12.41% / 33.84% / 1.86% |
| Decode GPTQ gate-up | 192×1×24 | 2 | 254.304 µs | 42.94% | 0 / 16,712 | 0 / 0 / 0 | 39.401 / 1.851 Mbyte | 18.37% / 28.40% | 78.34% | 53.24% / 44.44% / 61.57% |
| Decode GPTQ down | 24×1×96 | 2 | 166.656 µs | 33.53% | 0 / 8,494 | 0 / 0 / 0 | 19.701 / 1.711 Mbyte | 14.66% / 26.40% | 82.07% | 52.68% / 36.65% / 67.72% |
| Decode vocab GEMV (observed grid) | 835×1×1 | 2 | 62.080 µs | 28.99% | 0 / 2,515 | 2,317 / 8,016 / 64,128 | 41.060 / 1.533 Mbyte | 83.67% / 28.80% | 51.15% | 42.10% / 27.93% / 45.63% |
| Decode Mamba selective scan | 32×1×24 | 2 | 13.824 µs | 34.03% | 0 / 38 | 5,530 / 6,144 / 11,674 | 3.167 / 1.461 Mbyte | 43.10% / 18.92% | 51.90% | 62.77% / 40.75% / 32.01% |

NCU did not expose `smsp__inst_executed_pipe_fp32.avg` in the report/query for this GV100 set; SASS FADD/FMUL/FFMA counts are separate generic floating-operation instructions and are **not relabeled FP32-only**. Tensor activity was present in selected Prefill CUTLASS projections and zero in the selected Decode kernels. The actual S1 graph path executed the built-in Falcon GPTQ GEMM, cublas GEMV, CUTLASS/Mamba kernels; the SM70 Marlin MoE extension is unrelated to Falcon-H1's architecture and did not appear in its model config or observed decode kernel trace. No MoE result is included.

NCU replay is highly intrusive here: benchmark's same-run output under NCU fell to roughly 9–11 tok/s during instrumentation versus ~3.6K input tok/s prefill outside NCU. This difference is profiler cost, **not** service performance or NCU overhead correction. Normal service TPS/TTFT remains the uninstrumented service matrix. Reports: `/mnt/data/pc01/sglang-v100-migration/profiles/r17-ncu-tmp/s1-prefill-pipe-final.ncu-rep` (SHA `81894e694e22522dc9987c4378dafd0ffc7c0b11a303a1192c7d8689f14780b7`) and `s1-decode-pipe-final.ncu-rep` (SHA `8ce82c0243279464b898b75508b303fa494e99f949f0e2d609b9b98b8a3c3801`). Normalized external CSV SHA is recorded in `artifacts/manifest.json`; parser `parse_ncu_hotspots.py` is in this report directory. These are selected hot kernel records from a real Falcon Forward, not full Graph counters or a complete HBM/SM efficiency budget.

## G020 GPU decode owner ledger — partial profile window

G020 reused real G019 PyTorch/CUPTI trace windows; same V100 model/route lineage, profiler enabled, 3 requested decode steps. G020 annotations have common GPU timestamp coordinates with GPU activities and let the offline analyzer bound stable step windows, but not CUDA Graph node identities. Representative S1 stable step averages:

| Condition / actual annotated batch | GPTQ gate_up | GPTQ down | Attention | identified Mamba | Other GEMM | Elementwise/norm | memcpy | UNKNOWN | kernel interval union / annotation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K / bs1 | 9.91 ms | 6.40 | 2.46 | 0.69 | 0.00 | 2.19 | 0.26 | 7.69 | 29.58 / 30.55 ms |
| 1K / bs4 | 11.41 ms | 7.30 | 2.91 | 1.79 | 7.20 | 2.82 | 0.27 | 2.19 | 35.90 / 36.98 ms |
| 4K / bs1 | 9.86 ms | 6.36 | 3.00 | 0.69 | 0.00 | 2.17 | 0.26 | 7.76 | 30.08 / 31.04 ms |
| 4K / bs3 | 11.49 ms | 7.33 | 4.94 | 1.43 | 7.21 | 2.87 | 0.28 | 2.20 | 37.72 / 39.59 ms |

Owner duration sums can overlap; interval union removes temporal overlap. `UNKNOWN` includes generic GEMV/GEMM with no source-backed callsite. Annotation tail and auxiliary streams are not forced into owners. Actual graph-node association, complete owner exclusivity across all GPU activity, SM/Tensor/HBM utilization, and unprofiled cost are not available. G020 is trace-window attribution only; do not compare it by subtraction to a different FPM run.

## FPM production capture — PASS, bounded accepted Falcon S1 path

FPM implementation SHA `631cfc89c2644d398a4555fcfbcb509919275bef` was pushed and verified. This 2026-10-11 S1 run pinned the target V100, had radix and piecewise graphs disabled and Graph capture sizes [1,2,4], and used 12 waves / 30 requests at 1K/4K × c1/c4, 96 outputs each. An exact `.0` endpoint subscriber wrote READY before service launch; shutdown handshake drained scheduler work and persisted final publisher/timer counters.

| Ledger | Count | Observation |
|---|---:|---|
| scheduler Forward complete / Event timer complete / active emit / queue accept | 2,357 each | matched |
| internal queue drop / timer duplicate / unmatched / timeout | 0 each | none observed |
| pending scheduler snapshot / completed GPU iter / Event at shutdown | 0 each | drained |
| publisher sent / send-drop / send-failed | 2,506 / 0 / 0 | includes 149 idle heartbeats |
| subscriber received / active / idle | 2,506 / 2,357 / 149 | contiguous sequence 0..2505, zero gaps/errors |

Forward identity and request-owned snapshots are in the research sidecar, not v1 wire. Across 43 user Extend forwards, request-owner sum(req.extend_input_len)=75,021; aggregate scheduler PrefillStats.log_input_tokens sum=75,376, difference 355 retained as a token-definition discrepancy. The single health-check Extend adds one request token to all-Extend owner sum (75,022). Cache snapshots are zero (radix disabled). 43 Extend Forwards' Event interval sum is 21.6828437 s; 2,312 Decode Forwards' is 68.0653266 s for 2,888 output-token steps. Event interval p50/p95 28.794/33.646 ms. Decode actual scheduled batch distribution was 2,024 bs1 and 288 bs3; selected Graph keys 1 and 4 respectively (bs3 padded to graph4). Other configured replay key evidence is in the controlled direct replay audit. Sidecar carries scheduler wall and CUDA Event separately; do not sum overlapping intervals as active union.

The Event-based ratios 3,459.9 Extend scheduled tokens/s and 42.43 Decode steps/s are diagnostic interval-sum ratios. The latter is explicitly **not** the complete GPU Decode TPS required by R17. FPM production capture is PASS for this bounded already-accepted S1 path; the pub/sub transport remains best effort by design, not durable exactly-once. No further FPM qualification gate is needed or authorized. Overhead evidence remains separate: same-S1 OFF→ON→OFF three waves/cell, 256 outputs; ordinary cells show no material shift, 1K/c4 is high variance (a ~103 TPS wave in both OFF1 and ON but not OFF2), so no deterministic instrumentation percentage is claimed.

## Standalone Prefill/Decode speed diagnostic — single-run, not service authority

The out-of-tree `bench_one_batch` harness smoke (2026-10-12) ran the accepted S1 SGLang base and extension SHA under target UUID isolation. An ephemeral harness shim added the existing Mamba selective-state-update initializer omitted by this standalone command, constrained synthetic IDs to the local vocabulary and staged IDs at the scheduler's existing deferred-H2D boundary. The already-built SM70 MoE/dense Marlin runtime extensions were copied from the accepted main-tree runtime location into the clean G017 worktree after exact SHA equality checks; no tracked source or compute math changed. Runtime extension hashes and command/config are recorded in `metric-differential-notes.md`.

| Measurement | Result | Boundary / qualification |
|---|---:|---|
| Prefill, 1,024 synthetic input tokens, c1 | 3,595.75 tokens/s; 0.28478 s | Standalone synchronized wall duration, single run; not production traffic or cache/service authority |
| Decode, c1, 64 requested outputs | median 32.48 tokens/s; 30.79 ms/step | Median excludes initial step; single run |
| Prefill profiler EXTEND | 281.444 ms CUDA total | Single `step[EXTEND bs=1 toks=1024]`; profiler scoped attribution |
| Decode profiler Graph window | 246.441 ms summed Self CUDA / 8 = 30.805 ms/step | Eight selected decode steps, Graph replay; not full-device active union |

This is a first direct speed datum, not R17's complete workload-bound GPU TPS: it is one c1 input, synthetic prompt IDs, model warmup, standalone request path, and no statistical repetitions. A root NCU Graph attempt used this harness, `--graph-profiling graph`, profiler start/stop controls and 81 replay passes. Its report contained only one ordinary prefill initialization `unrolled_elementwise_kernel`, not the serving Forward Graph. No full Graph counter result is claimed. NCU global query confirms the relevant GV100 metric names exist; capability inventory does not repair the failed boundary. Keep full-Graph SM/Tensor/HBM status NOT_COLLECTED pending an attributable capture.

## Cache and state capability — source audit; restore/benefit NOT_RUN

Source review against the clean accepted G017 tree (`/home/charles/Workspaces/3rdparty/sglang-V100-G017-close`, pinned accepted S1 runtime identity) establishes capability boundaries, not measured Falcon cache benefit:

- **GPU L1 Radix:** implemented for hybrid SSM via `MambaRadixCache`; the hierarchical selector chooses `HiMambaRadixCache` when hierarchical caching and hybrid SSM are enabled (`python/sglang/srt/mem_cache/registry.py`). The measured accepted service explicitly disabled radix caching and the FPM request-owned cached-token sum was zero. Cold/warm benefit is **NOT_RUN**, not zero benefit.
- **Host L2 HiCache:** the HiMamba implementation has host KV and Mamba backup/restore hooks (`hi_mamba_radix_cache.py`), while `hybrid_pool_assembler.py` assembles host pools and `memory_pool_host.py` contains Mamba state pools. The source models Mamba storage as a distinct `PoolName.MAMBA` transfer; the storage v2 API computes usable prefix from KV and auxiliary-pool hits, and HiMamba tracks restore nodes separately. This is evidence that joint state handling is implemented, but does **not** prove a valid full Falcon prefix-frontier restore in a running server. No runtime HiCache configuration, successful restored-token accounting, correctness replay, transfer timing, or service benefit was collected. Status for empirical Falcon L2 restore correctness/benefit: **NOT_RUN** (source support present; outcome unknown), not UNSUPPORTED.
- **Disk L3:** the server accepts optional storage backends (`file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, and `simm` per `server_args.py`), and HiMamba has storage archive/prefetch hooks. No storage backend was enabled in these captures. L3 restore and read/write cost are **DISABLED/NOT_RUN** for the dataset; do not infer disk support for a particular deployment from the generic backend list.

Accordingly, no tier-exclusive hit tokens, recompute reduction, H→D/Disk bytes or time, cache wait, effective restore throughput, TTFT/E2E delta, or host/disk benefit is measured. No cache hit benefit or restore-overhead claim is made.

## Scheduler/request lifecycle and saturation

Existing client service JSON remains the uninstrumented throughput/latency authority and lacks server lifecycle fields. Separately, a new bounded S1 capture enabled the existing `--enable-request-time-stats-logging` flag (no source change) and used unique request IDs, single waves at 4K/c4, 4K/c1, 1K/c4 and 1K/c1, each request 256 greedy outputs. The client artifact and server `ReqTimeStats` log join **10/10** request IDs. The server reports `queue_duration = forward_entry_time - wait_queue_entry_time` and `forward_duration = completion_time - forward_entry_time`; client TTFT is request start to first streamed response and includes queue plus any first-token work/transport. This is an instrumented follow-up capture, not a replacement for the uninstrumented p95/service matrix.

| Context/concurrency | Request | Scheduler queue duration | Scheduler forward-to-completion | Client TTFT |
|---|---|---:|---:|---:|
| 4K/c4 | a3b2d6c3 | 25,620.53 ms | 8,600.43 ms | 27,070.48 ms |
| 4K/c4 | bf4a2aeb | 16,991.85 ms | 8,599.59 ms | 18,451.62 ms |
| 4K/c4 | 56469486 | 8,367.65 ms | 8,595.13 ms | 9,814.96 ms |
| 4K/c4 | bba5dc16 | 0.20 ms | 8,606.70 ms | 1,190.88 ms |
| 4K/c1 | 08b6dd5b | 0.24 ms | 8,596.77 ms | 1,197.99 ms |
| 1K/c4 | 145844d1 | 0.33 ms | 9,840.05 ms | 824.02 ms |
| 1K/c4 | 561cdec6 | 0.38 ms | 9,839.91 ms | 823.99 ms |
| 1K/c4 | 9637c805 | 251.32 ms | 9,589.23 ms | 1,125.13 ms |
| 1K/c4 | 405f25e6 | 251.36 ms | 9,589.10 ms | 1,125.51 ms |
| 1K/c1 | 2db04784 | 0.70 ms | 7,558.88 ms | 317.62 ms |

The four-request 4K wave shows progressively queued requests with ~8.6s forward-to-completion each; their large TTFT tail is directly explained by scheduler wait-queue residence plus first request service before each later admission. Zero cached input was reported in the request timing log; radix cache was disabled. This is one wave, only n=4 at 4K/c4 and n=1 other cells, so it explains an observed mechanism but does not establish a new quantile or all historical tail events. `forward_duration` is a lifecycle wall span that contains repeated decode scheduling; it is not CUDA time. TokenizerManager response timestamp and scheduler monotonic-to-wallclock `entry_time` are present in existing metadata/logs; do not equate client transport TTFT with GPU execution.

Artifacts: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/r17-s1-lifecycle-client.jsonl` and `/mnt/data/pc01/sglang-v100-migration/logs/r17-lifecycle-reqtime.log`, with full server log at `logs/r17-lifecycle-s1-server.log`; request-id join script `experiments/G021/analyze_lifecycle_capture.py`. Periodic running/queued samples remain non-ID-correlated and cannot be substituted for per-request times. FPM sidecar is Forward-level, not the complete lifecycle. Per-request cache wait, GPU admission delay, scheduling gaps and preemption remain NOT_MEASURED. c1/c2/c4 service cells exist; c8/12/16/20/24 saturation and safe upper bound were not run. This was a deliberate NOT_RUN: the bounded service matrix does not provide per-request live KV/Mamba pool headroom or a validated safe running-request limit above the accepted `max_num_reqs=4`; no larger concurrency point is claimed safe from static token capacity alone. Existing service quantiles pool only 3 requests at c1 and 12 at c4, so P99 is not statistically supportable. The single instrumented c4 lifecycle wave gives queue-residence evidence but does not fill saturation, admission delay, scheduler-gap, cache-wait, or GPU-time fields.

## Remaining metric dispositions (A–G)

- **A, hardware:** selected actual Falcon kernel counters and whole-device NVML are measured only at their stated boundaries. Full-Graph aggregate and active-union counters remain **NOT_RUN** after the prior graph-profile attempt was rejected for capturing initialization only. PCIe H→D/D→H bandwidth and CUDA launch overhead have no valid per-workload measurement in retained evidence: **NOT_RUN**, not zero/unsupported. Achieved occupancy, issue and stall values are available only for selected kernel records; no whole-Graph claim.
- **B, Prefill/Decode:** FPM is PASS in its bounded scope; Event sums/step percentiles and the standalone synthetic speed run remain diagnostic. Qualified complete request-bound GPU active TPS is still **PARTIAL**.
- **C, lifecycle:** queue residence is **PARTIAL measured** in one joined wave; admission subphases, scheduling gap, per-request GPU timing and cache wait are **NOT_RUN**.
- **D, HiCache:** see source audit above; GPU Radix/host paths source-supported, disk backend optional, all measured cold/warm/restore benefits **NOT_RUN**.
- **E, service:** c1/c2/c4 measured for stated bounded workload; c8+ saturation and statistically supported P99 **NOT_RUN**.
- **F, owner:** G020 and selected NCU records **PARTIAL**; unknown and nonexclusive owner time remains.
- **G, delivery/comparability:** matrix, manifest, scripts and provenance are retained. 3090 direct comparison is **UNRESOLVED_COMPARABILITY** because source data are not available/aligned.

## Required research answers and status

1. **Actual GPU Prefill/Decode speed — PARTIAL.** Request-owned scheduled prefill token snapshots and forward Event intervals are available, with 355-token aggregate-field discrepancy; Event ratios are not GPU active TPS. Decode Event steps/s excludes a shared active-union boundary.
2. **Service performance — PARTIAL / MEASURED at tested points.** See table and matrix. Output TPS and client TTFT/ITL/E2E p50/p95 reported; input TPS/RPS and p99 unavailable.
3. **Actual SM/Tensor/HBM use — PARTIAL.** Whole-device NVML only for service windows; fixture-level GPTQ NCU counters only. No full Graph counters.
4. **Major owner GPU time — PARTIAL.** G020 trace-step owner table plus selected live-call NCU records; unknown GEMV/GEMM and missing complete Graph-node/exclusive-owner mapping prevent a full budget.
5. **Queue/admission/gap outside GPU — PARTIAL.** The separate 10/10 RID-joined lifecycle capture measured queue residence and forward-to-completion wall span; one 4K/c4 wave ties the TTFT tail to queue residence. Admission subphases, GPU prefill/first-token time, inter-forward scheduling gap, cache wait, and preemption remain unmeasured. Do not claim the entire lifecycle outside-GPU budget.
6. **Cache hit reduced GPU prefill? — NOT_RUN.** Cache disabled, observed hits zero.
7. **Host/disk restore cost vs benefit? — NOT_RUN.** No such restore workload; complete Falcon state pairing not proven.
8. **First saturation point? — NOT_RUN beyond c4.** No c8+ scan.
9. **V100 vs RTX3090 same workload/boundary? — UNRESOLVED_COMPARABILITY.** The current checkout and searched capture store contain no raw RTX 3090 service dataset/config that can be aligned. The control record identifies an external historical E021 result (`/home/ubuntu/workspaces/3rdparty/sglang/.pi/experiments/E021-s3-v100-optimization/results.md`), not available in this environment. No direct ratio is calculated; workload, actual batch, cache and timing identity remain unverified.
10. **Best next optimization / recoverable budget? — UNRESOLVED.** Selected NCU kernels and G020 owner windows are not an exclusive uninstrumented whole-forward budget; do not nominate a kernel owner or recoverable time from them. Current concrete evidence only supports a bounded queue-tail mitigation hypothesis for the single 4K/c4 wave, not a general optimization budget.

## Artifacts and reproduction

- Matrix builder: `build_metric_matrix.py`; output `artifacts/r17-evidence-matrix.json`.
- Hash manifest: `artifacts/manifest.json` (includes retained source CSV/JSON and external FPM raw hashes).
- FPM reconciliation: `artifacts/fpm-source-gpu-event-audit.json`; implementation `g018-local-v100-gpu-audit` SHA above; exact subscriber utility and final notes at G018 `experiments/E027/`.
- Existing services: `artifacts/service-s1-all-cells.json`, `artifacts/service-candidate-dispatch-verified.json`; NVML: `artifacts/nvml-{s1,candidate}-matrix.csv`; fixture NCU: `artifacts/ncu-{s1,candidate}-{gateup,down}.csv`.
- Trace owner analysis: `experiments/G020/analysis.json`, offline generator `experiments/G020/analyze_traces.py`; raw traces remain in external G019 storage and are hashed in their manifest.
- FPM external raw files: `/mnt/data/pc01/sglang-v100-migration/logs/g021-fpm-delivery-window.log`, `experiments/E026/fpm-request-sidecar.jsonl`, `fpm-subscriber-delivery.json`, `.ready.json`, `fpm-production-delivery-requests.json`.

FPM PASS does not qualify the full performance contract. G021/R17 remains ACTIVE until remaining metrics are measured or individually evidence-dispositioned; no full bottleneck conclusion is claimed.
