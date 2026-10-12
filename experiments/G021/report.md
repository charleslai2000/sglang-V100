# G021 — Falcon-H1 V100 serving and GPTQ kernel measurements

**Status: measured subset reported; G021 / R17 is not accepted.** This report consolidates the completed evidence currently available. It does not claim a full-forward CUDA Graph hardware profile, complete decode owner ledger, or Cache/E2E qualification.

## Executive summary

- Actual Falcon-H1 service was measured with frozen S1 and a G017 Candidate binary that had both exact path/SHA and runtime candidate-dispatch log evidence. The Candidate binary remains opt-in; S1 remains default.
- In this 3-repeat, 256-output greedy service matrix, Candidate mean aggregate throughput exceeds S1 in each cell by approximately 5.3–8.4%. The strongest practical caveat is 4K prompt at concurrency 4: both modes show substantial queued/request latency tails. The small sample does not support p99 or robust production claims.
- Whole-device target-only NVML windows showed GPU busy around 95% on average during both matrices, with roughly 38% memory-controller busy. These NVML percentages are device-window telemetry—not SM/Tensor utilization, per-process use, or per-Graph counters.
- Root NCU collected actual Falcon-derived M4 GPTQ gate-up and down projection fixtures for each route. Candidate fixture kernel durations were 19.2% lower for gate-up and 32.7% lower for down under NCU kernel replay. These are isolated operator fixture measurements, not service or full-Graph speedups. NCU throughput metrics for SM/DRAM/L2 were `n/a`; Tensor pipe activity was zero for these kernels.
- FPM now has real Falcon Prefill/Decode CUDA Event completion and v1 active IPC payload evidence, and a three-stage S1 FPM-off/on/off service comparison found negligible change in the tested cell means. Its production identity/loss audit, direct replay-key proof for every requested graph size, and full GPU hardware/owner matrix remain incomplete. Cache hit/recovery, request queue wait, p99 and full Attention KV + Mamba state restore remain unmeasured or unsupported. No G021 acceptance decision follows from this partial report.

## Scope and method

Target GPU: Tesla V100-SXM2-16GB UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; RTX 2060 excluded. The captured service cells are 1K/4K prompt × concurrency 1/2/4, three repeats per cell, 256 greedy output tokens, using the Falcon model revision below. Service runs were uninstrumented for NCU; the subsequent NCU runs were separate. Service launch used CUDA Graph batch sizes 1/2/4, static memory fraction 0.76, tilelang FA V100 attention, radix cache disabled and piecewise CUDA Graph disabled. Candidate dispatch was separately confirmed via runtime log; mere extension loading is not the criterion.

| Route | Extension path | SHA-256 |
|---|---|---|
| S1 | `/mnt/data/pc01/sglang-v100-migration/experiments/E026/frozen-s1-runtime/sgl_kernel/sm70/common_ops.abi3.so` | `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842` |
| G017 Candidate (opt-in) | `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g017-close-final-sm70.so` | `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9` |

Model revision: `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; decode: temperature 0, greedy, ignore EOS. Each cell has 3 wave repeats. Within-wave token ID identity was true for every recorded wave. Reported TTFT / E2E percentiles pool the 3×concurrency request samples, using nearest-rank selection; ITL p50 is the median of per-request stream ITL p50 values, and ITL p95 is the p95 of per-request stream ITL p95 values. These request quantiles have only 3 samples at c1 and 12 at c4; no p99 is reported. Mean aggregate tok/s is the arithmetic mean of three wave aggregate rates. These are harness-level measured outputs, not confidence-bounded estimates.

## Service performance — OBSERVED

| Route | Prompt | Concurrency | Mean aggregate output tok/s | TTFT p50 / p95 (ms) | Per-request ITL p50 / p95 (ms) | E2E p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 1K | 1 | 33.70 | 312.7 / 312.7 | 28.55 / 32.51 | 7,596 / 7,596 |
| Candidate | 1K | 1 | 36.55 | 307.7 / 307.7 | 26.25 / 27.08 | 7,005 / 7,005 |
| S1 | 1K | 4 | 103.78 | 815.7 / 1,104.6 | 34.25 / 36.71 | 9,863 / 9,867 |
| Candidate | 1K | 4 | 111.08 | 804.7 / 1,099.7 | 31.74 / 35.27 | 9,213 / 9,220 |
| S1 | 4K | 1 | 29.68 | 1,198.3 / 1,198.3 | 29.11 / 30.03 | 8,625 / 8,625 |
| Candidate | 4K | 1 | 31.88 | 1,194.4 / 1,194.4 | 26.79 / 27.50 | 8,030 / 8,033 |
| S1 | 4K | 4 | 47.73 | 2,623.7 / 14,021.0 | 36.31 / 38.15 | 12,820 / 21,452 |
| Candidate | 4K | 4 | 50.87 | 2,622.6 / 13,294.3 | 33.48 / 34.60 | 12,093 / 20,131 |

Relative mean aggregate tok/s change Candidate vs S1 in the four reported cells: 1K/c1 +8.4%, 1K/c4 +7.0%, 4K/c1 +7.4%, 4K/c4 +6.6%. This is a descriptive comparison for these matched harness settings, not a causal confidence interval. Candidate's lower median/p95 ITL and E2E are also observed here; no claim is made about varied prompts, sampling, longer contexts, other batch schedules, or p99.

**Queue/latency interpretation:** 4K/c4 waves submit four requests together and show up to 13–14s client TTFT and ~20–21s E2E. The harness has no server-side enqueue/dequeue timestamps, so request queue waiting cannot be isolated from client TTFT. Treat these as user-visible client latencies, not decode-only service time. Aggregate wave tok/s includes request/wave scheduling behavior and is not RPS or prefill throughput.

## Whole-device NVML — OBSERVED

The per-second CSVs each contain 279 samples (~278s), targeted only to the V100 UUID, and span the six service cells; samples are not labelled per individual request/cell. Statistics are descriptive across that complete matrix window.

| Route | GPU busy mean / median / p95 | Memory busy mean / median / p95 | VRAM used mean (MiB) | SM clock mean / p95 (MHz) | Mem clock mean (MHz) | Power mean / p95 / max (W) | Temp mean / max (°C) |
|---|---|---|---:|---|---:|---|---|
| S1 | 95.25 / 99 / 100% | 38.43 / 41 / 43% | 11,478 | 1,513 / 1,530 | 877 | 225.85 / 289.30 / 307.75 | 66.90 / 72 |
| Candidate | 95.33 / 99 / 100% | 38.46 / 41 / 43% | 11,454 mean; 11,478 max | 1,521 / 1,530 | 877 | 227.03 / 290.24 / 300.93 | 67.12 / 72 |

Active throttle-reason mask included `0x4` in some samples and `0x0` in others. The mask is preserved as raw evidence; this report does not infer the reason represented by `0x4`. NVML GPU busy is a sampled whole-device busy percentage. It does not identify compute/Tensor utilization, attribute work to this process, or provide effective bandwidth.

## NCU actual Falcon-derived GPTQ M4 fixtures — OBSERVED

NCU 2025.1.1.0 used root, one selected kernel and kernel replay on the V100. Fixture tensors are captured from Falcon layer 43 production GPTQ gate-up/down execution (`x=[4,3072]`; gate-up output width 24576, down output width 3072); the wrapper was routed S1 or the SHA-pinned Candidate extension. Reports identify the exact kernel and grid. This measures the fixture's production kernel, **not a live CUDA Graph node or full Falcon decode pass**.

Values labelled `.avg` are NCU's average across profiler passes; bytes and GPU duration `.sum` are read from the same captured kernel replay boundary. Effective GB/s below is calculated as `(dram read bytes.sum + dram write bytes.sum) / gpu duration.sum` using decimal GB/s. It is replay-boundary effective DRAM traffic per kernel duration, not a sustained Graph rate. The sample may reflect replay/cache behavior and should not be extrapolated to whole-device HBM utilization.

| Route / projection | Grid (x,y,z) | GPU duration `.sum` (µs) | DRAM read/write `.sum` (MB) | Effective DRAM GB/s | L2 bytes `.avg` (MB) | SM active cycles `.avg` | SM issue active `.avg` | FP16 inst `.avg` | FMA inst `.avg` | Tensor inst / cycles `.avg` | Avg active warps / issue (NCU rate; % ref) | Long scoreboard `.avg` (warp cycles) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 gate-up | 192,1,24 | 290.496 | 39.875 / 2.342 | 145.3 | 3.139 | 364,604 | 170,734 | 134,609 | 165,137 | 0 / 0 | 18.20 (113.78%) | 1,805,282 |
| Candidate gate-up | 192,1,24 | 234.624 | 39.799 / 14.303 | 230.6 | 2.836 | 290,253 | 187,529 | 58,982 | 289,094 | 0 / 0 | 10.25 (64.05%) | 606,935 |
| S1 down | 24,1,96 | 186.112 | 19.793 / 1.709 | 115.5 | 1.890 | 220,165 | 86,385 | 67,962 | 83,226 | 0 / 0 | 21.62 (135.14%) | 1,206,732 |
| Candidate down | 24,1,96 | 125.248 | 19.771 / 5.391 | 200.9 | 1.290 | 155,734 | 93,771 | 29,491 | 144,547 | 0 / 0 | 9.24 (57.74%) | 276,988 |

NCU-replay duration difference: Candidate gate-up 19.2% shorter, Candidate down 32.7% shorter for this captured M4 input. **Inference limited to these isolated fixture kernels:** the two operators execute faster under the Candidate route in the NCU replay capture. This is not a whole-forward latency reduction. Candidate moves more FMA work and writes more DRAM bytes than S1 in the captured kernel; do not interpret the lower active-warps aggregate proxy as occupancy percentage. `sm__throughput.avg`, `dram__throughput.avg`, and `lts__throughput.avg` were returned as `n/a` for GV100 in these reports, not as zero. Tensor-pipe counters were observed zero for these two kernels in both routes; this says nothing about unrelated kernels in the model.

## Matched S1 FPM overhead experiment — OBSERVED, limited

A three-stage OFF→ON→OFF comparison was run after the earlier service runs proved to use different output lengths. The three phases used S1 SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`, the same Falcon revision, V100 UUID, launcher/runtime parameters, Graph capture configuration [1,2,4], max-running-requests=4, prompt construction, streaming harness, 256 greedy outputs, and 1K/4K × c1/c4 with three waves per cell. The Candidate remained unloaded. Raw request JSON and target-only 1 Hz whole-device telemetry are stored outside the repository at `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g021-fpm-{off-phase1,on-256-phase2,off-phase3}.json` and matching `*-telemetry.csv`; phase logs are under `/mnt/data/pc01/sglang-v100-migration/logs/g021-fpm-{off-phase1,on-256-phase2,off-phase3}.log`. Every request returned 256 output tokens.

| Phase | Prompt / c | Mean aggregate output TPS | Mean client TTFT (ms) | Mean client streamed ITL (ms) | Mean client E2E (ms) |
|---|---|---:|---:|---:|---:|
| OFF1 | 1K / 1 | 33.53 | 300.0 | 28.77 | 7,635.9 |
| ON | 1K / 1 | 33.49 | 309.1 | 28.76 | 7,643.5 |
| OFF2 | 1K / 1 | 33.48 | 308.4 | 28.77 | 7,645.3 |
| OFF1 | 1K / 4 | 73.95 | 2,369.1 | 33.70 | 10,962.8 |
| ON | 1K / 4 | 73.93 | 2,363.4 | 33.69 | 10,953.8 |
| OFF2 | 1K / 4 | 59.69 | 3,046.5 | 32.84 | 11,420.2 |
| OFF1 | 4K / 1 | 29.54 | 1,210.7 | 29.24 | 8,666.6 |
| ON | 4K / 1 | 29.55 | 1,212.9 | 29.22 | 8,663.0 |
| OFF2 | 4K / 1 | 29.60 | 1,194.6 | 29.23 | 8,649.1 |
| OFF1 | 4K / 4 | 29.58 | 14,179.3 | 29.24 | 21,634.9 |
| ON | 4K / 4 | 29.59 | 14,178.0 | 29.22 | 21,628.2 |
| OFF2 | 4K / 4 | 29.59 | 14,176.3 | 29.23 | 21,631.1 |

At 1K/c4, both OFF phases showed substantial wave dispersion: OFF1 TPS `[59.23,59.86,102.76]`, ON `[59.43,102.45,59.90]`, OFF2 `[59.33,59.89,59.84]`. The ON TPS mean is essentially the OFF1 mean but above OFF2 because one fast wave shifted phase ordering; it is not evidence of a large overhead. Per-request TTFT/E2E includes queueing; actual server logs show continuous batching and queued requests. The source harness uses client-side wall timings; it does not capture server request admission or prove identical per-wave scheduler batch sequences. Across service logs, decode running batch 1/2/3/4 counts were OFF1 `134/6/12/7`, ON `146/0/13/6`, OFF2 `135/0/18/0`; Prefill new-sequence counts were OFF1 `47×1, 3×3`, ON `44×1, 3×3`, OFF2 `41×1, 3×3`. Thus identical configuration and request plan are confirmed, but realized batch schedules differ (and SGLang logs only periodic samples); this is a bounded service overhead estimate, not a perfectly paired scheduler trajectory. NVML CSVs contain 451 OFF1 and 478 ON samples; OFF2 sampler duration was interrupted by process cleanup and is not treated as a complete comparison window. Whole-device utilization is not per-process attribution.

**Disposition:** For these S1, 256-token workloads, FPM-on/off means show no material TPS, TTFT or ITL change at c1 and 4K/c4; 1K/c4 is noisy but its paired ON mean is not lower than OFF1 and individual waves overlap substantially. This does not support the earlier apparent 1K/c4 103.78→75.11 or 4K/c4 47.73→24.22 differences as instrumentation effects: those were different service harness/output-length conditions (96 outputs in the later phase-varying benchmark versus 256 in the earlier matrix) and are not valid direct pairs. This three-stage test has 3 waves/cell but only 3 c1 requests/cell and 12 c4 requests/cell, so tail quantiles/P99 are not qualified. No production performance guarantee follows.

## FPM, Cache, and R17 disposition

- **FPM: PRODUCTION CAPTURE PASS for the bounded, already-accepted Falcon S1 path.** Commit `631cfc89c2644d398a4555fcfbcb509919275bef` is remotely verified. A ready-before-start exact `.0` subscriber capture reconciled scheduler Forward/timer/active emits/queue accepts at 2,357 each; publisher sent and subscriber received 2,506 messages (2,357 active plus 149 idle), with zero observed drops, failures, pending shutdown work, decode errors or sequence gaps. The experimental sidecar joins Forward identity, request snapshot, CUDA Event duration and graph selection; the v1 schema is unchanged and retains scheduler `wall_time` semantics. This is a bounded pass, not a PUB/SUB durability guarantee. FPM scope is closed; do not add further FPM qualification gates. See the final-capture section and audit artifact below. Earlier endpoint/lifecycle captures are superseded for the accepted-path decision.
- **Decode response percentiles: PARTIAL OBSERVED.** TTFT, per-request streamed ITL p50/p95, and E2E p50/p95 are reported at 3 repeats only. p99 is not statistically characterized. No queue start/dequeue timestamps, queue-wait duration, RPS, or independently measured prefill token throughput exist in these artifacts.
- **Cache cold/warm or hit rate: UNSUPPORTED / NOT MEASURED.** No token-weighted hit rate, cache layer recovered bytes/time, skipped prefill tokens, or cold-vs-warm causal test is present.
- **State recovery (Attention KV + Mamba state): UNSUPPORTED / NOT MEASURED.** No restoration-path execution or recovery correctness/latency evidence is present.
- **Whole-forward kernel-owner ledger / live Graph NCU: NOT MEASURED HERE.** Existing archived traces are separately bounded/partial; fixture NCU cannot fill their unknown ownership. Generic GEMV/GEMM names remain unowned absent verified source/call path.
- **Full R17 E2E acceptance: INCOMPLETE.** No status upgrade is implied. S1 remains default; Candidate stays exact-SHA opt-in.

## Reproduction and evidence files

All paths are relative to repository root `experiments/G021/`.

- Service source JSON: `artifacts/service-s1-all-cells.json`, `artifacts/service-candidate-dispatch-verified.json`.
- NVML source CSV: `artifacts/nvml-s1-matrix.csv`, `artifacts/nvml-candidate-matrix.csv`. `nvml-candidate-dispatch-verified.csv` is an additional dispatch-verified sample window, not the six-cell matrix summary above.
- NCU reports: `artifacts/ncu-s1-gateup.csv`, `ncu-candidate-gateup.csv`, `ncu-s1-down.csv`, `ncu-candidate-down.csv`.
- Offline summary calculator: `artifacts/analyze_ncu_service.py` (no GPU action).
- FPM: invalid old unsuffixed-endpoint outputs `artifacts/fpm-candidate.json`, `fpm-candidate-full.json`, `fpm-candidate-dispatch-verified.json`; corrected-endpoint zero-active captures `artifacts/fpm-candidate-fixed-endpoint-retry2.json`, `artifacts/fpm-candidate-fixed-endpoint-40step.json`, `artifacts/fpm-candidate-fallback.json` and their subscriber/server logs. Current-v1 compatible bounded decoder: `artifacts/fpm_subscriber_compat.py`. External archived `.0` bind corroboration: `/mnt/data/pc01/sglang-v100-migration/logs/g017-final-paired-s1-2.log`.
- Resource/restart constraints and preflight: `preflight.md`, `resource-gate.csv`, `r17-resume-config.json`, `r17-result-schema.json`.

**Bounded answer:** Under the tested local V100 Falcon-H1 settings, exact-SHA G017 Candidate dispatch is real and its M4 gate-up/down fixture kernels ran faster than S1 in targeted NCU replay; the original uninstrumented service matrix also observed higher mean output throughput in the four reported c1/c4 cells. The final bounded production FPM capture passes for the accepted S1 path and the matched OFF→ON→OFF 256-token runs provide a bounded overhead result; neither expands the measurement boundary. Whole-model/Graph SM/Tensor/HBM utilization, request queue lifecycle, cache/state recovery, p99 and several V1 matrix dimensions remain unmeasured or explicitly unsupported below. Candidate stays opt-in; S1 remains default. G021/R17 remains ACTIVE until its frozen metrics matrix is completed or each item is dispositioned with evidence.

## FPM final bounded production capture — observed, not general delivery guarantee

This section supersedes earlier report statements that real Falcon shutdown counters, identity, and loss accounting were absent. On 2026-10-11, a fresh S1-only Falcon service used exact endpoint `ipc:///tmp/g021-fpm-s1.0`. The subscriber established the exact socket and wrote its READY marker before server launch; its monotonic capture window enclosed the 12-wave/30-request, 1K/4K × c1/c4, 96-output request plan and graceful shutdown. Each request returned 96 tokens. V100 UUID was `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; Candidate was not loaded. The shutdown path used a private scheduler control request after request draining and logged an acknowledgement before scheduler exit, permitting the existing cleanup/final audit to run.

| Ledger stage | Count | Result |
|---|---:|---|
| Scheduler Forward complete / CUDA Event timer complete | 2,357 / 2,357 | matched |
| Active emit attempted / accepted into publisher queue | 2,357 / 2,357 | matched |
| Internal queue drops / duplicate callbacks / unmatched timers / timeout | 0 / 0 / 0 / 0 | none observed |
| Scheduler pending batches / completed GPU iters / timer intervals at shutdown | 0 / 0 / 0 | drained |
| Publisher sent / send-dropped / send-failed | 2,506 / 0 / 0 | includes idle heartbeat traffic |
| Subscriber received / active / idle heartbeat | 2,506 / 2,357 / 149 | active counts match |
| Subscriber transport sequence | 0…2,505 | contiguous, zero gaps/errors |

The final forward sidecar contains 2,357 unique `forward_iter` rows: 2,355 user-workload rows and 2 health-probe rows. It records per-Forward request IDs and pre-forward input/output token snapshot, scheduler wall duration, GPU Event duration, graph selection, and queue disposition. The service request harness independently confirms 30 non-health requests across 12 waves, each returning 96 tokens. The active v1 receiver rows account for the same 2,357 active records. The v1 schema itself remains unchanged: scheduler `wall_time` semantics remain distinct from Event time, and v1 carries no Forward identity or GPU elapsed.

For this bounded capture, 43 Extend forwards accumulated 75,021 scheduler-reported scheduled prefill tokens over 21.682844 CUDA Event seconds (diagnostic ratio **3,459.92 tokens/s**). This token total is the scheduler's sidecar field and is not asserted to equal client prompt tokens or cache-exclusive recomputation. The 2,312 user Decode forwards attributed 2,888 output-token steps over 68.065327 CUDA Event seconds (**42.4298 output-token steps/s**); CUDA Event interval p50/p95 was **28.794/33.646 ms**. Decode graph selections were key 1 for 2,024 forwards and key 4 for 288 actual scheduled batch-3 forwards (padding to graph key 4); no Extend forwards used Graph. The independent controlled graph audit previously recorded replay keys 1/2/4 (73/66/63 calls). Scheduler wall duration, CUDA Event elapsed, and sums of intervals remain separate; do not infer a GPU active union from these overlapping interval sums. These event/token ratios are request-workload-aligned diagnostics, not qualified end-to-end request-token throughput or a hardware counter profile.

The OFF1→ON→OFF2 overhead evidence remains the separate three-phase 256-output experiment above; it must not be conflated with this 96-output production accounting capture. In the overhead experiment, each phase had three waves/cell; at 1K/c4 a single ~103 TPS wave occurred in OFF1 and ON but not OFF2, so the data do not support a deterministic overhead percentage or a strict same-schedule causal estimate. Across the other cells mean changes are small. This is bounded overhead evidence only.

Artifacts and provenance: detailed machine-readable audit at `artifacts/fpm-source-gpu-event-audit.json`; FPM implementation is committed and remotely verified at `g018-local-v100-gpu-audit` SHA `631cfc89c2644d398a4555fcfbcb509919275bef`; external server log `/mnt/data/pc01/sglang-v100-migration/logs/g021-fpm-delivery-window.log`; request sidecar `/mnt/data/pc01/sglang-v100-migration/experiments/E026/fpm-request-sidecar.jsonl`; subscriber capture and READY marker `.../fpm-subscriber-delivery.json` and `.../fpm-subscriber-delivery.ready.json`; request plan/results `.../fpm-production-delivery-requests.json`. The subscriber script is `G018/experiments/E027/fpm_reconcile_subscriber.py`.

**Residual limitations:** Zero missing messages/gaps were observed in this ready-before-start local IPC window; PUB/SUB remains best-effort, not durable or universal exactly-once. The 96-token accounting capture differs from the 256-token overhead comparison. No dedicated prefill/decode hardware-counter Graph profile, supported-cache recovery matrix, adequately powered p99/concurrency characterization, complete owner ledger, or 3090-comparability result is added. FPM production capture **PASS** is limited to this accepted S1 path; **G021/R17 remains ACTIVE**.
