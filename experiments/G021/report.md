# G021 — Falcon-H1 V100 serving and GPTQ kernel measurements

**Status: measured subset reported; G021 / R17 is not accepted.** This report consolidates the completed evidence currently available. It does not claim a full-forward CUDA Graph hardware profile, complete decode owner ledger, or Cache/E2E qualification.

## Executive summary

- Actual Falcon-H1 service was measured with frozen S1 and a G017 Candidate binary that had both exact path/SHA and runtime candidate-dispatch log evidence. The Candidate binary remains opt-in; S1 remains default.
- In this 3-repeat, 256-output greedy service matrix, Candidate mean aggregate throughput exceeds S1 in each cell by approximately 5.3–8.4%. The strongest practical caveat is 4K prompt at concurrency 4: both modes show substantial queued/request latency tails. The small sample does not support p99 or robust production claims.
- Whole-device target-only NVML windows showed GPU busy around 95% on average during both matrices, with roughly 38% memory-controller busy. These NVML percentages are device-window telemetry—not SM/Tensor utilization, per-process use, or per-Graph counters.
- Root NCU collected actual Falcon-derived M4 GPTQ gate-up and down projection fixtures for each route. Candidate fixture kernel durations were 19.2% lower for gate-up and 32.7% lower for down under NCU kernel replay. These are isolated operator fixture measurements, not service or full-Graph speedups. NCU throughput metrics for SM/DRAM/L2 were `n/a`; Tensor pipe activity was zero for these kernels.
- FPM capture produced zero records. Cache hit/recovery, request queue wait, p99, RPS, prefill throughput, and full Attention KV + Mamba state restore remain unmeasured or unsupported. No G021 acceptance decision follows from this partial report.

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

## FPM, Cache, and R17 disposition

- **FPM: UNRESOLVED/NO OBSERVATION.** Three saved subscriber payloads have empty `records` arrays. Server launch had the metrics flag and IPC endpoint, but no decodable forward-pass records were received. Do not use FPM-derived queue or GPU-forward timing.
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
- FPM empty outputs: `artifacts/fpm-candidate.json`, `fpm-candidate-full.json`, `fpm-candidate-dispatch-verified.json`.
- Resource/restart constraints and preflight: `preflight.md`, `resource-gate.csv`, `r17-resume-config.json`, `r17-result-schema.json`.

**Bounded answer:** Under the tested local V100 Falcon-H1 settings, exact-SHA G017 Candidate dispatch is real and its M4 gate-up/down fixture kernels ran faster than S1 in targeted NCU replay; the uninstrumented service matrix also observed higher mean output throughput in the four reported c1/c4 cells. Whole-device busy/resource use was high in both service windows. Evidence does not establish full-model/Graph SM/Tensor/HBM utilization, queue decomposition, cache/state recovery, p99, or R17 acceptance. Candidate remains opt-in and S1 remains the default.
