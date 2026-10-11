# G021 plan

Status: **ACTIVE — measured subset complete; remaining R17/attribution work continues.** User authorized resume and termination of the prior resource job. No ACCEPTED claim.

## Resource release

- Confirmed authorized process PID `3903363`, UID 1002, Python/24-thread Parquet dedup scan from `/mnt/data/neuron/stock/E001-cortex-data-001/canonical/us`; direct bash PID `3903362`, same PGID/SID, no other child/session process. Sent `sudo kill -TERM 3903363` only. Target process and shell exited. Data directory remains with 413 entries.
- Post-stop 30-minute 1-minute-cadence resource series: `experiments/G021/resource-gate.csv`. 31 rows, min MemAvailable 16.055 GiB, min SwapFree 39.42 GiB, min `/mnt/data` free 302.87 GiB, 66% used; V100 7MiB / 0% throughout. PSI was zero for 29 samples; one avg60 transient 0.01% returned to zero; avg10/300 zero. The prior 20GiB gate was unmet, but user explicitly authorized resume; used previously successful G019 S1 graph configuration at static fraction 0.76 and monitored continuously. No global policies changed.

## Real S1/Candidate service + whole-device NVML (OBSERVED)

S1: `service-s1-all-cells.json` SHA module `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`. Candidate dispatch-verified: `service-candidate-dispatch-verified.json`, module `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`; server emitted the exact candidate dispatch path/SHA log while graph warmup/runtime. Both normal service series have 1K/4K × c1/c2/c4, 3 repeats/cell, 256 greedy tokens, exact within-wave token equality. No NCU was active during these client runs. Same accepted harness/launcher, model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`, graph sizes [1,2,4], radix/piecewise disabled, pinned V100.

Per condition mean aggregate tok/s; request TTFT median/p95 ms; request TPOT proxy is per-request streamed ITL p50 median and p95 of each request's p95 ITL; request E2E median/p95 ms. (All request samples are pooled across the three repeats.)

| Mode | ctx | c | tok/s | TTFT p50/p95 ms | ITL p50/p95 ms | E2E p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 1K | 1 | 33.70 | 312.7 / 312.7 | 28.55 / 32.51 | 7596 / 7596 |
| Candidate | 1K | 1 | 36.55 | 307.7 / 307.7 | 26.25 / 27.08 | 7005 / 7005 |
| S1 | 1K | 4 | 103.78 | 815.7 / 1104.6 | 34.25 / 36.71 | 9863 / 9867 |
| Candidate | 1K | 4 | 111.08 | 804.7 / 1099.7 | 31.74 / 35.27 | 9213 / 9220 |
| S1 | 4K | 1 | 29.68 | 1198.3 / 1198.3 | 29.11 / 30.03 | 8625 / 8625 |
| Candidate | 4K | 1 | 31.88 | 1194.4 / 1194.4 | 26.79 / 27.50 | 8030 / 8033 |
| S1 | 4K | 4 | 47.73 | 2623.7 / 14021.0 | 36.31 / 38.15 | 12820 / 21452 |
| Candidate | 4K | 4 | 50.87 | 2622.6 / 13294.3 | 33.48 / 34.60 | 12093 / 20132 |

4K/c4 12-request waves are saturated/queued: some requests have TTFT 13–14s and ~21s E2E. Request service harness has no server queue/start timestamps, so queue time is not separately measured; TTFT is client-side and includes queue. 1K/c1-c4 and 4K/c1 mean TPS, plus p50/p95 as shown, are ordinary uninstrumented throughput/latency authority, with client timing limitations. No p99 inference from 3 repeats; each cell has 12 requests at c4 and 3 at c1.

Target-only 1Hz whole-device NVML files: `nvml-s1-matrix.csv`, `nvml-candidate-matrix.csv`; both 279 samples / 278s. These windows overlap the six service cells, not an individually labelled request. S1 gpu_busy mean 95.25%, median 99%, p95/max 100%; memory_busy mean 38.43%, p95 43%, max 44%; VRAM 11,478MiB; SM clock mean 1513 / p95 1530 MHz; memory clock 877MHz; power mean 225.85W / p95 289.3 / max 307.75W; temp max72C. Candidate: GPU busy mean95.33%, median99%, memory busy mean38.46%, VRAM max 11,478MiB, SM mean1520.65 / p951530MHz, mem877MHz, power mean227.03 / p95300.93W, temp max72C. Throttle mask `0x4` appeared in some samples, so throttle was not always clear; map raw supported/active mask before a textual attribution. These are device-wide percentages, **not SM throughput or per-process share**.

## Actual S1/Candidate GPTQ M4 kernel NCU fixture measurements (OBSERVED but kernel-scoped)

The reused E026 NCU script profiles the captured Falcon L43 real gate_up/down M4 GPTQ fixture, with actual S1/candidate route, NCU kernel replay; this is real Falcon-derived operator input, **not a live serving Graph**. Target was isolated by `sudo env CUDA_VISIBLE_DEVICES=<V100 UUID>`; verified physical V100 only. Reports are under `experiments/G021/artifacts/ncu-{s1,candidate}-{gateup,down}.csv`; G017 candidate SHA was validated and client Candidate dispatch separately logged.

NCU's `.avg` values for these are averaged over profiler passes; `.sum` values aggregate the replayed capture. GPU kernel duration `.sum`, DRAM bytes `.sum`, and L2 bytes `.sum` refer to same NCU replayed kernel boundary. Calculated `(read+write bytes.sum)/gpu__time_duration.sum` is NCU replay average effective DRAM bandwidth; it is **not** a whole-Graph or steady serving bandwidth.

| Mode / projection | grid (x,y,z) | duration sum ms | read MB | write MB | HBM effective GB/s | L2 bytes.sum MB | FP16 pipe inst avg | FMA pipe inst avg | Tensor inst / Tensor cycles | active warps/issue avg | long-scoreboard avg (warp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 gate_up | 192,1,24 | .290496 | 39.875 | 2.342 | 145.3 | 100.5 | 134609 | 165137 | 0 / 0 | 18.20 (113.78% of NCU ref rate; aggregate instance proxy, not CTA occupancy) | 1,805,282 aggregate warp cycles |
| Candidate gate_up partial | 192,1,24 | .234624 | 39.799 | 14.303 | 230.6 | 90.763 | 58982 | 289094 | 0 / 0 | 10.25 (64.05% of NCU ref rate) | 606,935 aggregate warp cycles |
| S1 down | 24,1,96 | .186112 | 19.793 | 1.709 | 115.5 | 60.477 | 67962 | 83226 | 0 / 0 | 21.62 (135.14%) | 1,206,732 aggregate warp cycles |
| Candidate down partial | 24,1,96 | .125248 | 19.771 | 5.391 | 200.9 | 41.268 | 29491 | 144547 | 0 / 0 | 9.24 (57.74%) | 276,988 aggregate warp cycles |

NCU `sm__throughput.avg`, `dram__throughput.avg`, and `lts__throughput.avg` were reported `n/a` (not supported/evaluable for this aggregation on GV100), not zero. DRAM active cycles and bytes are populated; Tensor pipe has zero activity in these GPTQ kernels. Candidate reduces fixture kernel elapsed by ~19.2% gate_up and ~32.7% down under NCU replay; this is per operator fixture, not per-forward service savings. Both routes remain non-Tensor kernels; Candidate shows greater FMA count and write traffic. HBM ratio uses replays bytes/duration and should not be compared to nominal whole-device HBM peak as a whole-Graph rate.

## Still required / boundary constraints

1. Per request P99 (more repeats) and direct queued duration/start timestamps are absent; current c4 TTFT/E2E quantiles are coarse. No independent prefill TPS/RPS/Prefill TPS metric, request-start/queued telemetry, saturation curve or queue wait. Existing harness reports end-to-end wave tokens/sec only.
2. NVML is whole-device; not grouped by request. NCU fixture isn't live-serving Graph counters. Need targeted NCU on a live real selected graph/kernel if reasonably bounded; graph aggregate mode still untested. Do not infer whole device from fixture counters.
3. G020 actual decode event ledger has broad names/geometries. Source map must be verified before ascribing generic GEMV/GEMM. Existing 4K c4 is actual bs3 in archived traces; new service concurrent graph traces still need profile counter mapping.
4. The existing FPM subscriber produced zero records in two runs despite server announcing an IPC endpoint; artifact contains zero records and is not used. DeviceTimer timer is integrated in model runner, but no forward event stream validated yet. Diagnose subscriber pattern using accepted E027 method without changing Graph semantics.
5. Cold/Warm cache, token-weighted hit rate, L1/L2/L3 actual cache recovered bytes/time, skipped prefill tokens, full Attention KV + Mamba state restoration and saved computation are not measured. Inspect actual SGLang APIs/capability and mark Unsupported with evidence where architecture lacks restoration.
6. p99, cache/R17 full E2E acceptance and source-backed owner time/Recoverable budget remain incomplete. No ACCEPTED status. Candidate remains opt-in; S1 default unchanged.
