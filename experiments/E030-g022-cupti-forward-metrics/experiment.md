# E030 — G022 bounded CUPTI metrics on real Falcon Forward ranges

## Question
Can root CUPTI Range Profiling, with no simultaneous NCU/DCGM/Kineto profiler, collect one-pass counters over exact real Falcon Prefill EXTEND and Decode CUDA Graph1/2/4 Forward ranges on the designated V100 while maintaining MemAvailable >=8 GiB?

## Identity and scope
- Repo `/home/charles/Workspaces/3rdparty/sglang-V100`, existing dirty checkout preserved; no model/kernel/dispatch changes.
- Falcon-H1-7B-Instruct-GPTQ-Int4, model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`, S1 runtime extension identity as harness environment (no Candidate selection).
- Only V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; RTX 2060 excluded.
- Harness `/tmp/g022_cupti_runner.py`; runner `/tmp/g022_cupti_cell_smactive.sh` (Graph decode) and `/tmp/g022_cupti_cell_smactive_offgraph.sh` (eager Prefill); host memory supervisor logs each 2 seconds and aborts below 8 GiB.
- CUPTI 12.8 Range Profiler: UserRange + UserReplay, one metric/pass where available; no CUDA Profiler API / Kineto / DCGM / NCU concurrent with CUPTI collection. Complete SGLang model setup, warmup, real Forward; CUPTI range begins after warmup immediately before requested Forward and ends after Forward GPU synchronization. CUDA Event brackets on that same Forward are emitted separately.

## Procedure
Use valid deterministic Falcon input IDs `[0,127000)`; preflight target UUID/idle and MemAvailable. Run one workload per process, use supported c1/c2/c4 graph size as requested, cap graph capture to that size; Prefill disables CUDA Graph. CUPTI host pass count is queried before collection; current runner refuses multi-pass profiles (separate counters must be captured in separate matched runs). Keep each benchmark's full log in `/mnt/data/pc01/sglang-v100-migration/logs/`; CUPTI values and event bracket in log. Raw input fixture retained under `/mnt/data/pc01/sglang-v100-migration/experiments/G022/cupti-fixtures/` when saver succeeds.

## Result (2026-10-13)

**PASS for repeated CUPTI Forward-range utilization vectors.** Corrected CUPTI 12.8 Range Profiler UserRange/UserReplay plus CUDA ApplicationReplay captured matched seven-metric vectors (SM active, SM throughput, active warps, Tensor pipe active, raw DRAM read/write counters, DRAM throughput, L2 hit) three times each for Prefill EXTEND 1K/4K and Decode Graph1/2/4 (actual batch 1/2/4). L2 throughput, which requires 15 passes on GV100, was captured twice per workload using the same state-restoring ApplicationReplay path. All accepted runs verified final `allPassSubmitted=true`, zero dropped ranges, and replay completion; fixture SHA, batch, input length, and per-run Forward CUDA Event are in the raw supervisor logs.

The earlier one-pass result has been superseded for repeated vector coverage. The corrected lifecycle and original `INSUFFICIENT_PRIVILEGES`/`HARDWARE_BUSY` history are retained. No external profiler was disrupted, no global profiler ownership policy changed, and RTX2060 was not accessed. Supervisor enforced MemAvailable >=8 GiB; accepted runs remained safely above the floor and returned V100 idle.

Full measured table, conditions, counter semantics, raw log identities/SHA-256, and limitations: [`../../control/experiments/G022/cupti-forward-utilization.md`](../../control/experiments/G022/cupti-forward-utilization.md). These are repeated single-Forward range observations, not a DCGM/NVML 100-ms time series, peak or device-wide mean. `dram__throughput...` percent-of-peak is reported as a CUPTI formula output. The `dram__bytes_*` counters are interpreted as byte sums per NVIDIA Nsight Compute metric definitions, aggregated across DRAM instances for the selected CUPTI range. Dividing their sum by the matched Forward Event duration gives effective range-average DRAM bandwidth; the result agrees within 0.7 percentage points with independent `dram__throughput...`/nominal 900 GB/s for all five workloads. This validates the stated arithmetic/physical interpretation for these captures, without implying a temporal peak. L2 throughput percent-of-peak is measured twice per workload. L2 bytes/s, issue/stalls and FP32/FP16 pipe activity remain unavailable.

## Measures
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` — three range observations per workload.
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — three range observations per workload.
- `sm__warps_active.avg.pct_of_peak_sustained_active` — three range observations per workload.
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` — three range observations per workload; measured zero on Decode Graph1.
- `dram__bytes_read.sum` / `dram__bytes_write.sum` — three raw range observations per workload; unit/aggregation interpretation withheld.
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — three range observations per workload.
- `lts__t_sector_hit_rate.pct` — three range observations per workload.
- `lts__throughput.avg.pct_of_peak_sustained_elapsed` — two range observations per workload.
- L2 bytes/s, FP32/FP16 pipe activity, issue/stalls, device time-series mean/peak — unavailable/not claimed. Physical effective range-average DRAM bandwidth is derived and cross-checked; temporal peak bandwidth is not measured.
