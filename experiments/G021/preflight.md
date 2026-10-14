# G021 resource-blocked hold — read-only evidence and resume gate

Status: **BLOCKED_RESOURCE. No Falcon/model/NCU/GPU workload was launched under G021.** All telemetry below is host/device preflight, not decode evidence. RTX 2060 was not queried or profiled.

## Read-only host risk observation

The current unrelated job is PID 3903363 (`dev-neu+`, Python), parent PID 3903362, scanning `/mnt/data/neuron/stock/E001-cortex-data-001/canonical/us` Parquet files in 500k-row batches with 24 threads. Its command line explicitly performs cross-row de-duplication in a Python dictionary (`seen`), so RSS is data-dependent. It is not ours and was not signalled, throttled, paused, deleted or otherwise changed.

Observed during read-only preflight (not a stable capacity guarantee):

| Observation | Earlier | Later | Meaning |
|---|---:|---:|---|
| PID 3903363 RSS | ~4.3 GiB | ~11.7 GiB, later ~8.0–11.7 GiB across samples | Existing job materially varies; one free-memory point is not enough |
| Host `MemAvailable` | ~13.7 GiB | ~7.4 GiB; later ranged ~7.4–16.5 GiB | Can change substantially while task runs |
| Memory PSI | nonzero; `some/full avg60` reached ~0.2% | `avg60` later reached ~0.6% | Memory contention already observable |
| Swap | 46,137,332 kB total, ~41,069,156 kB free; zram 8 GiB with ~4.7 GiB used | small changes | Host is already using compressed swap; do not use swap as a headroom plan |
| `/mnt/data` | 99%, 10,123 MiB available | unchanged at last check | Almost-full volume; no broad staging/cache growth |

The present process value is a sampled condition, not a guarantee of its peak. No process limit or host resource policy was changed. Rerun the exact read-only commands in “Resume gate” when reconsidering.

## Model/service and profiler peak anchors already on disk

These are prior observations and bounds for planning only. They are **not new G021 measurements**.

1. Falcon checkpoint directory `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4` is 7.3 GiB on disk (two safetensors shards); physical file size is not equal to peak host RAM because the loader can mmap/stream, but cannot serve as a complete RSS estimate.
2. Accepted G019 S1 and candidate service launches report `mem_fraction_static=0.76`, graph sizes 1/2/4, S1 no profiling. G019 startup log measured CUDA free memory during graph capture: 5.20 GiB before bs4 capture; 5.02 GiB after bs1 for S1, and 5.20→5.14 GiB for c57. G018 repeated this range: S1 5.20→5.02 GiB, c57 5.20→5.14 GiB. Thus an already-qualified live Falcon server requires at least ~5.2 GiB free V100 memory at graph capture, plus service/KV working set; do not profile during graph capture.
3. The existing local runner `/mnt/data/pc01/sglang-v100-migration/logs/start-falcon-s1-local.sh` is a different recovered wrapper that uses `mem_fraction_static=0.88`; do not edit it under the hold. It is not the selected resume launcher. G019’s pinned exact worktree/base and mode artifacts should be reused for S1 (G019 worker script run) or the existing E026 `start_candidate_server.sh` explicitly with mode S1. Verify the S1 binary path/SHA and launch log before sending requests.
4. Prior NCU evidence is explicitly limited to `M=4` standalone SGLang GPTQ operator fixtures, not a Falcon-serving graph. E026 results report S1 named gate_up/down `gpu__time_duration` 288.736/185.472 µs (one exact M4 launch, NCU kernel replay; matched candidate numbers also exist). These validate that root NCU can collect metric reports on real V100 GPTQ code and give a kernel-duration scale. E026 NCU used a `systemd-run --scope -p MemoryMax=6G` per-invocation cap; no persistent host memory policy was changed. NCU has multiple hardware-counter passes and an intrusive replay boundary; do not add its metrics run on top of the normal service wave or treat its result as device/Graph duration.
5. Existing G019 traces already cover S1/c57 × 1K/4K × requested c1/c4 with three DECODE GPU annotation steps; archived 4K/c4 actual batches are bs3. They are prior execution-timeline evidence, have no Graph node IDs, and are not new NVML/Ncu resource measurements. Do not repeat trace profiling as G021 evidence.
6. Prior G017 startup failure: `/mnt/data/pc01/sglang-v100-migration/logs/g017-candidate-server-v8.log`, SHA-256 `c1e5e6ca9002e10d1fcf660bcb4030d4d081806e69d2f037c392f7d9e043bc99`, ends with shell `Killed` after CUDA Graph bs4/2/1 captures. Other later G017/G018/G019 services did reach readiness and complete bounded serving; this is evidence startup can fail under pressure, not proof current resources are safe.

## Metric/tool boundaries already qualified

- Device UUID: `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, V100-SXM2-16GB, physical `nvidia-smi` index 1. Root NCU must still be passed `--devices 0` only inside process `CUDA_VISIBLE_DEVICES` pinned by UUID; verify resulting device metadata matches V100.
- `nvidia-smi --query-gpu` supports timestamp, UUID, GPU/memory busy, memory used, current SM/memory clock, power, temperature and throttle supported/active fields. Existing G018 `gpu_sampler.py` is a target-UUID sampler; reuse it, do not create a second telemetry framework. Device samples are whole-device, not per-process.
- `/usr/local/cuda/bin/ncu`, root-only invocation as needed, 2025.1.1.0; `sudo -n ncu --query-metrics --chip GV100` succeeds. NCU help says `--graph-profiling=graph` profiles entire graphs, but that exact real-service graph aggregation remains untested; do not claim it qualified until an unblocked bounded run. No global profiling state changed.
- DCGM is absent: `command -v dcgmi` and `nv-hostengine` no paths; package query had no DCGM packages; system service is inactive; no install/start done.

## Auditable safe resume gate (all conditions required)

This gate is intentionally conservative because the co-running job is unbounded and has shown significant RSS variance. Resume only after the job exits naturally **or** after separate, stable capacity evidence demonstrates all conditions:

1. PID 3903363 and its owner are no longer running (or owner provides a separately authorized capacity window). Do not stop it. If it remains alive, a free-memory snapshot alone does not satisfy this condition.
2. Over at least 30 minutes of read-only samples at 1-minute intervals: minimum `MemAvailable` >= **20 GiB**, no increasing `some/full` memory-PSI trend (10/60/300s fields recorded; require 10-minute full pressure to return to 0.00% on each final sample), and `SwapFree` remains >= **36 GiB** (existing zram usage is recorded separately). This reserves >7 GiB above the ~12 GiB minimum estimated by the previous preflight/model baseline and absorbs normal fluctuation; after the job exits, available memory must actually meet this floor.
3. `/mnt/data` must have >= **25 GiB** free and <97% used before starting. The G021 artifact budget is capped at 5 GiB; don't copy model weights or write profiler temporaries to this filesystem. Existing 10 GiB free is insufficiently resilient at 99% volume usage.
4. The target V100 must remain idle at 7 MiB-class usage before model start; no unrelated process may be occupying the target. No inference about or access to RTX 2060.
5. After S1 server startup and before profiling, host MemAvailable must remain >= **12 GiB**, the SGLang worker RSS and V100 memory after graph capture must be recorded, CUDA free memory at graph capture >= **5.2 GiB**, and no memory OOM/kill event or rising pressure. If any lower bound fails, stop this G021 attempt with normal server shutdown and do not start NCU.
6. NCU is a separate, authorized profile condition: ensure host MemAvailable remains >=12 GiB before attaching; the server is already fully initialized; target UUID is isolated; scope to one selected real graph/kernel launch/condition with a known finite launch count; no broad application replay or all-kernel NCU. If actual attached NCU makes service unresponsive, reaches pressure floor, touches an unexpected GPU or needs an unapproved profiler architecture, stop, preserve evidence and return BLOCKED_RESOURCE.

The thresholds are admission gates, not a claim that model loading consumes exactly 8 GiB. They exceed the observed 7.3 GiB on-disk model size plus Python/runtime/capture overhead and allow >7 GiB for unmodeled growth. Recheck as a time series after the unrelated job is finished; never reclaim capacity by killing the job, evicting cache, changing swap, cgroups, systemd OOM policy or GPU clock/power settings.

## Pre-prepared R17 collection format (no profiling now)

`experiments/G021/r17-resume-config.json` records exact S1 SHA, target UUID, 1K/4K × c1/c4, 256 greedy outputs, existing graph sizes 1/2/4, existing sampler and metric boundaries. Existing harness `experiments/E026-local-v100-s4-m3c/scripts/service_baseline_v2.py` defines prompt shape, output count, concurrency and repeats; existing E027 `gpu_sampler.py` provides NVML columns. Reuse those as accepted infrastructure; no new profiler/harness framework has been created.

When gates pass:

- Service authority: NCU off, S1 exact SHA, NVML sampler concurrently target-UUID only, one 1K/4K × c1/c4 condition per window, three repeats; report request TTFT/TPOT/ITL and E2E separately. This is the only timing used for ordinary service performance.
- NCU hardware metrics: separate, bounded profile conditions, graph aggregate mode only if verified for captured graph, exact V100 metadata, first confirm each selected actual Falcon kernel/graph range and repeat/counter pass behavior. NCU instrumented performance is not service authority.
- Output shape is predeclared in `r17-resume-config.json`: target-only `nvml-<condition>-<repeat>.csv`, NCU `.ncu-rep` plus exported CSV, service JSON and a manifest with source SHA, run/capture times and condition IDs. Keep whole-device NVML percentages in their native sampling window; per-kernel/Graph bytes and duration only yield boundary-matched GB/s. Don't label a kernel or Graph metric “whole-device utilization.”
- After the hardware matrix resolves, continue requested Cold/Warm cache, concurrency, TTFT/TPOT/ITL and E2E under separately stated conditions; do not repeat G018/G020 work.

## Hashes of historical planning sources

- `g019-server-s1.log` SHA-256 `e42d9be2f104e0858ffde82d5afeaf6daa9e24c8d752ae1627722bbcefc4797e`
- `g019-server-c57.log` SHA-256 `8ae22a062e2aa4dd05df35e81146377eb3373fe06f96ba84ee19a276bc522295`
- prior G017 failed start log SHA-256 above
- E026 `results.md` SHA-256 `6dcbca6b8098aede77b0faf7e38f03f2cbdb7c3d81757b3e6a4dcfb538cb96be`
- G017 final S1 M4 gate_up/down NCU report hashes and candidate counterpart hashes are recorded in the experiment manifest (source evidence only; no report copied).

## Bounded outcome

Host/device query fields and root NCU metric database are available; DCGM is uninstalled/inactive. No G021 workload was run. Resource hold remains until the auditable resume gate passes. None of this preflight/fixture data counts as Falcon busy %, SM/Tensor/DRAM activity or HBM bandwidth. G021 remains BLOCKED_RESOURCE, not ACCEPTED.
