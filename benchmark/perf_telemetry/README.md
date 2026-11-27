# GPU performance telemetry toolkit

Reusable measurement assets salvaged from the G019–G022 / E030 evidence series and
extended with operator-level attribution. Metric definitions and boundaries follow
[`design/LLM-Inference-Performance-Metrics-V1.md`](../../design/LLM-Inference-Performance-Metrics-V1.md):
per-kernel, per-forward and whole-device boundaries are kept distinct, and NVML
utilization is never reported as SM/HBM peak.

Nothing here changes model or kernel behaviour; all measurement is read-only.

## Components

| File | Role | Provenance |
|---|---|---|
| `capture_forward.py` | Runs real SGLang prefill/decode forwards in-process and exports a Kineto chrome trace with one `record_function` span per operator (installed via forward hooks, no model edit). | new, reuses `sglang.bench_one_batch` loaders |
| `op_budget.py` | Attributes every kernel to its operator through Kineto launch correlation, then reports kernel counts/time, share, GPU idle gaps and host dispatch per operator. | new; replaces kernel-name guessing |
| `profiles.json` | Hook patterns + owner rules per model profile (`lfm3`). | new |
| `forward_timeline.py` | CPU/GPU interval union, per-forward GPU gaps and FPM-sidecar clock-sync join for G022-style evidence. Its owner classifier targets Falcon GPTQ, kept for historical reproduction. | `experiments/G020/analyze_forward_timeline.py`, unchanged |
| `gpu_sampler.py` | Target-UUID-only `nvidia-smi` time series (utilization, clocks, power). | `experiments/G021/gpu_sampler.py`, unchanged |

Related in-tree instrumentation that this toolkit composes with (do not rebuild):
`python/sglang/srt/utils/device_timer.py` (`DeviceTimer` / `GapTimer`, CUDA-event
brackets used by the FPM metrics reporter) and
`python/sglang/srt/managers/scheduler_components/metrics_reporter.py`.

## Dependencies

- Python 3.12, PyTorch with CUDA (Kineto); Triton for the SGLang kernels.
- `nvidia-smi` for `gpu_sampler.py` (no DCGM, no root required).
- `PYTHONPATH=python` from the repository root so `sglang` and `sglang.bench_one_batch` import.
- No external profiler (`nsys`/`ncu`) is required. On this host `nsys` cannot import
  its `qdstrm` converter and `ncu --target-processes all` aborts the scheduler at
  init, so neither is part of the default path.

## Minimal run

```bash
cd <repo>
PYTHONPATH=python python benchmark/perf_telemetry/capture_forward.py \
    --model-path /var/tmp/lfm3 --profile lfm3 --mode decode \
    --batch-size 8 --input-len 1024 --steps 10 --out-dir /tmp/lfm3_trace

PYTHONPATH=python python benchmark/perf_telemetry/op_budget.py \
    /tmp/lfm3_trace/trace-decode-bs8-ctx1024.json.gz \
    --profile lfm3 --out /tmp/lfm3_trace/budget-decode.json
```

`capture_forward.py` prints the trace path; `op_budget.py` prints the per-owner
table and writes the full JSON (including the operator breakdown and the largest
GPU idle gaps).

## Measurement contract

- `envelope`: first to last kernel of the captured region.
- `activity_union_us`: union of kernel intervals = GPU busy; `gpu_idle_us` is the
  remainder inside the envelope (launch gaps, host stalls) and is reported
  separately, never as kernel time.
- `kernel_share_pct` uses the kernel-duration sum; `envelope_share_pct` includes
  idle, so both are shown.
- `host_ms` is CPU time inside the operator's own span; on CUDA-graph replay the
  graph launch collapses these spans, so host numbers are only meaningful for
  eager paths (prefill, graph capture).
- Hardware counters (SM/Tensor/HBM/L2) are **not** produced here. On this host the
  only validated route is CUPTI Range Profiling as used in E030, which must run
  without any concurrent Kineto/NCU profiler and is therefore measured in a
  separate run; see `experiments/E047-lfm3-width7-conv/` for that evidence.

## Notes

- `capture_forward.py` calls `os._exit(0)` after writing artifacts: profiler
  teardown on SM70 can segfault after a successful capture.
- Prefill capture reuses `prepare_synthetic_inputs_for_latency_test`, which does
  not release request slots, so keep `--steps` small for prefill.
