# Traces

`torch.profiler` (CUDA activities only), in-process `ModelRunner`, deliverable config
(graph ON, radix ON, `extra_buffer`, `mem_fraction_static=0.60`).

| File | Content |
|---|---|
| `prefill-bs1-64.json.gz` | 3 prefill calls, 1 request × 64 tokens (chrome trace) |
| `decode-bs8-1024.json.gz` | 10 decode steps, batch 8, ctx 1024 (chrome trace) |
| `prefill-bs1-64.txt`, `decode-bs8-1024.txt`, `decode-bs1-1024.txt` | `key_averages` kernel tables |
| `correctness-matrix-extra_buffer-graphon.json` | raw cold/warm ids + logprobs per prompt |

Open with `chrome://tracing` / Perfetto after `gunzip`.

## GPU busy vs wall (merged kernel intervals)

| Workload | Kernels | Wall span | GPU busy | GPU idle | Busy |
|---|---|---|---|---|---|
| prefill, 1 × 64 tokens × 3 calls | 2,322 | 195.7 ms | 56.3 ms | 139.4 ms | **28.8 %** |
| decode, bs 8 @ 1024 ctx × 10 steps | 8,140 | 123.0 ms | 106.2 ms | 16.8 ms | **86.3 %** |

`OBSERVED`: short-prompt prefill is CPU-dispatch bound (65 ms wall vs 18.8 ms GPU per call),
decode replay is GPU bound with ~1.7 ms/step of launch gaps.

Reproduce: `trace_export.py` (`PYTHONPATH=python python trace_export.py`), see `results.md`.
`nsys` is unusable in this environment (`qdstrm` importer missing; `ncu` aborts at scheduler init),
so no `.nsys-rep` is attached.
