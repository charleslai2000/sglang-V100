# Results — E021 S3 V100-only optimization

## Scope and fixed references

- **V100 only.** Frozen S1 source/baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e`; checkpoint revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; the isolated S1 environment on `gpushare-v100`. RTX 3090 is historical reference only; no rebuild/run/profile/requalification was done.
- Fixed RTX 3090 reference: decode-heavy c1/c4/c8/c12/c24 = 64.8/206.6/371.2/485.7/495.1 tok/s; agent-like c8/c16/c24 = 224.6/352.2/396.3; steady B16 = 459.53 tok/s. These are not fresh comparison runs.
- S2 graph-enabled c8/1K/64 measured ~75.5 aggregate tok/s. E021 c8/1K/256 measured ~100.0 in three early correctness-valid runs; this different output length amortizes TTFT and is not an optimization comparison.

## Measured configuration optimization: graph capture coverage for batch 8

All rows below use the same frozen V100 S1 source, checkpoint, dtype/backends, 1K prompt, 256 generated tokens, concurrency 8, disabled radix/piecewise graph, and the same request harness; all output/correctness checks passed. Each rate is the harness's aggregate generated tokens/s.

| graph config | max running | measured tok/s, 3 repeats | mean |
|---|---:|---:|---:|
| `[1,2,4]` (capture max 4) | 4 (initial reference only) | 100.06, 100.55, 100.54 | 100.38 |
| `[1,2,4]` (capture max 4) | 8 | 74.62, 76.72, 75.81 | 75.72 |
| `[1,2,4,8]` (capture max 8) | 8 | 153.01, 154.45, 154.54 | 154.00 |

The **causal paired comparison is the latter two rows**: `max_running_requests=8` is fixed and the captured graph set is the only configuration change. With metrics enabled, SGLang `sglang:cuda_graph_passes_total` delta was decode graph/eager = 0/256, 0/256, 0/256 for graph4 and 255/0, 256/0, 256/0 for graph8. Prompt passes remained eager (4–5 `prefill_none` each). The measured mean improvement is **+103.4%** (154.00 vs 75.72 tok/s). This establishes that the workload's batch-8 decode was falling back to eager when the largest graph was 4; adding the exact batch-8 graph removed that fallback and improved throughput. The initial max-running=4 to max-running=8 comparison (+53.4%) changed both concurrency ceiling and graph coverage, so it is not used to attribute the optimization.

SGLang counter units are forward passes, not token-weighted scheduler histograms; report the concrete per-run deltas above rather than extrapolating them as universal graph coverage. The graph4-vs-graph8 result applies to this sustained c8 workload. No direct Nsight timeline attribution was obtained; this is a framework-instrumented graph/eager attribution with unprofiled throughput. It is software/configuration optimization evidence, not an Nsight-profiled kernel change.

Graph capture resource delta from startup logs: graph4 consumed 0.21 GB, graph8 0.22 GB; available memory after capture 20.46 vs 20.44 GB. Both initialized Mamba pool for 8 logical requests: 1.16 GB SSM plus 0.01 GB conv in rounded logs. Graph capture list observed `[1,2,4]` and `[1,2,4,8]`, respectively.

Raw requests on V100 under `/hy-tmp/sglang-logs/`:
- `s3-v100-graph4-before.jsonl` SHA256 `88102b040e894f340bb775bc6e7d856133db832115bbeccd6465d38e6137d08b`.
- `s3-v100-graph4-r8-paired.jsonl` SHA256 `af139abbd91609b64224c0f7f717935cb114bf59cdc9503aa1a345d374d09a9b`.
- `s3-v100-graph8-after.jsonl` SHA256 `94ae9f17d14f9f5b64e2fc7c49375b2888d2c0a1c2726ed982fdbd5ad7d8a346`.
- `s3-v100-graph8-r8-paired.jsonl` SHA256 `632339c8f061d7eb540febb27a8998ad316f3ddae123592039740ecd71283644`.
- Paired server logs: `s3-v100-graph4-r8-metrics.log`, `s3-v100-graph8-r8-metrics.log`; A/B service was stopped after capture; final GPU check 1 MiB, 0% utilization.

## Mamba state accounting correction (source-derived, checked against S2 logs)

S1 config: 44 linear-attention layers; intermediate 12288; groups 1; conv kernel 4; heads 24; head dim 128; state size 256. SSM temporal state defaults FP32; SM70 convolution state is FP16.

- Temporal bytes per slot: `44*24*128*256*4 = 138,412,032 B`.
- Conv bytes per slot: `44*(12288+2*1*256)*(4-1)*2 = 3,379,200 B`.
- Total incremental state per **logical** request slot: **141,791,232 B = 0.1320 GiB**.
- `ReqToTokenPool` reserves `size+1` physical rows; Mamba cache uses a padded dummy slot for graph batches as well. For S=1/4/8 logical slots, physical tensor totals are `(S+1)*141,791,232` = 0.2640/0.6604/1.1880 GiB, matching S2 pool logs rounded as SSM 0.26/0.64/1.16 GB and conv 0/0/0.01 GB (decimal/rounding explains display).

Correction: S2's “0.256GB/request” was not the marginal per-request increment; it conflated temporal-only allocation including the padding slot with request count. Marginal state allocation is 0.1320 GiB per logical request, plus one fixed 0.1320 GiB dummy slot. In the S=8 graph comparison, Mamba state was fixed at the same pool capacity; graph8 capture itself added only ~0.01 GB vs graph4.

## Profiling and prefill status

- Nsight Systems 2024.6.2 and Nsight Compute 2025.1.1 are installed on V100. Systems GPU hardware metrics unsupported on this V100; NCU metric query returns `ERR_NVGPUCTRPERM`; CPU sampling denied by `perf_event_open` restrictions.
- Client-only Nsight Systems capture did not include server CUDA activity (server outside captured process tree), so invalid for GPU ownership.
- Capturing a fresh server in the Nsight process tree with `--trace=cuda,nvtx,osrt --cuda-graph-trace=graph` caused scheduler SIGSEGV during warmup at SGLang `kvcache.store_cache` (scheduler exit -11). Ordinary unprofiled S1 service passed correctness. Trace had no usable kernel table; rejected as invalid. Do not infer an S1 defect or kernel timings.
- Mature prefill graph support was not A/B qualified for Falcon. Prefill remained eager in graph coverage counters; piecewise graphs are disabled per S2 capture failure. No prefill performance claim made.
- No kernel-owner timeline, idle-gap/synchronization quantification, targeted kernel counters, or hardware/software ceiling decomposition is established. The measured graph-coverage change is the one evidence-backed optimization; it was selected from source/API fallback logic and pass counters, not from an Nsight dominant-kernel ranking.

## Conditional V100-16GB estimate (not physical qualification)

S2's recommended configuration remains an accounting-based conditional candidate: 24,576 total tokens, default 4 running requests, standard graphs for 1/2/4, piecewise graphs off, and at least 2 GiB free headroom. The new c8 experiment proves a specific sustained-c8 decode setting can gain substantially by reserving the batch-8 graph, but both the A/B services ran on the 32GB V100 and held an eight-request Mamba pool (about 1.19 GiB exact physical state). It does **not** establish that c8 plus graph8 fits or is desirable on 16GB. Incremental state for 8 rather than 4 logical slots is only 4*0.1320 = 0.5281 GiB; graph capture extra measured ~0.01 GB, but model/KV/cache pressure and runtime reserve remain dominant and no 16GB device/cap was available.

Therefore retain the S2 16GB candidate unchanged pending actual 16GB qualification; explicitly budget 0.1320 GiB per additional logical Mamba slot and one fixed dummy slot. The measured +103.4% sustained c8 result is not extrapolated to c4/production mixture or 16GB.

## Milestone disposition

**PASS for the bounded V100-only configuration optimization and Mamba accounting correction; overall profiling/characterization scope remains incomplete.** The requested Systems-first owner attribution followed by NCU, host/runtime gap analysis, mature prefill-graph A/B, and full software-vs-hardware decomposition did not complete because the valid capture path segfaults and device counters are denied. Those gaps are explicitly retained; they do not invalidate the independently measured graph4-vs-graph8 correctness and throughput result. No full dominant-kernel explanation or universal V100 ceiling is claimed.
