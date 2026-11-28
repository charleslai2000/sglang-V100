# E049 — results

All numbers measured on the merged tree (`v100-harness` @ forward-merge commit),
LFM3 fp16, `--attention-backend triton --sampling-backend pytorch
--mamba-radix-cache-strategy extra_buffer` (CUDA graph + radix cache on).

## 1. Correctness anchors unchanged

| Check | E046/E047 record | After the merge |
|---|---|---|
| 8K prompt vs HF fp16 oracle | gen logprob diff 4.77e-07, ids equal | **4.7676439862698317e-07**, ids equal |
| E046 matrix (graph ON, 6 prompts) | ids equal, drift ≤7.6e-04 | ids equal everywhere; drift 6.0e-04 (p1024), 8.0e-04 (pA), 4.9e-06 (p4096) |
| cache hits in that matrix | 256/512, 1024/3072, 512/512 | identical |
| E047 width-7 conv (5 varlen cases) | conv state bit-exact | bit-exact; output ≤3.9e-03 (fp16 rounding of the FP32 reference) |
| E047 fused decode (6 cases) | 0.0 except padded bs32 2.44e-04 | identical, state bit-exact |
| E047 edge set (len 1–13, chunked, prefix hit, graph off) | ids fixed | reproduced |

So the newer runtime reproduces the LFM3 numerics bit-for-bit at the kernel
level and within fp16 noise end to end.

## 2. `/v1/score` and `/v1/decisions` (now official code)

| Question | Server | Independent HF oracle | max abs diff |
|---|---|---|---|
| yes_no | 0.999913 / 8.7e-05 | 0.999914 / 8.6e-05 | 1.0e-06 |
| choice | 0.999920 / 7.7e-05 / 3.0e-06 | 0.999920 / 7.7e-05 / 2.0e-06 | 2.4e-07 |
| score (3 levels) | 0.476839 / 0.507592 / 0.015569 | 0.503930 / 0.480853 / 0.015217 | 2.7e-02 |

`label_mass` agrees to 5.2e-06 / 3.0e-04 / 1.9e-04, so the underlying label
logprobs match; the `score` question is a near-tie (oracle levels 0.504 vs 0.481)
that fp16 accumulation is free to reorder. `/v1/score` raw label probabilities
are byte-identical to the pre-merge run (4.745e-05 / 3.93e-07).

## 3. Hot-prefix incremental extend (1024-token cached prefix)

| New tokens | Cold TTFT | Hot TTFT | Speedup | Cached tokens |
|---|---|---|---|---|
| 5 | 313.0 ms | 78.1 ms | 4.01x | 1024 |
| 8 | 313.4 ms | 77.7 ms | 4.03x | 1024 |
| 10 | 313.6 ms | 78.5 ms | 4.00x | 1024 |

Cold extend is ~5 % slower than before the merge, hot extend ~4 % faster, and
the cache hit is still the full 1024-token prefix.

## 4. E047 HTTP matrix, warm

| Input | Concurrency | E047 tok/s | Now tok/s | Delta | E047 TTFT | Now TTFT |
|---|---|---|---|---|---|---|
| 64 | 1 | 121.44 | 140.51 | +15.7 % | 64.8 | 65.3 |
| 64 | 8 | 827.15 | 909.71 | +10.0 % | 121.8 | 206.3 |
| 64 | 16 | 1638.27 | 1810.84 | +10.5 % | 143.9 | 145.4 |
| 1024 | 1 | 111.05 | 122.47 | +10.3 % | 271.6 | 287.7 |
| 1024 | 8 | 511.74 | 592.43 | +15.8 % | 998.0 | 1074.8 |
| 1024 | 16 | 981.93 | 1081.51 | +10.1 % | 1054.4 | 1041.3 |
| 4096 | 1 | 37.58 | 38.64 | +2.8 % | 2306.8 | 2335.4 |
| 4096 | 8 | 156.72 | 175.30 | +11.9 % | 3992.0 | 3965.1 |
| 4096 | 16 | 350.93 | 385.08 | +9.7 % | 2872.4 | 2719.9 |

Every cell improves. The one apparent regression seen in the first sweep
(64/1 at −9.0 %, TTFT 132 ms) was cold-start: the merged tree dropped the fork's
SM70 prefill warmup and upstream compiles its Triton attention kernel on first
use, so the first cell of a fresh process pays a one-time cost. Four warm
repetitions give 140.5 tok/s and 65 ms TTFT, i.e. +15.7 %.

TTFT at 64/8 rises to 206 ms while throughput improves 10 %: the newer scheduler
trades first-token latency for batching in that cell.

## 5. Tests on the merged tree

| Suite | Result |
|---|---|
| `unit/mem_cache/test_registry.py` + `unit/configs/test_model_config_parser_registry.py` | 23 passed — the E046 hybrid routing test passes against upstream's `create_unified_radix_cache` |
| `layers/mamba/test_causal_conv1d.py` | 232 passed, 14 failed — the same pre-existing failures (the test's own `F.conv1d` reference cannot run on this GPU), unchanged from E047 |
| whole `unit/mem_cache` + `unit/configs` tree | not usable as a gate in this environment: new upstream tests need `nixl`, `mori`, `parameterized` and a newer `sgl_kernel.kvcacheio`, and the collection aborts (4 import errors, 1 core dump in a rust-tree test) |

Weight-free construction still works for the non-LFM3 hybrid/most-common models:
`GemmaForCausalLM` 744.5 M, `Gemma2ForCausalLM` 745.6 M, `FalconH1ForCausalLM`
173.6 M parameters, all built on the meta device (script:
`experiments/E048-p1-version-integration/compat_without_weights.py`, updated for
the new `runtime_context` parallel API).
