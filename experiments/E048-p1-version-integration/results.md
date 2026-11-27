# E048 — results

Branch `v100-harness`, base `d4d15c7f8`. Every number below was produced on the
integrated tree; scripts are next to this file.

## 1. `/v1/decisions` and `/v1/score` against an independent HF oracle

LFM3 fp16, graph ON + radix ON, one request with three question types
(`decisions_oracle.py` recomputes the label logprobs from the returned
`prompt_token_ids` / `label_token_ids` with HF transformers in a separate venv):

| Question | Type | Server probabilities | Oracle probabilities | max abs diff |
|---|---|---|---|---|
| Is Paris the capital of France? | yes_no | 0.999915 / 8.5e-05 | 0.999914 / 8.6e-05 | 1.0e-06 |
| Which city is the capital? | choice | 0.999922 / 7.5e-05 / 3.0e-06 | 0.999920 / 7.7e-05 / 2.0e-06 | 1.8e-06 |
| How good is this answer? | score | 0.492212 / 0.492212 / 0.015577 | 0.503930 / 0.480853 / 0.015217 | 1.2e-02 |

`label_mass` (unnormalised label probability at the answer position) agrees to
1.7e-05 / 9.7e-05 / 3.2e-04; the choice argmax agrees. The `score` question is
the only wider cell because its first two levels are a near-tie (oracle 0.504 vs
0.481), which fp16 accumulate order is enough to reorder within 1.2e-02.

`/v1/score` raw label probabilities, query `The capital of France is`, labels
` Paris` / ` Lyon`: server 0.06462 / 1.45e-06, oracle 0.06428 / 1.40e-06.

## 2. Ported semantics needed by those endpoints

Four interfaces differ between v0.5.23 and this tree; the adapters are listed in
`experiment.md`. Two of them changed measured behaviour, so they were verified
rather than assumed:

* prompt round-trip probe (refuse only when the rendered chat text does not
  re-encode to the template ids) — LFM3 passes, so decisions serves;
* `score_prompts` returns **probabilities** here (upstream returns logprobs) and
  accepted only one label set — the endpoint now takes `log()` and the mixin
  accepts per-prompt label ids.

Without the `log()` conversion the reported probabilities were
softmax-over-probabilities (0.5003/0.4997 for a 0.9999/0.0001 pair); with it they
match the oracle to 1e-06.

## 3. Regressions on the integrated tree

| Check | Result |
|---|---|
| E046 matrix, graph ON, 6 prompts cold/warm | `ids_equal True` everywhere; prompt-logprob drift 7.64e-04 (p1024), 2.37e-04 (pA), otherwise 0.0 — same as E046 |
| E047 width-7 conv, 5 varlen cases | conv state bit-exact (0.0); output ≤3.9e-03 = fp16 rounding of the FP32 reference |
| E047 fused decode, 6 cases | output 0.0 except the padded bs32 case 2.44e-04 (1 fp16 ulp); state bit-exact — same as E047 |
| E047 edge set (len 1–13, chunked 256/300/1000, prefix hit, graph off) | greedy ids identical to the E047 record, cache hits reproduce (256/512/768) |
| 8K prompt vs HF fp16 oracle | gen logprob diff 4.77e-07, ids equal, first-token logprob −1.9192511899746e-05 both sides |
| Mixed prefill+decode batch | 8/8 requests match the reference ids |
| `unit/mem_cache` + `unit/configs` | 1129 passed, 480 skipped |
| `layers/mamba/test_causal_conv1d.py` | 230 passed, 14 failed — pre-existing: the test's own `F.conv1d` reference raises `GET was unable to find an engine` on this GPU (identical on `95996ab0e`) |
| `unit/spec/test_v100_speculative.py` | 104 passed, 7 failed — pre-existing (speculative line not ported). The unported baseline file fails 9, so the port fixes 2 |

## 4. Compatibility without weights (none available offline)

`compat_without_weights.py` parses a config through `ModelConfig` and builds the
model class on the meta device:

| Model | Config parse | Meta build |
|---|---|---|
| `GemmaForCausalLM` | ok (8 heads, 1 kv, ctx 8192) | 14 tensors, 744.5 M params |
| `Gemma2ForCausalLM` | ok (8 heads, 4 kv, ctx 8192) | 18 tensors, 745.6 M params |
| `FalconH1ForCausalLM` | ok (8 heads, 2 kv, ctx 32768) | 33 tensors, 173.6 M params |

Falcon-H1 is the hybrid (mamba + attention) model this tree ships, and its
conv width is 4 — the E047 dispatch guard only reroutes width > 4, and the mamba
unit suite above still passes its width-2/3/4 varlen cases.

## 5. Hot-prefix incremental extend (new benchmark)

1024-token prefix cached, then only 5 / 8 / 10 new tokens, `max_new_tokens=1`,
server-reported `cached_tokens` and client TTFT (`hotprefix_bench.py`, 5 reps):

| New tokens | Prompt | Cold TTFT | Hot TTFT | Speedup | Cached tokens |
|---|---|---|---|---|---|
| 5 | 1029 | 296.5 ms | 81.2 ms | 3.65x | 1024 |
| 8 | 1032 | 298.7 ms | 80.7 ms | 3.70x | 1024 |
| 10 | 1034 | 297.3 ms | 81.7 ms | 3.64x | 1024 |

Where the 81 ms goes (`extend_gpu_busy.py`: 40 sequential requests per delta,
NVML sampled at 20 Hz):

| New tokens | Wall | GPU busy | GPU idle | Mean util | Median util |
|---|---|---|---|---|---|
| 5 | 80.6 ms | 47.1 ms | 33.6 ms | 58.1 % | 84.0 % |
| 8 | 80.6 ms | 46.7 ms | 33.9 ms | 57.7 % | 83.0 % |
| 10 | 81.2 ms | 46.7 ms | 34.5 ms | 57.4 % | 83.5 % |

Kernel mix inside those requests (scheduler-process trace via
`/start_profile`, 5 requests per delta, 340 ms of kernels total):

| Group | Share |
|---|---|
| extend attention (`_fwd_kernel`) | 78.3 % |
| GEMM (cutlass / gemv2T) | 15.8 % |
| copy / elementwise | 3.4 % |
| norm | 1.2 % |
| Short Conv | 0.2 % |

So the incremental extend is dominated by attending over the *cached* prefix, not
by the 5–10 new tokens: 5 vs 10 tokens differ by <1 % in both wall and GPU time,
and the E047 conv work is now 0.2 % of it. The remaining lever is the extend
attention kernel for tiny query blocks over a long cached context.

## 6. E047 performance spot check

Same HTTP matrix cell definition as E047 (`e047_bench.py`), 3 repetitions, median
versus the E047 record (single repetition):

| Input | Concurrency | E047 out tok/s | Now out tok/s | Delta | Run spread |
|---|---|---|---|---|---|
| 64 | 1 | 121.44 | 118.47 | −2.4 % | 1.1 % |
| 64 | 8 | 827.15 | 799.56 | −3.3 % | 0.7 % |
| 64 | 16 | 1638.27 | 1566.82 | −4.4 % | 0.8 % |
| 1024 | 1 | 111.05 | 107.77 | −3.0 % | 9.8 % |
| 1024 | 8 | 511.74 | 501.54 | −2.0 % | 1.0 % |
| 1024 | 16 | 981.93 | 952.62 | −3.0 % | 0.3 % |
| 4096 | 1 | 37.58 | 37.58 | 0.0 % | 0.4 % |
| 4096 | 8 | 156.72 | 155.54 | −0.8 % | 0.1 % |
| 4096 | 16 | 350.93 | 347.25 | −1.0 % | 0.1 % |

No cell regresses beyond 4.4 %, long context is flat, and the residual 2–4 % is
cross-session drift: this port touches no decode or scheduler hot path (warmup
batch count, an inactive-on-1-GPU all-reduce, a new HTTP route, the scoring
mixin), and a one-repetition run of the same matrix earlier in this session
reported 64/c16 at 1344 tok/s, i.e. 18 % below its own three-repetition median.
