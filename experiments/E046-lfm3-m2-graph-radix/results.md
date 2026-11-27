# E046 — LFM3 M2 results: CUDA Graph + Radix Cache

All numbers observed on Tesla V100-SXM2-16GB (SM70), fp16, single card, SGLang `main` worktree,
flags as in `experiment.md`. `OBSERVED` = measured, `INFERRED` = reasoned from measurements.

## 1. Correctness

### 1.1 Cache strategy × graph (`/var/tmp/m2_validate.py`, in-process `sgl.Engine`)

Prompts: 64 / 256 / 1024 / 4096 tokens plus `pA`,`pB` sharing a 600-token prefix. `cold` = first
request (prefix cache empty), `warm` = repeat request (prefix hit). 8 greedy tokens, prompt
logprobs for the last 9 positions.

| Comparison | Greedy ids | prompt logprob max Δ | gen logprob max Δ |
|---|---|---|---|
| `extra_buffer` graph ON vs OFF, 6 prompts × {cold, warm} | all equal | **0.0** (bit-identical) | **0.0** |
| cold (prefill) vs warm (prefix hit), graph ON | all equal | ≤ 7.6e-4 | ≤ 1.9e-4 |
| `no_buffer` vs `extra_buffer`, graph ON | all equal | ≤ 3.5e-3 | ≤ 2.1e-4 |

Cold/warm differences come only from a different cache split point (cold hit 256/1024 tokens,
warm 512/3072), i.e. a different fp16 accumulation order in prefill — not a state bug.
`no_buffer` vs `extra_buffer` differs the same way.

### 1.2 Long context (8K, radix on, graph on)

Prompt = 4096-token block ×2 = 8192 tokens, `logprob_start_len = T-9` (no full-prompt logits).

| | Value |
|---|---|
| SGLang greedy ids | `[278, 8659, 16672, 3434, 20, 902, 1818, 607]` |
| HF fp16 oracle ids | identical |
| generated logprob max Δ | **4.8e-7** |
| last prompt logprob | −4.89e-6 (matches oracle) |

### 1.3 Concurrency and state-slot reuse

- 8 concurrent HTTP requests (`pA`×3, `pB`×3, `p1024`, `p4096`): 8/8 first-8 ids equal to the cold
  reference; shared-prefix requests hit the cache (`cached_tokens` = 256/512/1024) and stayed exact.
- Forced **mixed prefill+decode batch**: 4 decoders in flight, a 4096-token prefill submitted
  mid-decode (server log: `Prefill batch, #new-seq: 3 … #running-req: 1`), then a second wave of
  requests reusing slots: 8/8 responses matched the cold reference. This path exercised the
  `layer_cache` scope bug fixed in this milestone.
- Unit tests: `test/registered/unit/mem_cache/` + `test/registered/unit/configs/` →
  **1128 passed, 480 skipped** (includes the new `UnifiedRadixCache` routing test).

`OBSERVED`: graph replay and radix prefix hits are numerically transparent; conv state is restored
per cache boundary, not carried stale.

## 2. Performance matrix

HTTP streaming client, prefix cache flushed before every combo (`/flush_cache`), 30 s ceiling,
`out` tokens fixed. `TTFT`/`TPOT` from client-side token arrival times. Server ran the deliverable
config (graph ON, radix ON, `extra_buffer`).

| in | out | c | wall s | out tok/s | TTFT mean ms | TTFT p95 ms | TPOT ms | req/s |
|---|---|---|---|---|---|---|---|---|
| 64 | 256 | 1 | 2.64 | 96.9 | 168 | 168 | 9.70 | 97.0 |
| 64 | 256 | 4 | 3.03 | 338.6 | 333 | 335 | 10.54 | 84.8 |
| 64 | 256 | 8 | 3.14 | 651.6 | 336 | 342 | 10.97 | 81.7 |
| 64 | 256 | 16 | 3.25 | 1259.4 | 426 | 440 | 10.99 | 79.3 |
| 64 | 256 | 32 | 3.21 | 2554.0 | 430 | 461 | 10.70 | 81.1 |
| 1024 | 256 | 1 | 2.41 | 106.0 | 287 | 287 | 8.34 | 106.1 |
| 1024 | 256 | 4 | 3.85 | 265.9 | 1137 | 1251 | 10.63 | 66.5 |
| 1024 | 256 | 8 | 3.72 | 550.0 | 767 | 773 | 11.56 | 68.9 |
| 1024 | 256 | 16 | 4.42 | 926.1 | 1423 | 1469 | 11.68 | 58.2 |
| 1024 | 256 | 32 | 6.38 | 1283.2 | 2335 | 2879 | 15.71 | 40.4 |
| 4096 | 128 | 1 | 3.43 | 37.3 | 2282 | 2282 | 9.02 | 37.3 |
| 4096 | 128 | 4 | 5.25 | 97.6 | 3487 | 3619 | 13.80 | 24.4 |
| 4096 | 128 | 8 | 4.07 | 251.7 | 1937 | 1945 | 16.71 | 31.5 |
| 4096 | 128 | 16 | 6.18 | 331.2 | 3632 | 4005 | 19.91 | 20.8 |
| 4096 | 128 | 32 | 10.45 | 391.8 | 6137 | 7595 | 33.62 | 12.3 |

`OBSERVED`: at c≥8 every decode batch is a graph replay (`cuda graph: True`, 96/96 batches);
`#running-req` tracked the client concurrency (padding to the captured buckets 1/2/4/8/12/16/24/32).
With a warm prefix cache the 4096-token row drops from 2282 ms to 297 ms TTFT (c1) — i.e. the
`TTFT` column above is the cold-prefill cost.

### 2.1 GPU-side latency (in-process, CUDA events)

| bs | ctx | prefill ms | decode step ms | decode tok/s |
|---|---|---|---|---|
| 1 | 64 | 55.7 | 8.64 | 115.8 |
| 1 | 1024 | 285.5 | 8.29 | 120.7 |
| 1 | 4096 | 2263.7 | 8.96 | 111.6 |
| 8 | 1024 | 1918 | 11.17 | 716 |
| 16 | 1024 | 3799 | 11.37 | 1407 |
| 32 | 1024 | 7953 | 13.14 | 2435 |
| 8 | 4096 | 17617 | 16.68 | 480 |

Prefill scales linearly in tokens up to 1024 (0.28 ms/token) and super-linearly at 4096
(0.55 ms/token, quadratic extend attention). Decode step grows 8.3 → 13.1 ms from bs 1 to 32,
i.e. 120 → 2435 tok/s, ~20× for 32× the batch — the step is weight-bandwidth dominated.

## 3. Memory (mem_fraction_static 0.60, cuda_graph_max_bs 32)

| Item | Size |
|---|---|
| Weights (fp16, 1.63 B params) | 3.27 GB |
| Conv state pool (`max_mamba_cache_size` 9483 slots, state 6, hidden 2048) | 2.82 GB |
| SSM state pool | 0 (LFM3 has no linear-attention SSM state) |
| SWA KV pool (131,512 tokens) | 0.63 + 0.63 GB |
| FULL KV pool (164,390 tokens) | 0.94 + 0.94 GB |
| Memory pool end (available) | 6.13 GB |
| CUDA graph capture (8 buckets, 2.0 s) | 0.11 GB |
| Available after init | 4.36 GB |
| `nvidia-smi` steady | 11,917 / 16,384 MiB |

## 4. GPU trace and bottlenecks

Traces: `traces/decode-bs8-1024.txt`, `traces/decode-bs1-1024.txt`,
`traces/prefill-bs1-64.txt` (`torch.profiler`, CUDA activities, `key_averages` table).
Environment limits: `nsys` cannot import `qdstrm` (no `.nsys-rep`), `ncu --target-processes all`
kills the scheduler at init, and `sgl.Engine` children are not visible to a parent-side profiler —
so the in-process tables below are the trace of record.

Decode, bs=8, ctx 1024 (GPU 10.7 ms of the 11.2 ms step):

| Kernel | ms/step | Share |
|---|---|---|
| cutlass SM70 fp16 wmma GEMM | 4.26 | 40% |
| cutlass SM70 fp16 GEMM (small shapes) | 1.38 | 13% |
| Triton paged decode attention stage1 (11 layers × 205 µs) | 2.26 | 21% |
| RMSNorm (HF-cast semantics, ~97 calls) | 0.36 | 3% |
| elementwise / index_select / cat (conv window, gates, residual) | ~2.4 | 22% |

Decode, bs=1, ctx 1024 (GPU 7.7 ms):

| Kernel | ms/step | Share |
|---|---|---|
| cutlass fp16 GEMM | 2.80 | 36% |
| cuBLAS GEMV `gemv2T_kernel_val` (59 + 13 calls) | 2.24 | 29% |
| Triton decode attention stage1 | 0.40 | 5% |
| RMSNorm | 0.35 | 4.5% |
| elementwise / conv window bookkeeping | ~1.9 | 25% |

Prefill, bs=1, 64 tokens (GPU 18.9 ms, wall 57 ms):

| Kernel | ms/request | Share |
|---|---|---|
| cutlass fp16 GEMM | 7.9 | 42% |
| Triton paged extend `_fwd_kernel` (11 × 615 µs) | 6.8 | 36% |
| volta s884 GEMM + LM head | 1.7 | 9% |
| RMSNorm | 0.4 | 2% |

`INFERRED` bottlenecks:

1. **Decode is weight-bandwidth bound.** Each step streams 3.27 GB of weights; the GEMM/GEMV
   portion runs at 584 GB/s (bs=8) and 649 GB/s (bs=1) of ~900 GB/s peak. Remaining headroom is
   limited without quantization (excluded from scope) — the lever is larger batch, which is why
   throughput keeps rising to c32.
2. **Attention is the second cost at bs≥8** (21%): the SM70 split-K decode kernel launches
   205 µs per layer for 8 requests × 1024 keys. SWA layers (5 of 11) only scan 256 keys.
3. **Short-prompt prefill is CPU/launch bound**: 64 tokens take 57 ms wall but 18.9 ms GPU; the
   gap is eager per-layer Python dispatch. The Triton extend kernel itself (615 µs/layer at
   64 tokens) is the largest GPU item, and it grows quadratically, so 4096-token prefill is
   attention dominated (0.55 ms/token vs 0.28 ms/token at 1024).
4. Conv state bookkeeping (index ops, cats, gated elementwise) is ~22% of a bs=1 step; it is a
   handful of small kernels per conv layer, not a bandwidth problem.

No optimisation was adopted: the candidates measured or reasoned about (fused residual RMSNorm
≈3%, conv update fusion ≈2–4%, RoPE table caching — prefill only, already <1% of a 64-token
prefill's GPU time) are below the noise of the remaining runway, and each would perturb the
M1-validated numerics. Recorded here as evidence rather than applied.

## 5. Limitations

- `enable_hierarchical_cache` + LFM3 still routes to `HiMambaRadixCache` (no SWA component);
  HiCache is out of scope for this milestone.
- 4096-token prompts with `logprob_start_len=0` still OOM on 16 GB (logits buffer 1.95 GiB), so
  full-prompt logprob comparison is only done at ≤1000 tokens; 8K is validated by greedy ids +
  generated-token logprobs.
- `torch_native` attention backend cannot warm up this hybrid model
  (`get_cuda_graph_seq_len_fill_value` NotImplementedError); only `triton` is validated.
