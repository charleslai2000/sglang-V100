# E047 — results: LFM3 width-7 Short Conv

All numbers on Tesla V100-SXM2-16GB (SM70), fp16, single card, deliverable flags
(graph ON, radix ON, `extra_buffer`). `OBSERVED` = measured, `INFERRED` = derived.

## 1. Correctness

### 1.1 Conv kernels vs independent FP32 reference

`F.conv1d` reference in fp32 over the same window; the kernel is fed the fp16
window (varlen layout, six historical values).

| Case | state bit-exact | output max abs err | output mean abs err |
|---|---|---|---|
| single seq 40 tok | yes | 5.7e-3 | 3.6e-4 |
| two seqs 20+20 | yes | 3.9e-3 | 3.6e-4 |
| length 1 / 3 / 6 / 7 | yes | 1.9e-3 / 3.6e-3 / 2.6e-3 / 3.8e-3 | ~3.5e-4 |
| mixed 1+8+3, mixed initial-state flags | yes | 3.1e-3 | 3.0e-4 |
| chunk continuation, 300 tok, bias on | yes | 3.9e-3 | 3.9e-4 |
| no initial state | yes | 3.7e-3 | 2.9e-4 |

Output error is pure fp16 output rounding (values are O(2.6), 1 ulp ≈ 2e-3); the
conv state (the last six `B*x` values) is copied, so it is bit-exact. Accumulating
exact fp16 products in fp32 (instead of fp16 products) cut the mean error from
5.6e-4 to 3.6e-4 and is what makes the layer reproduce the previous cuDNN path.

### 1.2 Fused decode kernel vs the previous six-kernel step

| Config | output max abs err | state |
|---|---|---|
| bs 8, 2048 ch, no bias | **0.0** | bit-exact |
| bs 8, 2048 ch, bias | **0.0** | bit-exact |
| bs 1 / bs 32 / 256 ch / 128 ch | 0.0 (one case 2.4e-4 = 1 fp16 ulp) | bit-exact |

Padded rows are skipped by their reserved slot 0 instead of aliasing a real slot
(the previous `index_select`/`index_put` with `-1` raises a CUDA device assert in
this PyTorch build, so the skip also removes a latent crash).

### 1.3 End-to-end vs the `F.conv1d` implementation (E046 harness)

| Phase | prompts | greedy ids | prompt logprob | generated logprob |
|---|---|---|---|---|
| cold | 64 / 256 / 1024 / 4096 / pA / pB | all equal | **0.0** | **0.0** |
| prefix hit | same six | all equal | **0.0** | **0.0** |

Bit-identical to the previous implementation, so E046's oracle agreement
(4.8e-7 on 8K, ≤1e-3 cold-vs-warm) carries over unchanged.

### 1.4 Edge coverage (`perf/e047_edges.json`)

- Prompt lengths 1–8 (shorter than and equal to the conv window) and 13: all run;
  graph ON and OFF give identical ids (len 1/6/64/1024 checked pairwise).
- Chunked prefill (`chunked_prefill_size=256`) vs unchunked: ids equal at 300 and
  1000 tokens.
- Repeated prefix hits (600+64 token prompt): ids equal, `cached_tokens` 256→512.
- Forced mixed prefill+decode batch (server log: `Prefill batch #new-seq: 1 …
  #running-req: 1/2/3`) plus a second request wave reusing slots: 8/8 responses
  match the cold reference.
- Unit tests: `test/registered/unit/mem_cache/` + `test/registered/unit/configs/`
  pass (1128 passed / 480 skipped). `test/registered/layers/mamba/test_causal_conv1d.py`
  has 14 failures in `test_causal_conv1d_varlen[10-…]` / `[4-…]`, and they are
  **pre-existing**: the failure is the test's own `F.conv1d` reference raising
  `RuntimeError: GET was unable to find an engine to execute this computation`
  (cuDNN engine gap for those shapes/shapes+dtype on this host). Re-running the
  same selection against the `95996ab0e` kernels reproduces exactly
  `14 failed, 26 passed`, so E047 changes no width-2/3/4 behaviour.

## 2. GPU-side A/B (3 reps each, in-process CUDA events)

| batch / ctx | prefill old → new (ms) | Δ | decode step old → new (ms) | Δ | tok/s new |
|---|---|---|---|---|---|
| 1 / 64 | 54.8 → 50.7 | **−7.5 %** | 8.526 → 8.110 | **−4.9 %** | 123.3 |
| 1 / 1024 | 283.3 → 260.1 | **−8.2 %** | 8.289 → 7.868 | **−5.1 %** | 127.1 |
| 8 / 1024 | 1921 → 1885 | −1.9 % | 11.212 → 10.541 | **−6.0 %** | 758.9 |
| 8 / 4096 | 17619 → 17598 | −0.1 % | 16.747 → 16.129 | −3.7 % | 496.0 |
| 32 / 1024 | — | — | 13.26 → 12.75 (2 reps) | **−3.9 %** | 2510 |

The prefill gain is largest at low batch (per-layer host syncs and per-request
loops dominate there) and vanishes at 8×4096, where the Triton extend attention
kernel is 78 % of the prefill.

## 3. Service A/B (HTTP streaming, 3 reps per cell, cache flushed per cell)

| in/out | c | output tok/s old → new | Δ | TTFT mean old → new | Δ | TPOT old → new | Δ |
|---|---|---|---|---|---|---|---|
| 64/256 | 1 | 115.6 → 122.0 | +5.5 % | 66.1 → 66.5 ms | +0.7 % | 8.42 → 7.96 ms | −5.4 % |
| 64/256 | 8 | 767 → 831 | +8.4 % | 150.8 → 117.6 ms | −22.0 % | 9.84 → 9.17 ms | −6.8 % |
| 64/256 | 16 | 1467 → 1645 | +12.1 % | 204.6 → 142.2 ms | −30.5 % | 10.08 → 9.14 ms | −9.4 % |
| 64/256 | 32 | 2336 → 2840 | +21.6 % | 430.2 → 238.1 ms | −44.7 % | 11.99 → 10.22 ms | −14.8 % |
| 1024/256 | 1 | 104.5 → 111.1 | +6.3 % | 296.9 → 273.3 ms | −7.9 % | 8.43 → 7.96 ms | −5.6 % |
| 1024/256 | 8 | 469 → 513 | +9.4 % | 1320 → 1003 ms | −24.0 % | 11.92 → 11.68 ms | −2.0 % |
| 1024/256 | 16 | 917 → 983 | +7.2 % | 1440 → 1056 ms | −26.6 % | 11.80 → 12.12 ms | +2.7 % |
| 1024/256 | 32 | 1271 → 1351 | +6.3 % | 2357 → 1774 ms | −24.7 % | 15.88 → 16.65 ms | +4.9 % |
| 4096/128 | 1 | 37.2 → 37.7 | +1.5 % | 2295 → 2300 ms | +0.2 % | 9.05 → 8.60 ms | −4.9 % |
| 4096/128 | 8 | 153.8 → 156.7 | +1.8 % | 4329 → 3989 ms | −7.9 % | 18.26 → 19.99 ms | +9.5 % |
| 4096/128 | 16 | 340.5 → 350.5 | +3.0 % | 3738 → 2869 ms | −23.3 % | 17.76 → 23.25 ms | **+30.9 %** |
| 4096/128 | 32 | 389.7 → 398.7 | +2.3 % | 6150 → 4849 ms | −21.2 % | 33.94 → 42.35 ms | **+24.8 %** |

`OBSERVED`: output throughput improves in all 12 cells (+1.5 % … +21.6 %), TTFT
drops at c≥8 (−8 % … −45 %), TPOT drops at low/mid concurrency. The two
saturated 4096-token cells show worse TPOT: faster prefill admits more requests
into decode simultaneously, so the decode batch grows (same effect visible in the
step A/B, where per-step time at fixed batch falls 3.7–6 %). Total throughput
still rises, so this is a queueing trade-off, not a kernel regression.

## 4. Operator budget (`benchmark/perf_telemetry/`)

Captured with `capture_forward.py`, attributed with `op_budget.py` (kernels joined
to their launching operator through Kineto `External id` / correlation; unmatched
kernels fall back to a kernel-name class). Traces and raw budgets:
`traces/`. Instrumented runs are separate from §2/§3 uninstrumented numbers.

### 4.1 Prefill, 1 × 1024 tokens, 2 forwards, eager (exact operator spans)

Reproduce with the shipped profile (the JSON in `traces/` is the output of this
command, not an older ruleset):

```bash
PYTHONPATH=python python benchmark/perf_telemetry/op_budget.py \
    experiments/E047-lfm3-width7-conv/traces/trace-new-prefill-bs1-ctx1024.json.gz \
    --profile lfm3 --out /tmp/budget.json
```

| owner | old kernels / ms | new kernels / ms |
|---|---|---|
| forward envelope | — / 637.0 | — / 535.5 |
| GPU busy / idle | 79.4 % / 131.5 ms | **94.0 % / 31.9 ms** |
| attention (rope + paged extend + KV store) | 720 / 423.0 | 681 / 348.2 † |
| `mlp.gate_up` | 48 / 32.5 | 50 / 68.8 ‡ |
| `mlp.down` | 98 / 19.5 | 98 / 19.5 |
| `conv.in_proj` | 28 / 8.49 | 33 / 8.55 |
| **`conv.kernel+state`** | **474 / 5.55** | **119 / 1.46** |
| `conv.out_proj` | 26 / 2.97 | 30 / 2.99 |
| `mlp.act` | 49 / 2.81 | 51 / 2.86 |
| host time in the conv span | 452.8 ms | **33.3 ms** |
| host time total (all spans) | 1267 ms | 212 ms |

† In the new trace 14 large extend-attention kernels (265.7 ms) are reported
under the `name:attn.prefill_kernel` class because their launcher has no
resolvable aten op, while in the old trace the same kernels resolve into the
`self_attn` span. Both totals are shown; the reliable comparisons are the conv
rows, the envelope and the GPU idle/ busy split, plus the uninstrumented A/B in
§2 (which is the performance of record).
‡ `gate_up` differs because in the old run part of those kernels fell into the
`self_attn` bucket by launch containment; totals are what matter.

### 4.2 Decode, batch 8 @ 1024, CUDA-graph replay (name-class attribution)

The Python forward never runs under graph replay, so per-operator spans cannot
exist; kernels are classified by name and the `step[...]` span is the envelope.

| owner | old kernels / ms | new kernels / ms |
|---|---|---|
| envelope (10 steps) | — / 125.4 | — / 119.6 |
| GPU busy | 85.4 % | 84.1 % |
| kernel duration sum | 107.2 | 100.6 |
| `gemm.cutlass` | 1080 / 56.5 | 1080 / 56.3 |
| `attn.decode_kernel` | 110 / 22.7 | 110 / 22.5 |
| `elementwise` | 3820 / 14.9 | 3170 / 11.3 |
| `norm` | 1190 / 4.30 | 1190 / 4.23 |
| `conv.cat` | 350 / 1.79 | 220 / 1.02 |
| `conv.index` | 140 / 1.12 | 10 / 0.07 |
| `conv.kernel` (`_lfm3_short_conv_decode`) | 130 / 0.57 | 130 / 0.57 |
| `reduce` | 140 / 1.30 | 10 / 0.36 |

### 4.3 Decode, batch 8 @ 1024, eager (exact operator spans, graph off)

| owner | old kernels / ms | new kernels / ms |
|---|---|---|
| `conv.kernel+state` | 1179 / 4.96 | **246 / 0.65 (−87 %)** |
| host time in the conv span | 201.5 ms | 133.2 ms |
| `attn.core` | 3494 / 32.5 | 3490 / 31.6 |
| `mlp.gate_up` | 244 / 22.7 | 242 / 22.9 |
| `mlp.down` | 486 / 13.9 | 481 / 14.0 |
| `lm_head` | 20 / 6.32 | 20 / 6.29 |
| `conv.in_proj` | 262 / 5.34 | 262 / 5.39 |
| `norm` | 908 / 2.90 | 914 / 2.94 |
| GPU busy (eager) | 9.2 % | 9.5 % |

`INFERRED` bottleneck after the change: decode is still weight-bandwidth bound —
GEMM 56 %, attention 22 %, elementwise 11 %, norms 4 %, conv ≈ 1 %. The conv
optimisation is exhausted: what remains inside it is one fused kernel plus the
checkpoint copy. Eager decode is host-dispatch bound (9 % GPU busy), which is why
graph replay matters far more than any kernel micro-optimisation on this path.

## 5. Hardware counters — UNSUPPORTED (permission)

`ncu` is installed but the process cannot access GPU performance counters:

```
$ ncu --target-processes all --clock-control none \
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
      --launch-count 3 python /var/tmp/e047_ncu_probe.py
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

CUPTI Range Profiling (the E030 route) needs the same privileged counter access,
so SM / Tensor / HBM / L2 counters are **UNSUPPORTED** here — not zero. The
permitted device-level signal is the NVML sampler, verified live during a
bs=32 decode loop: 20 samples, `utilization.gpu` peak 100 % (mean 16 % including
idle), `power.draw` peak 147 W, clocks 1312/877 MHz
(`benchmark/perf_telemetry/gpu_sampler.py`). NVML utilization is reported as a
device-level busy proxy only, never as SM or HBM peak.

## 6. Decisions

Kept:
- width-7 varlen Triton prefill kernel (one launch per layer, no per-request loop,
  no host sync) and the once-per-forward `_Lfm3ConvBatchContext`;
- fused decode kernel (six kernels → one per layer, −87 % conv kernel time);
- fp32 accumulation of exact fp16 products on the width-7 branch: identical
  numerics to the previous implementation, no tolerance widening.

Rejected / not adopted:
- Extending `_causal_conv1d_update_kernel` for width 7 (it is a per-token loop
  specialised for widths ≤ 4 and cannot fuse the `B*x`/`C*` gates; a separate
  fused kernel is smaller and faster).
- `sgl_kernel` dispatch for width 7: silently wrong (verified), so the dispatch
  guard stays.
- Prefill graph / PDL / tile-size retuning: out of scope, and the prefill budget
  shows the extend attention kernel (78 %) rather than the conv as the limiter.
- The 22 `cat` kernels per decode step are outside the conv path (scheduler/mamba
  state plumbing) and were left untouched.
