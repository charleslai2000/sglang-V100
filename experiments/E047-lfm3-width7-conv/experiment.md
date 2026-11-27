# E047 — LFM3 width-7 Short Conv on SM70

## Question

The LFM3 Short Conv (depthwise causal conv, kernel width 7) ran on the reference
`F.conv1d` fallback: per request a `cat` + `conv1d` + slice + index copy on prefill,
a six-kernel step on decode, and a host sync per layer. Can the same numbers be
produced by the in-tree Triton varlen kernel (extended to width 7) plus one fused
decode kernel, and does that show up end to end?

## Setup

| Item | Value |
|---|---|
| Base commit | `95996ab0e` (E046) |
| Worktree | `/data/workspace/3rdparty/sglang-V100`, branch `main` |
| GPU | Tesla V100-SXM2-16GB, cc 7.0, UUID `GPU-eaeb0cc6-…992189` |
| Python | `/data/t011/repos/marlin_v100-g002/source/.venv/bin/python` |
| Model | `/var/tmp/lfm3` (LiquidAI/LFM3-1.7B, fp16) |
| Server flags | `--dtype float16 --attention-backend triton --sampling-backend pytorch --context-length 8448 --mem-fraction-static 0.60 --max-running-requests 32 --cuda-graph-max-bs 32 --mamba-scheduler-strategy extra_buffer` (graph ON, radix ON) |

## Port scope

Upstream reference: the vLLM-derived kernels already in this fork
(`srt/layers/attention/mamba/causal_conv1d_triton.py`, `_causal_conv1d_fwd_kernel`
/ `_causal_conv1d_update_kernel`); width 7 is the missing branch. No new kernel
architecture, no quantization, no PDL, no prefill graph, no cache changes.

1. `causal_conv1d_triton.py` — width-7 branch in `_causal_conv1d_fwd_kernel`:
   six history columns loaded by the chunk-0 program, weights `w_col0..w_col6`,
   window shift over six slots, and fp32 accumulation of exact fp16 products so
   the swap keeps the previous cuDNN numerics. Widths 2/3/4/5 untouched.
2. `causal_conv1d.py` — `sgl_kernel`'s `causal_conv1d_fwd` silently computes
   garbage for width > 4 (verified), so width-7 inputs are routed to Triton.
3. `models/lfm3_conv.py` (new) — fused decode kernel: reads the packed
   `[B | C | X]` projection, forms `B*x` with fp16 rounding, convolves the six
   carried values, shifts the window in place, writes `C*conv_out`; skips the
   reserved slot 0 used by padded CUDA-graph rows.
4. `models/lfm3.py` — `_Lfm3ConvBatchContext` resolves the batch metadata once
   per forward (state slots, `query_start_loc`, per-sequence lengths,
   initial-state flags, checkpoint targets) and shares it across the 13 conv
   layers; prefill calls the varlen kernel, decode calls the fused kernel, mixed
   batches run both and concatenate.

## Protocol

- Conv kernel vs independent FP32 reference (varlen, short sequences, chunk
  continuation, no-state, bias on/off) — output and full conv state.
- Fused decode vs the previous torch step, including padded rows.
- E046 correctness matrix (prompts 64/256/1024/4096/pA/pB, cold vs prefix hit,
  graph on) compared against the `F.conv1d` implementation.
- Edge coverage: prompt lengths 1–8, chunked prefill (chunk 256), repeated prefix
  hits, mixed batch, graph off.
- A/B: three repetitions per (batch, context) point, old and new code, same host,
  same flags; HTTP matrix c1/c8/c16/c32 × (64/256, 1024/256, 4096/128) with three
  repetitions and a cache flush before each cell.
- Operator budget via `benchmark/perf_telemetry/` (see `results.md` §5).
