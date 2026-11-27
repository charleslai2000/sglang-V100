# E046 — LFM3 (M2): CUDA Graph + Radix Cache serving path on V100

## Question

Can LFM3 FP16 on SM70 run with **both CUDA Graph and Radix Cache enabled**, keeping the
M1-validated numerics, while `--mamba-scheduler-strategy extra_buffer` tracks the Short Conv
state across prefix-cache hits?

## Setup

| Item | Value |
|---|---|
| Base commit | `cfcc36c11` (M1) |
| Worktree | `/data/workspace/3rdparty/sglang-V100`, branch `main` |
| GPU | Tesla V100-SXM2-16GB, cc 7.0 |
| Python (SGLang) | `/data/t011/repos/marlin_v100-g002/source/.venv/bin/python` |
| Oracle | `/var/tmp/oracle-venv` (transformers LFM3 eager, fp16) |
| Weights | `/var/tmp/lfm3` (LiquidAI/LFM3-1.7B) |
| Server flags | `--dtype float16 --attention-backend triton --sampling-backend pytorch --context-length 8448 --mem-fraction-static 0.60 --max-running-requests 32 --cuda-graph-max-bs 32 --mamba-scheduler-strategy extra_buffer` (radix cache and CUDA graph ON by default) |

## Changes

1. `python/sglang/srt/models/lfm3.py` — Short Conv state tracking for `extra_buffer`:
   - state slots come from the backend metadata (`mamba_cache_indices`) instead of a
     per-batch `req_pool_indices` lookup, so the captured decode graph sees the captured
     index buffer;
   - decode rows advance the window and then copy it into the planned checkpoint slots
     (`track_mamba_states_if_needed`), pure indexed copies, no host sync, graph-safe;
   - prefill rows snapshot `B*x` at the backend-planned boundaries
     (`metadata.track_conv_indices`) before consuming the carried window;
   - prefill always reads the stored window (fresh slots are zeroed and cache-hit slots are
     COW-copied by the deferred `init_forward_metadata` step), so chunked prefill, prefix hit
     and empty prefix share one code path.
2. `python/sglang/srt/mem_cache/registry.py` — a model that is both hybrid-SWA and hybrid-SSM
   (LFM3) now builds `UnifiedRadixCache` with `(FULL, SWA, MAMBA)`. The previous routing sent it
   to `SWARadixCache`, which stores no conv state, so a prefix hit silently kept the stale
   window (logprob error ~1.5 nats).
3. `test/registered/unit/mem_cache/test_registry.py` — regression test for the new routing.

## Protocol

- Correctness matrix (graph ON/OFF × `extra_buffer`/`no_buffer`, cold vs prefix hit) over
  prompts 64/256/1024/4096 and two prompts sharing a 600-token prefix.
- 8K-token prompt (4096-token block repeated) against the HF fp16 oracle, greedy 8 tokens,
  no full-prompt logits.
- Mixed prefill+decode batches and state-slot reuse under concurrent HTTP load.
- Perf matrix over c1/c4/c8/c16/c32 with prefix cache flushed before each combo.
- GPU-side latency via in-process `runner.extend`/`runner.decode` with CUDA events, kernel
  breakdown via `torch.profiler` (CUDA activities).
