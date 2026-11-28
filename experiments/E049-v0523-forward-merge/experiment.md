# E049 — Forward merge of official v0.5.23 into the V100 harness

## Question

`v100-harness` sat on an official base from June. Which release does it actually
inherit, how far behind is it, and does a real forward merge (not a patch
harvest) keep LFM3, the E045–E048 results and the measured V100 performance?

## Setup

| Item | Value |
|---|---|
| Branch | `v100-harness` |
| Merge subject | `v0.5.23` = `297e54fd10` (2026-10-28) |
| History | complete: `git fetch --unshallow` from `sgl-project/sglang` |
| Python | `/data/t011/repos/marlin_v100-g002/source/.venv/bin/python` (torch 2.9.1+cu128) |
| GPU | Tesla V100-SXM2-16GB (cc 7.0) |
| Model | `/var/tmp/lfm3` (LiquidAI/LFM3-1.7B, fp16) |

## Ancestry (measured on complete history)

| Ref | Commit | Relation |
|---|---|---|
| merge-base(v100-harness, v0.5.23) | `cdd06011a1` (2026-06-01) | the real official base inherited by this tree |
| `v0.5.23` | `297e54fd10` (2026-10-28) | 7647 commits ahead of that base |
| v100-harness before the merge | `a7e29308b3` | 328 commits ahead of that base |
| `official/main` at merge time | `d13cf59d83` (2026-11-28) | 2460 commits past the tag |

So the previous version-integration round was working against a five-month-old
base: the tag was not an ancestor, and the six "missing" commits were only the
tail of a 7647-commit gap. The earlier patch-id comparison could not show that
because the clone was shallow.

## Merge scope

`git merge v0.5.23` touched 11408 files; 278 paths conflicted (222 exist
upstream, 56 are ours only). Resolution policy:

| Area | Resolution |
|---|---|
| scheduler, managers, model_runner, server_args, arg_groups, mem_cache, attention backends, speculative, quantization, model registry | upstream |
| new `kernels/` + `jit_kernel/` layout, AOT extension sources | upstream |
| LFM3: `configs/lfm3.py`, config registration, `hybrid_arch`, `is_hybrid_swa_model`, `get_hybrid_layer_ids`, `models/lfm3.py`, `models/lfm3_conv.py` | ours, re-applied on the new runtime |
| hybrid SWA+SSM cache routing (E046) | upstream now builds `UnifiedRadixCache(FULL, SWA, MAMBA)` itself; no local patch |
| width-7 varlen conv (E047) | moved to `kernels/ops/mamba/causal_conv1d_triton_w7.py`; `srt/.../causal_conv1d.py` routes width > 4 there, upstream kernel verbatim for widths 2/3/4 |
| `/v1/decisions`, `/v1/score` (E048) | upstream's implementation; the E048 adapters were deleted because the runtime they bridged to is now the official one |
| fork model work upstream reimplemented (Qwen3.8/Qwen4/QSA/dflash/dspark/PLE, C-AR PCIe) | dropped, not duplicated |
| `README.md` | ours (V100 harness); `.gitignore` union; `python/pyproject.toml` keeps the installed CUDA 12.x pins plus additive upstream entries |

Environment bridges required by the newer code on this pinned stack (all
documented in the commit body): SM70 allowed again in `load_model_utils`, bf16
intrinsic fallbacks and a `#pragma unroll` removal in the JIT headers, flashinfer
name aliases, guarded mxint4 / KimiK3 imports, and a fallback for the grouped
small-copy Triton kernel.

## Results

See `results.md`: correctness anchors unchanged (8K oracle 4.77e-07), cache hits
and graph parity reproduced, `/v1/decisions` still matches the independent
oracle, hot-prefix extend slightly faster, and the E047 HTTP matrix improves in
every warm cell.
