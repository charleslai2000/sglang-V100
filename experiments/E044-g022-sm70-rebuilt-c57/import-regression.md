# GPTQ circular-import regression (2026-10-21)

## Source root cause and fix

The cycle was `gptq_kernels -> quantization.marlin_utils -> quantization.__init__ -> gptq.__init__ -> gptq.gptq -> gptq.schemes -> gptq.schemes.gptq_linear -> gptq_kernels.GPTQLinearKernel` (and similarly for Marlin classes). The targeted fix removes those top-level GPU-kernel imports. `GPTQLinearScheme._init_kernel` resolves `GPTQLinearKernel`; Marlin scheme `__init__` resolves `GPTQMarlinLinearKernel`; `create_weights` resolves `MarlinLinearLayerConfig`. The public `gptq_marlin_moe_repack` export is now a lazy wrapper, retaining its name and behavior while avoiding eager kernel-module initialization. No quantization math or dispatch was changed by these import edits.

## Fresh-process regression

All commands used `/data/t011/venv/sglang/bin/python`, `PYTHONPATH=$PWD/python`, from this checkout; each was a separate fresh process:

A. `from ...gptq_kernels import ...; from ...gptq import GPTQConfig` — PASS.
B. `GPTQConfig`, `GPTQLinearScheme`, `GPTQLinearKernel` reverse import — PASS.
C. `get_quantization_config('gptq') is GPTQConfig` — PASS. Constructed `GPTQConfig(4,128,True,False,{},'',False,False)`, `GPTQLinearScheme(config)` and asserted exact kernel class identity — PASS.
D. Imported `GPTQMarlinConfig`, `MoeWNA16Config`, `CompressedTensorsWNA16MoE`, and public repack wrapper — PASS.

GPTQ's `get_moe_scheme` was also checked and explicitly rejects unsupported GPTQ MoE (raises, not silently accepting). The V100 environment emits its existing warning that the optional `marlin_v100` MoE kernel was not found; this test does not claim MoE execution support or numerical qualification.

## Commit/push

Validated source/evidence committed as `f472803a6a14b24a3e4ec9bc68c1d9fb79260d8b` and pushed to `origin/main`; remote ref was independently read back as the same SHA. The unrelated experimental runtime dispatch and compressed_tensors WNA16 import adjustment remain uncommitted and are not part of this commit.

## Exploratory dispatch status

The `gptq_kernels.py` REBUILT-C57 route remains unqualified; its current diff is archived outside the repository at `/data/t011/artifacts/rebuilt-c57-dispatch-exploratory.patch`, SHA256 `5be2078fa9cc4b198ea257618cc7338fbf5356c5f874a6ca8226539d8243ba70`. Standalone binary SHA `ab5db27bade8cb868faed6ee4b72c9f7047f838b0ac23c4cdfa4c2b895944d95` has only build/load/small-matrix smoke evidence. No worker dispatch, Graph kernel identity, model correctness, state, capacity or service result is claimed.
