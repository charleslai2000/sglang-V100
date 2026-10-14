# S1 Falcon-H1 GPTQ serving acceptance

Date: 2026-10-04

## Target

- Host: `gpushare-v100`, Tesla V100-SXM2-32GB (SM70), CUDA 12.8, Torch 2.9.1+cu128.
- Model: `/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`, configured for 8K context; checkpoint revision pinned by S1 is `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- SGLang worktree: `milestone/s1-falcon-h1`, baseline `dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3`.
- Runtime log: `/hy-tmp/sglang-logs/falcon-h1-s1-clean.log`; long-context run: `/hy-tmp/sglang-logs/falcon-h1-s1-context.log`.

## Findings and corrective changes

- The checkpoint is mixed precision: BF16 attention and Mamba weights; GPTQ v1 INT4 FFN, group size 128, symmetric, `desc_act=true`.
- The existing SM70 GPTQ shuffle GEMM yielded FP16 infinities for captured layer-43 `down_proj` activations: with real activation max 6,664, its output had an infinity while an independently dequantized FP32 reference remained finite (max absolute output 46,798). The existing no-shuffle, `g_idx`-aware path remained finite (max difference from reference 88.61 before the model's down multiplier). For a separate captured activation with max 4,176, no-shuffle max difference was 60.54 against reference max 30,906.
- `GPTQLinearScheme` now selects `use_shuffle=False` for `desc_act` on devices below SM80. It preserves original packed weights and `g_idx`; no checkpoint canonicalization or new GPTQ kernel was added.
- In that route, the recurrent no-shuffle kernel's clear/atomic accumulation had the same multi-K-block output-initialization race. `gptq_kernel.cu` now clears the output slice on the launch stream before the kernel, and removes in-kernel z==0 clears. This applies to the shuffle and alt dispatchers and their affected variants.
- Falcon-H1 Mamba2 prefill produced finite FP32 reference SSD outputs as large as 1.07e5, outside FP16 range. SGLang's SM70 state policy had forced temporal state to FP16. The temporal state now retains the upstream FP32 default; convolution state remains FP16. The SSD scan/output buffer and gated norm use FP32 on SM70 FP16 execution, then cast to model dtype before the output projection.
- Falcon-H1 loader fixes preserve mixed BF16 attention / GPTQ FFN construction and make stacked parameter name probing non-mutating. Optional FlashInfer communication imports are skipped on SM70.

## Serving results

Server log identifies:

- `FalconH1ForCausalLM`, `quant=gptq`, `bits=4`; model load completed in 8.03 s, reported weight memory 8.09 GB.
- Attention decode: grouped TileLang G6/D256 split-KV; paged prefill: TileLang.
- Linear-attention backend: Triton decode and prefill.
- FP16 model compute on SM70; Mamba temporal state FP32.

Acceptance runs on the clean (uninstrumented) runtime tree:

- `/model_info`: HTTP 200, generation model type `falcon_h1`.
- Three prompt types (France capital, first primes, Python function) each generated 16 nonzero tokens. Each repeated greedy request returned exactly the same token IDs as its first request.
- Prefill/decode request: 400 prompt tokens, 24 generated nonzero tokens.
- Factual QA with 1,041 prompt tokens and 4 generated tokens returned `42.`.
- Factual QA with 4,113 prompt tokens and 4 generated tokens returned `42.`.
- Concurrent requests (two distinct prompts) both returned nonzero, distinct token sequences.
- GPU process was stopped after testing; `nvidia-smi` reported 1 MiB used.

Caveat: tokenizer reports `TokenizersBackend` after retries (model-specific tokenizer attributes may be absent). The tested plain-text prompts work; chat-template behavior was not separately qualified in this acceptance run.

## Result

S1's stated serving smoke matrix passed on the V100 with the existing g_idx-aware legacy GPTQ path. This is serving acceptance, not a claim of broad quality benchmarking or performance qualification. No commit or push is recorded by this file.
