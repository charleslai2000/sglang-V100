# Owner-authorized double-FP16 residual experiment (not qualified)

Purpose: one bounded Tensor Core organization after measured FP32 Prefill regression. Formula implemented per weight element: `exact=(q-(qzero+1))*scale` in FP32, `hi=round_fp16(exact)`, `lo=round_fp16(exact-float32(hi))`; two FP16-input/FP16-weight cuBLAS GEMMs each accumulate to FP32 output; FP32 outputs are added, then existing Falcon multiplier is applied in FP32 and only then cast FP16. No persistent expanded weights. This changes the reduction organization and has no presumed exact equivalence.

Source symbols:
- `reconstruct_q4_fp32_kernel`: creates exact FP32 value for the strict baseline or (hi,lo) FP16 components for the experiment.
- `gptq_prefill_fp32_gemm(..., use_double_fp16)`: selects two FP32-output `cublasGemmEx` calls and FP32 sum, without changing shared handle policy.
- `FalconH1MLP.forward`: gated only by the existing opt-in plus `SGLANG_T011_FP16_RESIDUAL_DOWN=1`; candidate remains off by default.

Current status: compiles successfully for SM70. Binary SHA256 `/data/t011/build/prefill-gemm/t011_prefill_gptq.so`: `1085990cf345240ff2652ce776cfaa3ac879c468a258ba7c442aef355bafa055`; CUDA source SHA256 `5f9c7343eaa0533792b23816a1240517a67a4c8db0d54be021e36401bcb81bee`; op schema SHA256 `b7cf0cb087382caaa6e5ee1d2d56b2396c01974f68c93c1b280090232cd79f88`.

No double-FP16 numerical run has passed yet. Before use, verify all weights' residual distribution (zero/subnormal/underflow, max residual, relative residual), sampled and exhaustive FP64 oracle strict-bad=0 for L0/L22/L43 and all 44 real down layers including near-cancellation outputs, and final scaled FP32/FP16. In particular, FP32 accumulation rounds each high and low GEMM independently; their final FP32 sum differs from a single FP32 GEMM's reduction order. A strict local operator pass on the direct FP32 route does not transfer to this route. Then run fixed 512/1K/4K tests and full model hidden/logits/Graph before any cost qualification.
