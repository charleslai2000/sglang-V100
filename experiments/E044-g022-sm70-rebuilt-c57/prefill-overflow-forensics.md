# Deterministic L43/down Prefill overflow forensic (2026-10-21)

## Root cause

Frozen T011 binary: `/data/t011/build/qpn128-final/frozen/qpn2_multiwarp-6f3ee5732ca571149c9209e403c1d03330a2a151a56541ca40b38c2f849af90f.so`, SHA256 `6f3ee5732ca571149c9209e403c1d03330a2a151a56541ca40b38c2f849af90f`. Exact local checkpoint path `/data/t011/hf/Falcon-H1-7B-Instruct-GPTQ-Int4`; a fresh worker in each process, FP16 activation/runtime, Q4 int32 qweight, FP16 scales, group128, desc_act. Deterministic input IDs `[0,1,...,511]`, positions `[0,1,...,511]`, worker's newly initialized recurrent state. This reproduced the issue at L43/down coordinate `[229,175]`.

The first nonfinite was +Inf from `model.layers.43.mlp.down_proj`; all prior observed quantized Linear outputs and down input were finite. Down input row 229 max abs 5196. The actual down multiplier is `0.032552083333333336` (from Falcon model config). FP64 GPTQ math from captured activation and packed checkpoint nibble/zero/scale/g_idx at N=175 gives raw `77871.5990610119`; FP32 groupwise helper gives `77871.6171875` (absolute difference .0181265, relative `2.33e-7`). Casting raw output directly to FP16 produces +Inf. FP64 times original down multiplier is `2534.8827819339813`; FP32 result after multiplier is `2534.8833719889326`; FP16 result is 2534. This identifies FP16 output cast before the multiplier as the cause, not cuBLAS FP32 internal reduction overflow. The checkpoint values and multiplier were unchanged.

## Isolated experiment implementation/result

Uncommitted experimental source currently adds `gptq_gemm_fp32_groupwise` to `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py` and an explicit `SGLANG_T011_FP32_PREFILL_DOWN=1` branch to `python/sglang/srt/models/falcon_h1.py`. It decodes original packed Q4/zeros/scales/g_idx, accumulates FP32 group matmuls and applies the original multiplier before output cast. Unsupported TP/reduce-scatter/config/dtype/shape fail closed. No clamp, scale, token, checkpoint or QPN2 algorithm change. The 512-token Prefill with the opt-in helper yielded no nonfinite quantized Linear outputs; final hidden/logits finite. Hidden SHA `bb6ebb7c8c022ff666014555fe136ef944801675af90036dbb538bfd5a859216`, logits SHA `377f118ca6df8a4ca24ebc5bbf73b0e49d84778e873f1ba634c53218a22760d7`. Independent one-row FP64 vs helper coordinate comparison is above.

This proves a first targeted finite/correctly ranged result, not full numerical qualification: the helper uses FP32 `torch.matmul` on dequantized group weights, while independent reference uses FP64; report the rounding difference. It changes floating-point operation order versus native grouped FP16 kernel. Need compare full output against high-precision all-coordinate references or a validated FP32 baseline, execute sequential request state, Graph, and include identical arithmetic path in both service arms before performance acceptance.

## Artifacts

- Baseline forensics harness: `/data/t011/artifacts/t011-fp64-prefill-forensics.py`, current SHA256 `964171a8ac1abe682656674f72a2a6301a82330f8a2aeb8ee81e2c0fa8fc7971` (includes the repaired diagnostic capture and early exit after the complete Prefill). Baseline log path `/data/t011/artifacts/commonops128/t011-fp64-prefill-forensics.log`, SHA256 `a3cc0d8e5a12794c0493f87a826c4161404e863a7c18f44501e8987e2516d058`; includes exact coordinate and checkpoint tensor fingerprints.
- Opt-in finite full Prefill log: `/data/t011/artifacts/commonops128/t011-fp64-prefill-fixed.log`, SHA256 `207908d9ad5286e14d60f15b5a4e2f13f2d5aadce894c63eb53abeb63cc7c722`.
- Helper-vs-FP64 harness: `/data/t011/artifacts/t011-fp32-helper-oracle.py`, SHA256 `b721b3e21930a20033ea6800af0c667a19ebdd91878b332144c73762da2c5a9a`; run log `/data/t011/artifacts/commonops128/t011-fp32-helper-oracle.log`, SHA256 `cf1ce36c14781a2e97326b1110e7217019c6e74aa5d03a232b46add4fef53b2c`. Output reports raw FP64/FP32 and scaled/cast coordinate.

## Status

No runtime source fix is committed or enabled by default. Existing uncommitted REBUILT-C57 dispatch experiment and WNA16 import edit remain separate and untouched. A production/service change is not warranted until all-row numerical comparison, full model/Graph/state tests and paired Prefill regression measures pass. T011 remains ACTIVE.
