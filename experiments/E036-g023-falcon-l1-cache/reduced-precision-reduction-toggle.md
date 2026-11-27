# 768-row QKV reduced-precision reduction diagnostic

## Method

Using the existing frozen layer-0 position-767 FP16 hidden input and loaded FP16 QKV weight, a separate diagnostic process ran the same 768-row projection twice. The first call used `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True` as the production baseline; the second changed only this flag to `False`. The comparator is the existing FP64 dot product over exact FP16 operands, rounded to FP16. No production default/source was modified.

## Results

| Run | Elements unequal to FP64-rounded | Absolute max / mean | Relative max / mean | FP16 ULP max / mean |
|---|---:|---:|---:|---:|
| Baseline flag True | 698/2048 | 0.015625 / 0.000809006 | 7.506% / 0.0499% | 99 / 0.6963 |
| Flag False | 16/2048 | 0.0078125 / 0.000013558 | 0.1408% / 0.000592% | 2 / 0.00830 |

Changing the flag altered 696/2048 outputs (max absolute difference `0.015625`, mean `0.000807155`). With reduction disabled, the result is much closer to FP64 and the maximum FP16 ULP distance drops from 99 to 2. It does not become bit-identical to the FP64-rounded reference, but strongly removes the anomalous 768-row deviation in this frozen-input experiment.

## Disposition

This implicates PyTorch/CUDA FP16 reduced-precision reduction in the abnormal 768-row path and provides a focused diagnostic lead. It is not a production fix or proof that changing the global flag is safe: the flag was toggled only in an isolated process, without a full service correctness regression or impact assessment. Do not alter production defaults from this evidence alone. G023/T001 remains BLOCKED; no Mamba/KV, logits, lifecycle, or performance experiment was repeated.

Artifact: `/data/t011/artifacts/g023-qkv-reduced-precision-toggle.py`; result `/data/t011/artifacts/g023-qkv-reduced-precision-toggle.log`.
