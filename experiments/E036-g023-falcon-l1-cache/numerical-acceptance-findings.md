# G023 numerical acceptance evidence

## FP32 accumulation GEMM comparison

Using the frozen identical FP16 position-767 hidden input and loaded FP16 checkpoint weight, an independent PyTorch FP32 accumulation GEMM computed `X.float() @ W.float().T` in 768-row and 789-row batches. Results are compared to the captured production FP16 QKV concatenation; the prior FP64 oracle uses exact FP16 operands, FP64 dot product, then FP16 rounding. The captured QKV actuals correspond to these exact batch sizes.

| Prefill shape | captured vs FP32 | max/mean absolute | max/mean relative* | FP16 ULP distance max/mean | captured vs FP64-rounded (unequal, max/mean abs) |
|---|---:|---:|---:|---:|---:|
| 768 (builder) | 2048/2048 differ | 0.0141277 / 0.00103940 | 0.07507 / 0.000556 | 99 / 0.6973 | 698; 0.015625 / 0.00080901 |
| 789 (miss) | 2047/2048 differ | 0.00799942 / 0.00067855 | 0.001310 / 0.000176 | 2 / 0.00781 | 16; 0.0078125 / 1.36e-5 |

*Relative error uses `abs(captured - FP32) / max(abs(FP32), smallest positive FP32)`. ULP is ordered FP16 representable-value distance, comparing the FP16-cast captured result with FP16-cast FP32 output.

FP32 accumulation results cast to FP16 differ from FP64-rounded results in 2/2048 elements at shape 768 and 3/2048 at shape 789. The 789 path is close to both independent references; the 768 path is dramatically farther, with up to 99 FP16 ULP and max relative error ~7.5%. This is strong evidence of anomalous numerical behavior in the 768-row production path, not ordinary FP16 output rounding. The difference is highly shape-dependent and repeatable. It implicates the dense GEMM execution selected for the 768-row shape; source/kernel root cause is not yet identified.

Artifacts: `/data/t011/artifacts/g023-qkv-fp32-compare.py`, `/data/t011/artifacts/g023-qkv-fp32-compare.log`, existing FP64 oracle `/data/t011/artifacts/g023-qkv-oracle.py` and `.log`.

## Exact-c57 logits and greedy impact

For aligned RID pair `g023-hit-branch-alpha-first-1794339480490170347` and `g023-miss-branch-alpha-first-1794339412136741981`, both sampler-input vectors have shape `[1,130049]`. 130,046 elements differ; max/mean/RMS absolute delta is `0.26229095 / 0.04113317 / 0.05153204`. Top-1 is token 1221 in both; top-2 is token 1376 in both. Hit top-1/top-2 margin is `0.24952292`; miss margin is `0.28782392`. Raw selected logits are -1.55807137 hit and -1.52851319 miss. Selected-token logprobs are -1.01814556 hit and -1.00248170 miss (delta -0.01566386). Both runs actually produced `[1221,12681]`; this two-token probe has no greedy ID divergence, despite full-logit and logprob differences.

This is the exact-c57 aligned first Prefill sampling evidence; it does not establish equivalence over longer generation or cache lifecycle. It quantifies that this particular logit delta did not alter the observed greedy output.

## Disposition

The anomalous 768-row QKV result warrants dense GEMM kernel investigation (not GPTQ). No production kernel fix has yet been made or validated. Keep G023/T001 BLOCKED and do not start performance qualification. If a defect cannot be isolated, do not silently relax the correctness gate; seek explicit acceptance criteria for numerical tolerances versus exact cached/miss behavior.
