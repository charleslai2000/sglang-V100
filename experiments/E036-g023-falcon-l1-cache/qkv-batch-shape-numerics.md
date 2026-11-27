# Falcon-H1 QKV Prefill batch-shape numerics

## Projection identification and oracle

The layer-0 Falcon-H1 attention projection is `QKVParallelLinear` using `UnquantizedLinearMethod`, with FP16 weight shape `[2048,3072]`; it is not GPTQ. Using the actual identical position-767 hidden row from the aligned hit-builder and force-miss Prefill traces, the QKV projection was independently rerun with batch row counts 1, 8, 21, 768, and 789. All five calls were individually repeatable for the same input and shape. A float64 dot-product using the loaded FP16 checkpoint weight, rounded to FP16, was used as reference.

The actual cache-builder trace (768 Prefill rows) reproduces its captured QKV: 698 output elements differ from the FP64-rounded reference, max absolute delta `0.015625`, mean `0.000809`. The actual miss trace (789 Prefill rows) reproduces its captured output: 16 elements differ, max `0.0078125`, mean `1.36e-5`. For the shared row, batch sizes 21 and 789 produce the same output; batch sizes 1 and 8 also match that result. Batch size 768 yields a different result. Across all shapes, repeat calls were exactly equal.

## Interpretation and unresolved logits

The result is deterministic shape-dependent FP16 GEMM arithmetic, not call-to-call nondeterminism. Batch 789 agrees much more closely with the FP64-rounded dot-product than batch 768, while the 768-row path deviates more substantially. The oracle is a mathematical reference but not a proof of implementation contract: without an independent trusted cuBLAS/GEMM implementation run on the exact input and FP16 weight, it cannot distinguish valid shape-specific accumulation/rounding from a V100 kernel defect. It does establish that the captured attention QKV discrepancy is reproducible by the unquantized dense projection alone.

The numerical effect reaches cached attention state: builder→hit prefix KV rows are exactly identical, but hit and force-miss prefix KV differ because their QKV-producing Prefills used 768 and 789 rows. On the subsequent first decode token, the final input hidden and layer-0 QKV match, while attention output differs (max `4.88e-4`, mean `1.81e-5`); layer-43 input differs by max `1.0`, mean `0.0580`. The sampler's first selected raw logit differs (`-1.5285132` hit vs `-1.5580714` miss), producing selected-token logprobs `-1.0181456` vs `-1.0024817`. Thus the Prefill KV delta propagates through attention and later layers into logits; the full-vector delta was 130,046/130,049 elements, max `0.26229095`, mean `0.04113317`. No acceptable tolerance or correctness waiver has been established.

## Disposition

QKV is unquantized FP16; GPTQ changes are not an explanation for this attention difference. Numerical acceptability and independent kernel correctness remain unproven. G023/T001 remains BLOCKED; do not start performance qualification.

Oracle artifacts: `/data/t011/artifacts/g023-qkv-oracle.py`, `/data/t011/artifacts/g023-qkv-oracle.log`; aligned source traces `/data/t011/artifacts/g023-layer-samples-{hit,miss}.jsonl`; exact-c57 logits/logprob evidence is documented in `logprob-dataflow-finding.md`.
