# Isolated SM70 Prefill GPTQ GEMM experiment (not qualified)

The subsequent FP32-weight reconstruction iteration and controlled same-input first-error investigation are recorded in `prefill-first-error-closure.md`. They supersede the earlier FP16-weight prototype implementation, but do not retroactively qualify it or the current candidate.

This experiment is independent of the QPN2 Decode kernel and is disabled by default. The prototype reconstructs checkpoint Q4/group128 weights into FP16, invokes `cublasGemmEx` with FP16 inputs and FP32 output using `CUBLAS_COMPUTE_32F` / `CUBLAS_GEMM_DEFAULT`, and can apply the existing Falcon down multiplier as FP32 alpha. It does not mutate cuBLAS handle math mode or alter checkpoint parameters.

## Current V100 evidence

- Device UUID: `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`.
- Extension SHA256 after a clean rebuild from the currently recorded source: `0f173f35b137ea953f663f2b235427f0b66fc8fd7c55238e900007b0bd1612f9`; current source SHA256: `723e5ac73bcd4e5900dbf66def6167412b2c09269fbaaeba9a01957c6424bd05`. The numerical log below was produced by the functionally equivalent prior build SHA `fc6706d64a9511e1f2de191c7bcf6bb02f7b126c829af765603e5bb37c7c5331` (explicit local algo variable added afterward); do not transfer those numerical results to the rebuilt identity.
- Experiment script: `/data/t011/artifacts/t011-cublas-experiment.py`, SHA256 `17acbdabfa78f9f31041007dc277b0c78537fcf7a065b5f3bf103ec8b1422b67`; numerical log SHA256 `5d8af6bd0f2f25073b53e811c45e4f2974b3f05c7b365f950869ff4208103b69`.
- A newly initialized T011 worker on deterministic 512-token input still has non-finite upstream activation at `model.layers.43.mlp.down_proj` (`nan` at `[229,175]`), so these measurements isolate the candidate GEMM on a corrupted activation and do not constitute end-to-end repaired Prefill evidence.
- Independent FP64 raw GPTQ dot at L43 `[229,175]` is `77708.75079141936`; raw FP16 cast is Inf, while applying the existing multiplier first gives `2529.5817314915157` and FP16 `2530.0`.
- FP16 reconstructed-weight dot oracle at that coordinate is `77719.7878450187` (rounding delta `11.03705359934`, relative `1.42031e-4`). Other sampled coordinate relative deltas include near-zero cancellation at `[229,176]` (`-0.00793` relative); this is evidence of material FP16 reconstruction rounding, not a pass under any frozen strict oracle.
- Candidate `alpha=1` returned FP32 `77719.859375` at the overflow coordinate, finite output, 512-row call `10.62 ms`; candidate alpha=down_multiplier returned FP32 `2529.943115234375`, finite output, 512-row call `0.797 ms`. The first-call/reconstruction/timing environment is uncontrolled and this does not include weight-materialization cost as deployed across layers. No baseline-matched Prefill 512/1K/4K comparison or service performance has been measured.
- Peak PyTorch allocation during the isolated experiment was about 13.1 GB (whole worker already loaded); this is not a capacity qualification or safe-headroom result.

## Disposition

The prototype proves the requested API route can keep the sampled raw output finite in FP32 and can scale before a later cast. It does **not** yet satisfy numerical acceptance: FP16 reconstructed weights diverge from the independent unrounded FP64 GPTQ oracle, and even sampled near-zero coordinates show substantial relative difference. Full exact reconstructed-rounding oracle, all relevant projection boundaries, corrected full Prefill, Decode/Graph/state correctness, peak VRAM, and frozen service A/B are incomplete. Keep the candidate experimental and disabled; do not route production to this path or claim READY.
