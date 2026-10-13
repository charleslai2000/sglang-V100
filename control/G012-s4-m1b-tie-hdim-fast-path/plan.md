# Plan: S4-M1B tied-head Triton fast path

## Established evidence

- Frozen c96 S1 service: `391.20 ± 0.98` aggregate tok/s, n=3. Node ledger in G010/S004: 44 SSU calls/forward at ~1.006 ms/call, 31.85%; next grouped decode-attention family `_fwd_grouped_kernel_stage1` ~22.46%, not Mamba.
- M1A FlashInfer STP-simple donor closed/rejected; do not revisit.
- Frozen node21/layer0 fixture hash `6d4d8d52af09fab253650baaae737dd60f4316568454ca1b3f222cd94ffe051f`. Full fixture FALSE-vs-TRUE comparison: all 75,497,472 state values exactly identical; both output have same captured-production error envelope (max abs `1.22070e-4`, rel-L2 `5.52548e-8`).
- Root cause: FP16 per-head dt was expanded then cast to FP32, materializing the repeated dimension; cast-before-expand preserves exact FP32 values at zero stride. A/dt_bias/D were already tied. The local source change is representation-only; focused regression added; direct equivalent checks passed, pytest unavailable.
- Live c96 production-entrypoint audit in isolated candidate runtime logged 8,778 invocations, all with dt `(24,1,0)`, A `(1,0,0)`, dt_bias `(1,0)`, D `(1,0)`, FP32. Thus the existing tied-head route activates in actual graph construction/replay calls. No graph identity/ABI change.
- Two candidate c96/1K/256 three-repeat waves all correct, each 96×256 outputs: first TPS 401.273/400.034/399.090; audited TPS 400.976/400.206/398.917. Service-info histogram was empty; logs confirm graph-true c96 decode. Against historical frozen 391.20±0.98, the difference is unpaired and not an A/B result.
- Latest fixture CUDA-event n=30 medians FALSE 1.13357 ms, TRUE 1.13562 ms. Preliminary n=20 faster TRUE numbers were not stable and are superseded. No controlled speed gain is established.

## Disposition

M1B REJECTED as a production optimization on available evidence. Do not integrate the candidate, do not claim service improvement, and do not run a residual profile. Keep the source edit/regression as an experimental patch only; do not change the frozen runtime or owner ledger. Candidate service stopped; V100 idle at 1 MiB/0%. Evidence and caveats are in `experiments/E021-s4-m1-mamba-decode-optimization/M1B-results.md`.
