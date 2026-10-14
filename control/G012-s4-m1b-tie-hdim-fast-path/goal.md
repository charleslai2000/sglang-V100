# Goal: S4-M1B preserve tied-head representation for Falcon SSU

## Authorized outcome

Enable the existing Triton `TIE_HDIM=True` specialization on the frozen Falcon-H1 c96 decode selective-state-update path solely by preserving a mathematically proven tied-head representation across the existing SM70 FP16→FP32 SSM dtype conversion. This is a representation/dataflow change, not a new kernel or equation change.

## Scope and constraints

- Use only assigned V100-SXM2-32GB and frozen qualified node21/layer0 fixture. Do not revisit graph capture, ABI recovery, or fixture extraction.
- Preserve exact FP32 recurrent state, FP16 model activation input semantics, all operand values, state indices, graph96 scheduling, and frozen S1 behavior/token correctness.
- No averaging, approximation, relaxed dtype, new kernel, scheduler/prefill/GPTQ change.
- FlashInfer M1A donor is closed/rejected and must not be reopened.
- Validate the existing specialization on exact fixture, benchmark representation and operator cost, then deterministic service correctness/A-B >=3 repeats and short profile/owner ledger only if gain is meaningful.

## Completion

Either accept a measured material end-to-end gain with exact fixture parity, token regression, c96 service A/B, intended TIE specialization on all 44 SSU nodes and updated node ledger; or reject the existing fast path based on correctness-qualified operator/service evidence and record the unchanged residual ledger/next owner.

## Result

DONE — rejected for production optimization. Root cause and the representation-only candidate were established; fixture output/state correctness passed and live c96 entrypoint calls proved zero-stride fast-path eligibility. Candidate service correctness passed in two 3-repeat waves, but no controlled paired service gain was established and latest n=30 fixture medians slightly favored FALSE (1.13357 ms vs TRUE 1.13562 ms). Do not integrate or profile; see `experiments/E021-s4-m1-mamba-decode-optimization/M1B-results.md`.
