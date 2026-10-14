# Goal: S4-M1 V100 Mamba decode optimization

## Authorized outcome

Improve the frozen Falcon-H1 S1 c96 / 1K prompt / 256 output V100 decode path by adapting a mature, correctness-preserving Mamba decode donor implementation, then demonstrate an end-to-end serving gain. The user explicitly authorizes Mamba implementation and service A/B for this milestone. GPTQ is mapping-only; no graph scheduler or prefill work.

## Scope and constraints

- V100-SXM2-32GB only; do not access or profile the RTX 3090.
- Preserve frozen S1 serving/model semantics, c96 graph96 scheduling, output tokens, FP32 temporal state, FP16 convolution state, state ownership/index semantics, and frozen throughput evidence.
- Donor-first: inspect current SGLang Triton SSU, state-spaces/mamba selective_state_update, current upstream SGLang, FlashInfer SM70 implementation already available, and any SM70-specific Mamba implementation in the V100 tree. Port mature deltas before considering new algorithm design.
- No approximation, dtype relaxation, or CUDA graph scheduling changes. Only Mamba decode operators; grouped state/layout may be changed with selective scan if source evidence establishes an inseparable mature fusion.
- Run exact-geometry operator correctness/performance before service A/B; then full c96 correctness and service A/B; after accepted change obtain short node-level Nsight residual ledger.
- Do not optimize GPTQ; map its kernel geometry/role as bounded parallel source analysis only.
- Stop only on demonstrated unsuitable donors, a genuinely new-algorithm requirement, or physical V100 limitation. Ordinary implementation/build/integration defects are to be resolved, not treated as blockers.

## Completion

An authorized mature-donor implementation or evidenced true blocker; exact current/donor source and real Falcon c96 geometry recorded; operator before/after timings and state/output differential; correctness through fixed 1K/256 c96 serving; unprofiled aggregate tok/s before/after with repeats; short post-change node-level report and updated owner ledger; exact GPTQ projection-role/geometry map; top residual and next target identified. No ceiling claim.

## Result

ACTIVE. Prior frozen evidence and node ledger are in G010/S004. No source implementation changes have yet been made for M1.
