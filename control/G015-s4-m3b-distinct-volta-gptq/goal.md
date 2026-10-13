# Goal: S4-M3B distinct mature Volta GPTQ qualification

## Authorized outcome

Find and qualify a distinct mature SM70 GPTQ implementation preserving frozen Q4/group128/GPTQ-v1/desc_act/nonidentity-g_idx semantics, or exhaust mature compatible candidates and establish a new SM70 kernel/design decision is required. Continue through source audit, real-weight parity/stress, fully-accounted timing, integration only if correct and materially faster, paired service A/B and refreshed authoritative c4 ledger until resolved.

## Scope and constraints

- Assigned V100-SXM2-32GB `gpushare-v100` only. Do not use RTX 3090.
- M3A and M3B are closed/rejected. Do not reopen Mamba or Attention. Frozen production use_shuffle=false remains baseline. No further donor shopping; a new SM70 GPTQ kernel requires separate authorization.
- Audit vLLM legacy Volta, lineage-compatible older vLLM/QwenLM distinct SM70 kernels, then AutoGPTQ CUDA-old only if exact semantics match. Exclude Marlin on SM70.
- Reject source-equivalent donors without redundant benchmarking. Do not drop/canonicalize/approximate g_idx or change GPTQ semantics.
- Reuse E024's six real c4 fixtures and the historical S1 acceptance evidence. No production changes until donor qualifies; preserve fallback and CUDA Graph compatibility if integration occurs.
- Do not stop on ordinary build/ABI/layout/benchmark issues; resolve within authorized scope. If identical correctness failure repeats twice, stop gating and directly diff packed tensors, source, and launch data.

## Completion

Either (a) M3B ACCEPTED with donor identity, distinction proof, real-fixture/reference/high-amplitude/token correctness, complete operator timing, c1/c2/c4/c96 A/B and refreshed c4 ledger; or (b) M3B DONOR REJECTED with mature candidates exhausted/source-equivalent, parity/timing evidence, remaining 51.655% GPTQ share quantified, and evidence a new SM70 GPTQ kernel is next architectural decision.
