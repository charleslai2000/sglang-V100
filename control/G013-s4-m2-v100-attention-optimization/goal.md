# Goal: S4-M2 V100 Falcon decode Attention optimization

## Authorized outcome

Reduce the measured c96 decode Attention cost represented by `_fwd_grouped_kernel_stage1` by identifying its exact production operation and comparing mature available V100 Attention donors, integrating only a correctness-qualified materially faster existing donor. If none is better, reject M2 and provide the precisely mapped next GPTQ residual target.

## Scope and constraints

- Only assigned V100-SXM2-32GB `gpushare-v100`; frozen S1 baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e`; no RTX 3090 access.
- Do not revisit Mamba SSU absent materially changed evidence; no prefill, scheduler/graph retuning, or new Attention algorithm before mature donors are exhausted.
- Preserve all Q/K/V and paged-cache semantics, dtype/model math, graph compatibility and output tokens.
- Donor order: existing sglang-V100 SM70 paths, pinned 1Cat V100 D128 paged decode, then other locally available upstream V100 donors.
- Qualification uses real Falcon c96 operands and complete output; timing includes all required adapters/fixups and full operation.
- Any accepted integration requires deterministic correctness, >=3 paired c96/1K/256 service A/B repeats, short batch96 Nsight node capture, and regenerated complete owner ledger.

## Completion

Accept only a materially faster, correctness-qualified donor with service A/B and updated owner ledger; otherwise reject after mature-donor comparison and map the next GPTQ reconstruction geometry/role precisely. If available donors cannot implement the production contract and evidence establishes a genuinely new SM70 Attention algorithm is necessary, report that bounded conclusion rather than inventing one.
