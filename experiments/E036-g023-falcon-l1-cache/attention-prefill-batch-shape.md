# Attention prefill batch-shape finding

## Scope

Real V100 service traces for the same 768-token prefix and 21-token suffix, FP16, Falcon-H1 GPTQ, `extra_buffer`, page size 1. Diagnostic `chunk=768` and eager runs are not exact-c57 qualification.

## Evidence

- At `chunk=2048`, the cache builder prefills 768 tokens, while a force-miss request prefills 789 tokens together. Layer-0 hidden vectors at positions 0 and 767 were identical; batched Q/K/V differed by FP16 rounding (at position 767: Q 536/1536, K 73/256, V 87/256 elements; max abs differences 0.015625/0.015625/0.0078125).
- A full layer-0 KV capture at the first decode step showed all 768 prefix rows differed between builder/cache-hit and force-miss; all 22 suffix/decode rows matched exactly. This points to prefix prefill batch-shape numerical differences, not a wrong KV-slot mapping. The hit prefix maps to the expected positions 0–767; the extra cache boundary slot is a separate allocator slot.
- At diagnostic `chunk=768`, builder and force-miss token-767 single-token QKV projections matched exactly. A 256-token eager/no-graph hit vs force-miss run produced identical greedy token IDs.
- At exact-c57 `chunk=2048`, production CUDA Graph remained enabled and 256 greedy IDs matched. Selected-token API logprobs differed (e.g. first: -1.0181456 hit vs -1.0024817 miss); one captured extend-logits row matched exactly, so those captures do not yet explain the reported logprob delta.

## Disposition

Likely cause is FP16 prefill batch-shape sensitivity at the cached prefix boundary. This is not a confirmed fix: diagnostic chunking/eager parity does not qualify exact-c57 production Graph, full logits, recurrent state/lifecycle, or requested lifecycle cases. G023/T001 remains BLOCKED; do not start performance qualification or claim readiness. Artifacts are under `/data/t011/artifacts/g023-{prefill-values,kvfull,final2048,768-256}-{hit,miss}.*`.
