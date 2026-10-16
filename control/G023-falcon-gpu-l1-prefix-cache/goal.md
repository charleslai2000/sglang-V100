# G023 — Falcon GPU L1 prefix reuse

## Authorized outcome
Deliver or reject an explicitly enabled, correct Falcon-H1 GPU-resident prefix cache using the existing MambaRadixCache, HybridReqToTokenPool, and Falcon cache wiring. Compare OFF, ON/cold, and ON/effective warm reuse through real service requests, including TTFT, throughput, resource cost, and rollback. This follows G022's separate GPU decode optimization work; it does not replace that goal's decode completion criterion.

## Scope and constraints
- Existing main checkout and worktree only; preserve the unowned packed-word dirty file and all unrelated modifications. No new branch/worktree.
- Target only V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; host MemAvailable must stay >=8 GiB; never use RTX2060. Use the existing UUID lock and isolated process/runtime.
- Reuse the local MambaRadixCache and HybridReqToTokenPool. No new cache architecture, host/disk tiers, GPTQ/attention/Mamba arithmetic changes, quantization/tolerance/precision changes, response cache, output replay, or reduced real concurrency.
- S1 default and exact c57 artifact remain unchanged. Do not reopen E035 candidate routes or alter their rejection status.
- A valid hit must bind same token prefix/positions/model config and matching Attention KV plus Mamba conv/SSM state at the same computed boundary. Reuse existing stream/event ordering; shared cached state must be immutable and request continuation must COW to a private writable slot. Incomplete state must fall back to a valid earlier boundary or recompute.
- Preserve existing raw evidence. Do not claim READY unless all correctness, resource, and fixed performance criteria pass. Otherwise report a precise NO-GO/blocker. Owner acceptance remains Owner-only.

## Completion
Record implementation investigation and exact changed files; prove state ownership, checkpoint boundary and exact state copies with real ordinary requests; compare each request and logits/greedy output against identical-input uncached execution; cover branching, chunking, continuation, slot reuse, pressure/eviction/re-request. Run the authorized alternating three-arm service matrix at 1K/4K, c1/c2/c4, 768/3072 shared prefixes plus unique suffixes and a no-shared control, with matching model/config/binary and cache semantics. Report cache build cost, hit/reused tokens, residual prefill, actual batch/Graph key, per-request Queue Wait/TTFT/ITL/E2E, GPU Event Prefill/Decode, copy bytes/time, allocator usage/VRAM and all limits. Commit/push scoped implementation, tests, report and control; keep other dirty changes out.
