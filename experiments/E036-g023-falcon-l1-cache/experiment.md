# E036 — Falcon-H1 GPU L1 prefix reuse

## Question
Can the repository's existing MambaRadixCache safely retain and reuse Falcon-H1 Attention KV plus Mamba conv/SSM state for an effective same-prefix warm hit, without regressions that violate the frozen service criteria?

## Hypothesis
Existing hybrid radix cache and Mamba state COW may enable effective GPU-resident prefix reuse when configured in its supported mode. This is unproven until ordinary requests create real checkpoints, state ownership is audited, and exact-c57 service runs show measured hits and TTFT benefit.

## Experimental object and identity
- Repository: current `main`, baseline `3b4393c63d943cf890beba38ac9d7c7bfde47d24`.
- Model: `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`, revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- Main compute arm: exact c57 `common_ops` SHA256 `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`; S1 SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842` preserved as separate baseline/rollback.
- GPU restricted to UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; host MemAvailable >=8 GiB. Do not use RTX2060.

## Controlled/varied inputs
Cache arm OFF, ON/cold after logical reset, and ON/real warmed hit; three alternating blocks/triplets. Same c57/model/compute/attention/dtype/scheduler settings. Inputs are frozen token IDs: 1K total with shared prefix 768, 4K total with shared prefix 3072; per-request suffixes differ; separate no-share controls. c1/c2/c4, 256-token greedy output, fixed seed; warm-up and cache-builder requests recorded separately. No response cache or output replay.

## Procedure
1. Static trace of local Falcon model config → cache builder → registry → MambaRadixCache and HybridReqToTokenPool → scheduler checkpoint tracking → deferred forward-stream COW/clear → state-slot release/eviction. Establish safe cache mode and computed state boundary before GPU.
2. Use ordinary service requests to create checkpoints. Verify exact positions, token/model identity, Attention KV ownership and Mamba conv/SSM bytes/slot; fork request state, preserve checkpoint bytes, compare same-input cached and uncached logits and 256-token IDs. Exercise distinct suffix, concurrent branches, chunk prefill, continuation, cancel/completion reuse, capacity eviction/re-request. No direct cache injection.
3. If correctness passes, run identity-locked alternating OFF/cold/warm service comparisons; connect client RIDs to existing FPM/Event/trace; report actual GPU batch and CUDA graph key, copy traffic/time separately from KV references, queue, TTFT, GPU prefill/decode, ITL/E2E, memory, warmup build cost and amortization. Do not substitute scheduler wall for GPU time.
4. Freeze all raw logs and SHA manifests; no production default change. If any correctness or performance gate fails, report bounded NO-GO.

## Measurements needed
Checkpoint boundary and exact state identities; cache hit tokens and remaining prefill; COW/copy bytes and GPU duration; per-request correctness; three-arm cache hit behavior and service metrics; actual running batch/Graph key; pool/VRAM occupancy, eviction and build/amortization costs.
