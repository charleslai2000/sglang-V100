# G023 Mamba checkpoint lifecycle evidence

## Evidence

A fresh serial production-cache probe ran the exact shared 768-token prompt followed by alpha, repeated alpha, bravo, then alpha again, with Falcon-H1 GPTQ, `extra_buffer`, chunked prefill 2048, one request at a time, and CUDA Graph disabled for trace safety. The probe validated 768 cached tokens on each follow-up and got identical Greedy IDs and first-token logprobs for repeated alpha before/after bravo. This is a same-service repeat/interleave check, not a force-miss equivalence or full lifecycle qualification.

The builder request `g023-hit-cache-builder-1794387787518067893` used `track_seqlens=[768]`, `mask=[true]`, source slot 7 and destination slot 8. Before tracking, destination slot 8 had the initial-state fingerprints; after the mixer, both source and destination had the same updated convolution and SSM fingerprints. The tracked destination therefore received the builder's post-prefix recurrent state.

Each subsequent cache-hit branch (alpha first, repeated alpha, bravo, alpha after bravo) COW-copied from retained slot 8 into a private slot (10, 13, 1, 4 respectively). Before each copy, source slot 8 had the same checkpoint fingerprints. After each copy, the private destination matched source in both convolution and SSM fingerprints. The request's following Prefill forward used that private slot. The evidence supports that this exercised cache checkpoint propagation and that the observed interleaved request did not mutate the retained checkpoint.

## Limits and disposition

These fingerprints do not independently establish state numerical correctness against a same-shape force-miss oracle, do not cover cancellation, eviction/reuse, concurrent branches, or CUDA Graph replay, and do not resolve the Prefill batch-shape logits mismatch documented in `logprob-dataflow-finding.md`. The probe's matching outputs/logprobs are repeatability within cache-hit mode only. G023/T001 remains BLOCKED; do not start performance qualification.

Trace: `/data/t011/artifacts/g023-state-lifecycle-track-final.jsonl`; probe result and full request records: `/data/t011/artifacts/g023-state-lifecycle-track-final.json`; comparison helper: `/data/t011/artifacts/g023-state-lifecycle-compare.py`.
