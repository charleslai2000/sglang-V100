# Attention KV builder-to-hit consistency

## Findings

A full first-decode layer-0 KV capture from one cache-hit run contains both the 769-token builder and 790-token hit request. The first 768 K rows and first 768 V rows are byte-identical between these snapshots (768/768), directly showing the shared prefix persisted through builder completion and was read back by the hit request. This complements the Mamba checkpoint trace, where tracked source/destination slot fingerprints agree after tracking and branch COW destinations match the retained checkpoint.

Comparing the same hit request with the aligned force-miss first-decode request (same 789-token input; 790 KV rows including generated token), K and V suffix rows 768–789 are identical (22/22 each). Prefix rows 0–767 differ (0/768 identical): K max/mean absolute delta `0.00048828125` / `2.03e-5`; V `0.0078125` / `2.09e-4`. This matches the earlier measured builder-vs-789-row miss Prefill QKV batch-shape difference. It does not indicate a bad cache transfer: the hit prefix exactly equals its own builder KV. But it demonstrates cache-hit versus force-miss attention state is numerically unequal under different Prefill batch shapes.

## Disposition

This verifies builder→cache→hit persistence and rules out a KV row-copy/slot-mapping discrepancy for this exact serial case. It does not provide a same-shape independent attention oracle, nor establish that batch-shape-dependent FP16 differences satisfy the required cached-vs-miss correctness criterion. The independently measured Mamba checkpoint transfer passes its current probe, but it does not resolve the attention KV/logit difference. G023/T001 remains BLOCKED; no lifecycle expansion or performance matrix started.

Artifacts: `/data/t011/artifacts/g023-kvfull-hit.jsonl`, `/data/t011/artifacts/g023-kvfull-miss.jsonl`; comparison script `/data/t011/artifacts/g023-kvfull-compare.py`.
