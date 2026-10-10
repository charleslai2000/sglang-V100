# Current frontier

## G018-local-v100-falcon-h1-gpu-audit
- T001 Profile ledger and resource audit — ACTIVE
  - frontier: final audit is evidence-backed PARTIAL; commit scoped G018 report/assets and sampler correction, push dedicated ref, verify remote SHA/clean tree. Do not claim uninstrumented A/B or owner budget; missing 4K c2/c4 FPM, worker allocator, and graph owner timing/counts stay UNKNOWN.
  - task: `G018-local-v100-falcon-h1-gpu-audit/tasks/T001-profile-ledger-and-resource-audit.md`
  - unblock/reopen: G017 base and exact S1/c57 pins verified. Existing root NCU authority collected isolated GPTQ fixture counters only; not model-owner evidence. DCGM unavailable, full-forward NCU/owner ledger/worker allocator and matched 4K c2/c4 FPM remain UNKNOWN. No global policy changes, no RTX2060 target, no production kernel change, full-service Nsys remains DEFERRED.
