# G020 — Existing CUDA Trace Offline Attribution

## Authorized outcome

Analyze only archived local V100 Falcon decode traces from G018/G019 and existing service/FPM authority. Recover reproducible, conservative kernel-category durations/counts/overlap by 1K/4K × c1/c4 for S1/c57 where trace annotations allow. No new GPU test, profiler, instrumentation, production source change, or kernel optimization.

## Constraints

- G017 ACCEPTED. G018/G019 CLOSED PARTIAL. Base G019 commit `db08f603033ccebb252ed8f8cb209b2d0203eda1`.
- V100 local 16GB traces only. Do not import 32GB remote results.
- Classify only where names, geometry, call pattern and source uniquely support identity; otherwise UNKNOWN.
- Kernel duration sums may overlap; separately compute union and per-owner exclusive overlaps. Do not infer full-device utilization from process kernel intervals.
- Explicitly distinguish trace-window kernel-category costs from whole-forward owner attribution. Stop when archived traces exhausted.

## Completion

Provide reproducible offline analyzer, raw artifact/hash manifest, per-window owner call counts/duration sums/union/overlap/UNKNOWN, source map and coverage limitations; compare S1/c57 and G018 FPM/service only where boundary and workload match; commit/push and verify remote SHA.
