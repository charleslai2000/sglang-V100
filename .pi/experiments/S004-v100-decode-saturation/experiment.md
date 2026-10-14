# S004 — V100 decode saturation and CUDA-Graph coverage optimization

## Frozen identity and scope

- Falcon correctness baseline: `07343bd165855e9cd35ae63acea6cee5819c8c4e`.
- Execute only on `gpushare-v100` Tesla V100-SXM2-32GB using frozen source/runtime extraction `/hy-tmp/sglang-V100-s2-07343bd165` and pinned checkpoint revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- RTX 3090 is fixed reference only; no access/build/run/profile/requalification.
- No new kernels or model math. Prefill graph optimization deferred.
- Keep local unrelated dirty files untouched; all new S4 material lives under `.pi/experiments/S004-v100-decode-saturation/`.

## Objective

Determine V100 steady-state decode throughput saturation by increasing concurrency and maximizing graph replay coverage. Record aggregate/per-request tok/s, ITL, graph/eager pass deltas, actual scheduler batch histogram, VRAM, power and utilization. Compare graph sets from observed distribution; use smallest set covering useful operating region. Add/validate only low-overhead framework/CUDA-event telemetry; no repeated Nsight injection or privileged NCU counter attempts.

## Workload and controls

~1K prompt, 256+ generated tokens, radix disabled initially, identical correctness harness; c=1,2,4,8,12,16,24 as memory/scheduler permits, 3+ stable correct repeats. Every graph capture set must cover target batch sizes; if 24 is impossible, record concrete failure and proceed within resource envelope. Separate config effects from batch-cap effects. Check output correctness each run.

## Historical accepted anchor and references

S3 graph-set A/B on 32GB V100, c8/1K/256: capture `[1,2,4]` at max-running=8 = 75.72 tok/s, decode graph/eager passes 0/256 per repeat; capture `[1,2,4,8]` = 154.00 tok/s, 255/0, 256/0, 256/0. Improvement +103.4%. Outputs correct; prompt remained eager. Mamba marginal state = 0.1320 GiB/logical request plus one fixed 0.1320 GiB dummy slot. Full result and raw hashes in sibling `../E021-s3-v100-optimization/results.md`.

Frozen 3090 historic references: c8 371.2 tok/s, c12 485.7, c24 peak 495.1, steady B16 459.53. They are not fresh comparisons.

## Results

Pending. Store scripts, raw data manifests, and final S4 report in this directory.
