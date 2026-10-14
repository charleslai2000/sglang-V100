# E021 — S3 V100-only Falcon-H1 performance optimization

## Question

Using only the established V100 S1 Falcon-H1 serving path, which decode and prefill costs are software-addressable, and what profiler-backed optimization yields a repeatable throughput gain? How does the remaining V100 gap compare with the fixed RTX 3090 reference supplied by the user?

## Hypothesis

The S2 graph-enabled peak (~75.8 aggregate tok/s) may be limited by a combination of uncovered/fallback decode batches, GPTQ/Mamba/attention device work, and host scheduling. Graph batch coverage or mature upstream prefill graph support may recover some software loss; quantify before changing.

## Identity and controls

- V100 only: frozen source/baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e`, checkpoint revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`, SGLang V100 environment and S1 backends. Reuse S2 artifacts/results and do not rebuild correctness baselines.
- RTX 3090 is a fixed external performance reference only: decode-heavy aggregate c1/c4/c8/c12/c24 = 64.8/206.6/371.2/485.7/495.1 tok/s; agent-like c8/c16/c24 = 224.6/352.2/396.3; steady B16 = 459.53 tok/s. No 3090 run, port, requalification, or implementation work is authorized by this experiment.
- Keep prefill and decode separate; no profiler run contributes throughput numbers. Preserve unrelated V100/local processes and coordinate shared-device windows.

## Procedure

1. Inspect S2 source, harness, raw logs, graph runs and existing profiler captures; avoid duplicating established results.
2. On V100 only, capture clean Nsight Systems for warmed steady decode (about 1K context; concurrency 4/8/plateau, graph enabled) and eager prefill at about 1K/4K/8K. Quantify busy/idle, CUDA graph coverage, launches/token, gaps, synchronization, owners/counts/durations.
3. Profile only dominant owners with targeted Nsight Compute or device-counter alternative: GPTQ, Mamba, attention, conversions/elementwise. Record achieved SM/tensor/DRAM/L2, occupancy, stalls, registers/shared memory and roofline where available.
4. Diagnose graph fallback/padding from actual batch distribution. A/B only evidence-selected graph batch sets and mature SGLang prefill graph mechanisms. Keep separate decode/prefill matrices and correctness gates.
5. Change only the isolated SGLang-V100 benchmark tree; prefer existing donors/configuration. Re-measure unprofiled with repeated workloads; preserve exact diffs and before/after.
6. Reconcile S2 Mamba bytes from source formula and controlled request-count deltas if safe/available; update 16GB projection explicitly as an estimate. Source-derived Falcon params at TP1: each logical slot 44×(24×128×256×4 B temporal FP32 + (12288+2×256)×3×2 B conv FP16)=141,791,232 B = 0.1320 GiB; allocator also has one dummy/padding slot, so configured S logical slots allocate S+1 physical states. Delta per extra logical slot is 0.1320 GiB; compare pool logs with allocator tensor sizes, not only process/device-used deltas.
7. Report only upon PASS or evidence-supported BLOCKED under the user's milestone criteria, including realistic software/hardware decomposition vs fixed 3090 reference.

## Measurements

Repeated before/after decode and prefill throughput/TTFT/ITL, correctness, resource use; Systems time ownership and graph coverage; targeted hardware counters; exact source/config/diff; Mamba state deltas; projected 16GB aggregate throughput.
