# G021 — R17 Inference Performance Metrics V1

## Authorized outcome

Execute and disposition the frozen **LLM Inference Performance Metrics V1** contract for Falcon-H1-7B + SGLang on Tesla V100-SXM2-16GB, integrating G018–G020 and G021 data. Publish a provenance-rich matrix and complete research report. This is a performance measurement system, not a warranty system. See `design/LLM-Inference-Performance-Metrics-V1.md` for the authoritative A–G contract and ten required final answers.

Completion means every metric/question is MEASURED, PARTIAL, DISABLED, UNSUPPORTED, NOT_RUN, or INVALID with evidence and boundaries. Missing actual speed, time attribution, or hardware utilization prohibits claiming the corresponding bottleneck analysis complete.

## Frozen authority and boundaries

- G017 ACCEPTED; S1 default; Candidate exact-path/SHA opt-in only. Preserve prior G018–G020/G021 data and claims accurately.
- Only target V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; RTX 2060 UUID `GPU-04ad2dd9-245d-0e7b-c96a-12da83a92b99` and unrelated workloads excluded.
- No GPTQ/Attention/Mamba/Graph math changes, new compute kernels, host-wide profiler/policy/CUDA/PyTorch/driver changes, global profiling ownership, or interference with other GPUs. Keep GV100 NCU working. Root target-only profiling and ordinary measurement/build/IPC/process wiring are authorized when safe.
- Reuse archived evidence. 3090 remains independent until workload, batch, cache, model and timing boundaries align; do not mix different batch data in a direct comparison.
- NVML busy is device activity only. Distinguish kernel/Graph/device boundaries. Never fabricate unsupported cache/state functionality, events, percentiles, utilization, or owner attribution. Unsupported is not zero; Output TPS is not complete inference performance.

## Completion

Complete and disposition sections A–G and all ten final questions in the frozen design contract. Preserve provenance for every row; publish unified matrix, raw artifact pointers, scripts, reproducible configs, artifact manifest and report. Commit/push evidence phases independently and verify remote SHA. G021 stays ACTIVE until the completion condition is genuinely met; current partial matrices alone are not acceptance.
