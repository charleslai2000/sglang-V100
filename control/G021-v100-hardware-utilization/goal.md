# G021 — R17 V100 hardware utilization measurements

## Authorized outcome

Close the required real Falcon-H1 decode hardware utilization evidence on the local Tesla V100-SXM2-16GB: device busy/resource telemetry and, only if permitted without global policy changes, actual-workload NCU hardware counters for representative kernel/Graph boundaries. Conditions: 1K/4K input × c1/c4, stable decode. Continue the broader R17 TTFT, cache, and E2E acceptance only after this bounded hardware task.

## Authority and constraints

- The only measurement target is V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`. Explicitly exclude RTX 2060 UUID `GPU-04ad2dd9-245d-0e7b-c96a-12da83a92b99`; do not query/profile it or disturb its workload.
- G017 accepted production implementation and default remain unchanged. Use the accepted S1 implementation unless an already-authorized candidate condition is required for R17; any mode must be path/SHA pinned. Do not alter model, GPTQ, Attention, Mamba, CUDA Graph semantics, global profiling mode, CUDA/PyTorch versions, driver, or host state.
- Use existing NVML/nvidia-smi and G016 validated root NCU path only. No new profiler framework. Full-service Nsys remains deferred. No global NCU permission or profiling configuration changes.
- Instrumented measurements are not normal service throughput authority. Keep per-kernel, per-Graph, and whole-device windows separate. Never promote a kernel metric to graph or device utilization.
- DCGM investigation is read-only; do not start/enable host-wide DCGM or change resource ownership.
- If profiling privilege or resource isolation prevents safe actual-service counter collection, record exact command, error, source/tool evidence, and stop that route rather than changing host policy.

## Completion

For stable decode at 1K/4K × c1/c4, provide timestamped, UUID-targeted NVML samples with explicit windows and units for GPU busy, memory busy, memory use, SM/memory clocks, power, temperature, and throttle. If permitted, use root NCU on real Falcon decode/representative actual Graph kernels for SM active/throughput/occupancy, Tensor/FP16/FP32 pipeline, DRAM bytes, L2 traffic, duration, stalls and issue activity. Compute HBM effective GB/s from measured DRAM bytes / GPU execution duration only at the matching kernel or Graph boundary, and compare with V100 theoretical bandwidth; label whole-device utilization separately. Check DCGM install, permissions, field support, and profiling ownership with concrete evidence. Any unmeasurable required metric must have a reproducible reason and precise scope. Preserve G017 production state. Continue R17 TTFT/Cache/E2E acceptance only after this task.
