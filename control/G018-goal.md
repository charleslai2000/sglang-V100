# G018 — Local V100 Falcon-H1 GPU utilization & decode bottleneck audit

## Authorized outcome

Quantify the local Tesla V100-SXM2-16GB resource use and decode-time ownership for Falcon-H1-7B GPTQ, distinguishing prefill/TTFT, steady decode/ITL, forward/Graph replay, device and host gaps. Produce reproducible workloads, resource telemetry and a bounded owner ledger, then recommend next optimization owner with measured evidence.

## Authority and constraints

- Base: accepted G017 commit `149c70f10b49ff1d1fe28ac5a7016526157a04a9`.
- Local V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189` is the only performance authority. Never target or profile RTX 2060 UUID `GPU-04ad2dd9-245d-0e7b-c96a-12da83a92b99`; do not disturb unrelated GPU jobs/services.
- S1 remains default; G017 candidate only with explicit opt-in and exact final binary SHA. Preserve checkpoint, GPTQ, Attention, Mamba, CUDA Graph and configuration. No kernel implementation changes.
- Full-service Nsys capture stays deferred. Targeted NCU was tested on a standalone V100 GEMM and denied with `ERR_NVGPUCTRPERM`; `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly:1`. Do not alter system-wide profiling mode or restart host/driver; NCU counter evidence is UNKNOWN. DCGM/DCGMI absent/inactive; do not start a system service.
- Existing production source/worktrees are not modified for instrumentation. G018 work occurs on a new clean branch/worktree at the accepted G017 commit. Add only instrumentation/report artifacts required to make results reproducible.
- No metric may be represented as zero when not collected. Unknown boundaries, ownership and unavailable metrics stay UNKNOWN.

## Completion

Reproducible S1/candidate prefill/decode/service matrix (1K/4K, c1/c2/c4, 256 greedy outputs); measured device-resource telemetry with exact capture windows and availability caveats; c1/c4 complete model-forward call inventory and matched GPU time ledger with GPTQ gate_up/down, Attention, Mamba, other owners, and residual; targeted matched-metric NCU evidence for main kernels; evidence-ranked bottlenecks and defensible recoverable budget; artifact/source/tool manifest; independent scoped commit, push to dedicated ref, remote SHA equality and clean G018 worktree.

The ledger must reconcile all accounted owners and label uncovered time UNKNOWN; no partial layer-fixture timing may be promoted as a full-model total. No production kernel change or default change is allowed.

## Result

PARTIAL / ACTIVE. G017 authority accepted at the frozen commit. Instrumented service and whole-forward FPM windows exist, but no uninstrumented throughput authority, no matched 4K c2/c4 FPM, no worker-local allocator stats, and no full owner ledger/recoverable budget. Existing root NCU authority was reused for a matched isolated GPTQ fixture only; fixture counters do not establish Falcon owner shares. DCGM absent/inactive; full-service Nsys DEFERRED. No production kernel/default change. Report and manifest are in `experiments/E027/`; commit/push still outstanding.