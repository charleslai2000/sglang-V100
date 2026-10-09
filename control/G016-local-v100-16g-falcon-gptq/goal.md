# Goal: Local V100-16GB Falcon-H1 profiling authority

## Authorized outcome

Qualify the local Falcon-H1-7B SGLang runtime and targeted root NCU capability on the Tesla V100-SXM2-16GB as authority for subsequent S4-M3C small-M GPTQ work. Nsight Systems Falcon serving capture is explicitly deferred and not a READY gate; retain forensic report `Nsys-recovery.md` and the proven 4GB staging mechanism. Do not repeat worker-injection experiments under this Goal. Capture limitations do not block targeted NCU-based optimization. Record local 16GB memory/headroom and bounded short stability evidence; do not claim long-soak completion.

## Scope and constraints

- Work on the local Tesla V100-SXM2-16GB only; preserve isolation from the desktop RTX 2060 and unrelated worktrees.
- This goal supersedes remote V100-32GB/RTX 3090 profiling as authority for local 16GB capacity and performance conclusions. Historical remote evidence remains context, not local measurement.
- Before any performance conclusion, restore the full pinned Falcon GPTQ model, runtime, source revision, and compatible dependencies; validate correctness against the frozen Falcon serving contract.
- Large artifacts belong under `/mnt/data`; account for its currently constrained free space and its root-owned permission policy. Do not delete unrelated data without authorization.
- Keep the NVIDIA driver/module state unchanged during this goal; do not reload modules or reboot merely to enable ordinary-user counters. `RmProfilingAdminOnly` is module-wide and the desktop uses the RTX 2060.
- Ordinary-user serving, benchmarks, development and Nsight Systems remain the default. Use authorized sudo/root only for targeted Nsight Compute on selected V100 kernels/real-tensor microbenchmarks, preserving the exact CUDA/Python/SGLang environment and explicitly targeting the V100 (never RTX 2060).
- Do not guess metric names. Enumerate supported Nsight Compute metrics and prove with a minimal V100 kernel that root NCU reads SM throughput, tensor, DRAM/L2, occupancy and warp-stall metrics. If that passes, counter access is READY without reboot.
- Existing `NVreg_RestrictProfilingToAdminUsers=0` modprobe/initramfs change may remain for a future convenient reboot, but is not an optimization blocker and does not authorize changing the running module state now.
- Nsight Systems: separate prefill/decode; capture a steady-state CUDA Graph window; record kernel timeline, launch gaps, graph coverage, sync and memcpy.
- Nsight Compute: profile only top real kernels/geometries from Systems; collect supported SM throughput, Tensor Core activity, DRAM and L2 throughput/hit, occupancy, registers/shared memory, warp stalls, and roofline/arithmetic-intensity evidence.
- DCGM/CUPTI may be used for low-overhead continuous SM/tensor/DRAM activity, clocks, power, and memory records where supported. Do not install/configure host-wide services without assessing impact and permission.
- Every optimization follows: before hardware profile → root cause → change → operator A/B → service A/B → after profile.
- Primary optimization object: c1/c2/c4 GPTQ small-M. Do not reopen rejected low-ROI Mamba SSU or Attention donor work absent materially new evidence.
- Do not claim measured performance before model/runtime recovery and correctness qualification. Keep physical measurements distinct from estimates.

## Completion condition

A repeatable local S1 runtime passes serving, interleaved state isolation, graph 1/2/4 replay, bounded c1/c2/c4/context/headroom and short stability checks; the existing targeted root NCU report is verified against the local V100 and production GPTQ kernel with usable supported hardware metrics; and the final source/model/runtime/dependency identity is frozen. Nsight Systems Falcon CUDA trace remains explicitly `DEFERRED / NOT QUALIFIED` and is not represented as PASS. No driver reload/reboot or RTX 2060 interference.

## Result

`G016 LOCAL FALCON + NCU AUTHORITY READY` under revised acceptance. Serving/state-isolation and graph replay checks, bounded local capacity/headroom and short stability evidence pass. Targeted root NCU production GPTQ kernel and hardware counters validated. `NSYS FALCON SERVER CUDA TRACE — DEFERRED / NOT QUALIFIED`; it is not a pass or gate. No performance conclusions are made under G016.
