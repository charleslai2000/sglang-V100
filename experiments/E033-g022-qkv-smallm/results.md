# E033 Results — Falcon QKV small-M candidate

## OBSERVED
- A temporary uncommitted opt-in route was added to `UnquantizedLinearMethod.apply`, exact class name `QKVParallelLinear`, FP16 contiguous M2/M4 `[2048,3072]` only, 16-byte aligned buffers, SM70 only; default flag was false. New M2/M4 launch configs used existing `sm70_small_gemm.linear`. After correctness checks and before NCU, all temporary production-source/test/flag/config changes were removed; no E033 implementation remains in the worktree.
- Attempted repo pytest gate could not start because `pytest` is not installed in the active `sglang-v100` environment (`No module named pytest`). This is not a test pass.
- Direct invocation of the actual opt-in production linear dispatch on concatenated exact Falcon layer-0 QKV checkpoint weights passed unchanged elementwise tolerance `rtol=0.002, atol=0.001` vs `F.linear`: M2 0/4096 outside tolerance, max abs `0.00390625`, relative L2 `1.23e-5`; M4 0/8192, max abs `0.00390625`, relative L2 `1.94e-5`. CUDA Graph capture/replay of that same module dispatch passed at both shapes.
- Earlier E032 direct-kernel timing screen on these exact weights measured small-M median vs F.linear: M2 28.672 vs 45.056 us; M4 29.696 vs 50.176 us (300 event-paired iterations, median; isolated math, not integrated model latency).
- E031 source spans prove one QKV call per each of 44 layers per real Forward; at c1 it contains one FP16 `gemv2T_kernel_val` child per invocation, at c2/c4 44 CUTLASS `[8,8,4]` source-correlated calls per Forward. E031 measured correlation sums ~0.87/1.06/1.17ms eager across c1/c2/c4; these are attribution only.

## INFERRED
- QKV is a plausible reversible low-batch target because the source module, exact checkpoint tensors and M1/M2/M4 launch geometry are now known; the candidate's exact weight screen passed but not the whole PyTorch QKV execution path.
- This does not establish achievable production GPU step recovery. No NCU counter for QKV and no model/service/Graph effect is measured.

## UNRESOLVED
- The temporary direct module-dispatch check used a constructed `QKVParallelLinear` instance with concatenated checkpoint weights; it is not a full-model module load, distributed slicing/layout/state audit.
- Full repository pytest could not start (`pytest` absent). No matched NCU, c1/c2/c4 full production Graph Forward, service TPS/ITL/TTFT, Prefill/Cold-Warm TTFT, memory/capacity result. Historical real NCU replay on this shared host drove MemAvailable below the 8-GiB hard floor, so no new replay was attempted. The task's required correctness → matched NCU → Forward → service sequence therefore stops before NCU; the candidate implementation was removed rather than leaving an unqualified route.

## Bounded answer
QKV small-M is a correctness-screened hypothesis with a temporary exact checkpoint dispatch + direct Graph pass, not full-model correctness or performance evidence. The route was removed after this gate because matched NCU cannot safely be attempted under the 8-GiB host floor; no implementation remains. Resume only when a demonstrably safe matched-NCU route exists, then complete full model/service A/B.
