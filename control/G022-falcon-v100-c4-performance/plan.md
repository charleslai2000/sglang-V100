# G022 plan

Status: **ACTIVE**. G022 is the implementation/optimization goal; G021 remains the frozen metrics-contract authority. Decode Step latency minimization at c1/c2/c4 has highest optimization priority, with c1 latency, c2 scaling efficiency, and c4 GPU/service throughput as distinct objectives. Prefill and Cold/Warm Cache TTFT remain R17-level co-equal service metrics. Do not treat G021's unresolved diagnostics as a reason to recreate profiling infrastructure.

## Established inputs

- G017 S1 and Candidate remain frozen comparison paths; Candidate identity must be exact-path/SHA.
- G021 service matrix, bounded S1 CUDA Event/FPM capture, root NCU hot-kernel measurements, and request lifecycle capture are at `experiments/G021/`; report `experiments/G021/report.md`.
- G020 conservative trace attribution is at `experiments/G020/`; actual 4K/c4 trace is batch 3, not batch 4. It reports 7.3 ms Other GEMM at batch4 in the separate 1K/batch4 window, and c1 unknown GEMV/GEMM near 7.7 ms. These trace windows are not whole-forward exclusive budgets.
- G021's single 4K/c4 request lifecycle wave joined 10/10 RIDs and observed queue residence 0.20–25.62 s tracking TTFT; the sample demonstrates a bounded queue-tail mechanism, not a service quantile.
- Prior G017 c4 Candidate showed service improvement and a 1K/c4 FPM GPU-forward change, but G021 records it as opt-in exact SHA. Revalidate final comparison identity/conditions; don't treat NCU kernel delta as whole-forward savings.
- This checkout has existing unrelated dirty source changes. Preserve them. Only change implementation files necessary for G022 after recording baseline source state; commit scoped files only.

## Decisive frontier

1. Establish actual admitted c1/c2/c4 baselines for 1K/4K × 256/512 outputs on S1 and exact Candidate, distinguishing actual GPU batch from submitted concurrency. Record GPU Decode Step latency/TPS for all three points; c1 optimizes single-token latency, c2 scaling efficiency, and c4 GPU/service throughput. Capture only the smallest missing bounded evidence for stable GPU-step + service comparison, owner/batch mapping, cache/lifecycle and memory headroom.
2. Build matched c1/c2/c4 GPU-forward budgets by reconciling G020/G021 traces and supported NCU data against the same actual batch/workload. Resolve top unknown GEMV/Other GEMM or bound the residual; map launch/graph gaps without calling request wall span GPU time. Keep Prefill and Cold/Warm Cache TTFT in the co-equal R17 metric ledger.
3. Rank the strongest evidenced compute owner; inspect mature donor/source alternatives, emphasizing GPTQ, Other GEMM/GEMV, memory dependency, SM/Tensor utilization and CUDA Graph efficiency only where evidence applies. Consider order-of-magnitude opportunity as a hypothesis, not a promise. Implement one evidence-selected change at a time; retain semantics, rollback and exact route identity.
4. For each change report c1/c2/c4 GPU Decode Step Time/TPS, service Output TPS, TTFT, ITL/TPOT, SM/Tensor/HBM, VRAM/KV/Mamba capacity, correctness and fallback. Run targeted correctness and graph/state checks, then repeated service and GPU-forward A/B; require improvements in both actual GPU forward and real service, with c1/c2 controls.
5. Report measured results and limitations, commit/push scoped deliverables and verify remote SHA. c8/c16 remain optional safe references only, never optimization gates.

## Resource and execution gate

The only inference/profiling GPU target is V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; RTX 2060 is unrelated and off limits. Before each V100 run, check MemAvailable >=8 GiB, target idle/ownership, existing process tree, and memory headroom. Stop safely on floor breach or unexpected device/workload. Do not raise max concurrency to claim compute performance.

## Result

Goal opened from user authorization on 2026-10-12. No G022 experiment or source optimization has run yet. Current repository HEAD at start: `c3f998fa1c172e1a73b28e5c9bd72ddd93e0b30a`; preserve unrelated worktree state. V100 last checked at 7 MiB / 0% utilization; RTX2060 was visible in the `nvidia-smi` inventory only and was not used.
