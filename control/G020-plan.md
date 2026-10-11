# G020 plan

Status: **PARTIAL, complete for authorized offline evidence.** No further collection or attribution route in scope.

## Integrated conclusion

Nine archived DECODE traces yield three GPU `step[DECODE bs=N]` annotation spans each. Using steps 1–2 after excluding the transitional first step, G020 reproducibly reports owner calls, raw duration sums, interval unions, exclusive time, pairwise overlap, UNKNOWN duration and kernel-annotation coverage. GPTQ gate_up/down, Attention, explicit Mamba scan/conv, elementwise/norm, copy/set, and conservative Other GEMM are distinguished; unidentified generic GEMM/GEMV remains UNKNOWN. The 4K/c4 traces actually annotate bs3. See `experiments/G020/report.md` and `artifact-manifest.json`.

These are profiled GPU activity categories bounded by annotation timestamps, not graph-node ownership, complete model-forward owner shares, or uninstrumented timings. Existing FPM and G017 service baselines are reported separately; only directly comparable descriptive contrasts are made. Main unresolved item is UNKNOWN per-callsite mapping, plus Graph node IDs and exact whole-forward reconciliation. Scope ends here; no new GPU test or profiler is authorized.

## Completion

- Analyzer: `experiments/G020/analyze_traces.py` (CPU-only), output `analysis.json`; rerun command appears in the report.
- Nine archived trace paths and hashes, analyzer/analysis/report hashes in `artifact-manifest.json`.
- Deterministic rerun compared byte-for-byte; Python syntax checked.
- Final report status: PARTIAL. Commit/push and remote SHA verification remain the final repository action.
