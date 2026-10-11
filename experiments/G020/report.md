# G020 — Existing CUDA trace offline attribution

Status: **PARTIAL (offline analysis completed; terminate as requested)**. No GPU jobs, profiling, instrumentation, production source edits, or kernel changes were made in G020. The original G019 trace claims that per-forward boundaries could not be joined are partially superseded: PyTorch exported three `gpu_user_annotation` spans `step[DECODE bs=N]` per bounded profile capture, with GPU timestamps in the same time coordinate as GPU activities. Those annotations allow a conservative per-step kernel-activity table by temporal containment. Graph node IDs still are not present; this report does **not** claim a per-node ledger or fully reconciled whole-forward owner attribution.

## Inputs and identity

- Trace files (nine DECODE traces, including c57 4K/c4 repeat) are listed with SHA-256 in `artifact-manifest.json`. Raw traces remain outside the git worktree at `/mnt/data/pc01/sglang-v100-migration/experiments/G019/`.
- Every trace reports `deviceProperties[0]`: `Tesla V100-SXM2-16GB`, CC 7.0, 80 SMs, 16,928,342,016 total bytes. The capture environment was G019's UUID-pinned V100; no RTX trace is present.
- Chrome trace event timestamps are microseconds, with a common `baseTimeNanoseconds` in metadata. GPU kernel events are on stream 13 almost exclusively, with a small set on stream 45. Each file contains exactly 3 `cudaGraphLaunch` API events; actual graph replay is further confirmed by `step[DECODE ...]` annotations, not inferred from launch count alone.
- G019's profiler asked for decode-only `num_steps=3`; each trace has 3 GPU `step[DECODE bs=N]` annotations. The condition requested c4, but archived 4K/c4 steps are annotated `bs=3` for both modes—tables therefore report **actual scheduler graph batch 3**, not c4. 1K/c4 is actual bs4. Per SGLang `SchedulerProfilerManager._profile_batch_predicate`, the profiler is started on the first decode scheduler iteration and stopped after the target count is exceeded; in continuous batching a concurrently scheduled request wave can thus yield actual batch 3. Do not label 4K/c4 values as batch4.
- S1 and c57 module SHA pins remain those in G019 manifest. The accepted c57 profiler server startup log verifies exact path/hash and candidate dispatch.

## Analyzer and accounting definitions

`experiments/G020/analyze_traces.py` reads only archived `.trace.json.gz` files. It does not import torch, initialize CUDA, or access a GPU. Run:

```bash
python3 experiments/G020/analyze_traces.py \
  --root /mnt/data/pc01/sglang-v100-migration/experiments/G019 \
  --out experiments/G020/analysis.json
```

For each GPU annotation, it selects GPU kernel activities wholly contained in the span for the kernel ledger. It additionally intersects GPU copy/set activities with the annotation. The analyzer classifies by exact kernel symbols and, for GPTQ gate_up/down, combines symbol + launch grid + pinned checkpoint shape/source. Rules and UNKNOWN defaults are in the script. It reports:

- calls and raw duration sum per owner; raw duration sum counts concurrent intervals more than once;
- per-owner interval union (removes intra-owner overlap);
- exclusive duration (time covered by exactly one owner category);
- pairwise owner overlap;
- full kernel interval union, kernel-only duration sum, all GPU-activity union, annotation span and annotation coverage;
- unknown call count and duration.

The raw kernel event names plus source mapping are inspectable; all kernel events use the profiler's GPU event timeline, but no graphNodeId field exists. Same-stream temporal steps are useful for offline categorization, not physical CUDA Graph node attribution.

### Conservative owner rules

- **S1 GPTQ gate_up/down:** exact `gemm_half_q_half_alt_4bit_kernel`; Falcon MLP source declares merged gate_up `[intermediate, intermediate]` then down projection. Checkpoint: hidden=3072, intermediate=12288, 44 layers, TP1. Captured grid `[192,1,24]` = N 24576 gate_up; `[24,1,96]` = N 3072 down. Each stable step has 44 events of each geometry.
- **Candidate GPTQ gate_up/down:** exact `packed_fp32_partial` + `packed_fp32_reduce<__half>` from accepted `gptq_candidate_v8.cu`; grids `[192,1,24]` partial and `[96,1,1]` reduce map to N=24576 gate_up, while `[24,1,96]` partial and `[12,1,1]` reduce map to N=3072 down. This mapping is shape/source-backed; per each operation both partial and reduction kernels are counted into that projection's owner.
- **Attention:** `_fwd_grouped_kernel_stage1`, `_fwd_kernel_stage2`, `fused_rope_kernel`, `store_kvcache`; traced kernels map to pinned `decode_attention.py` functions / Falcon attention call path and source names.
- **Mamba:** `_selective_scan_update_kernel`, `causal_conv1d_update_kernel`; Mamba uses its own causal convolution and selective scan source path. Generic `_layer_norm_fwd_1pass_kernel` is left UNKNOWN rather than presumed Mamba because the generic symbol alone does not prove its owning call site. Fused attention/Mamba communicator norms are attributed only where exact FlashInfer `FusedAddRMSNormKernel` symbol and source call path map; the broad elementwise category remains separate.
- **Other GEMM:** explicitly identified CUTLASS GEMM symbols are assigned only to the broad Other GEMM kernel category, not a Falcon projection. `gemmSN_TN_kernel` is not source-identified and stays UNKNOWN, as does CUBLAS GEMV. UNKNOWN is ~7.7–7.8 ms in c1 and ~2.19–2.20 ms in bs3/4 (51 calls); additional unprofiled work remains possible.
- **Elementwise / normalization:** generic PyTorch elementwise/activation symbols and exact FlashInfer norm symbols. The category denotes kernel kind; generic elementwise is not mapped to a unique model layer.
- **Memcpy/memset:** recognized symbols/activity categories. Some `memcpy32_post` is a kernel symbol; categorized as copy-like kernel work, not GPU memcpy activity.

## Stable per-step activity results

The first c1 step is excluded from the summary below as a startup/transitional observation: it can have anomalous duration/activity (S1 1K/c1 annotation 36.53 ms, versus stable subsequent 30.55 ms; its kernel set has much lower coverage). Use step indices 1 and 2 (0-based, annotations sorted by GPU timestamp) as the two stable observations per trace. These are profiles with `record_shapes=true`; their category sum is instrumentation-window kernel execution cost, not uninstrumented service cost.

Per row: average of the two stable steps; milliseconds. `calls` are events per step. `duration sum` can overlap; `exclusive` uses interval sweep; `kernel union` de-duplicates overlap across all kernel records. Percent coverage is kernel union / GPU step annotation span. The API span is not the whole-model forward CUDA event: it is a GPU annotation around the recorded step and can include asynchronous stream tail; do not force residual into owners.

| Mode / context / requested c | Actual annotated batch | GPTQ gate_up calls / ms | GPTQ down calls / ms | Attention calls / ms | Mamba calls / ms | Other GEMM calls / ms | Elementwise/norm calls / ms | memcpy/memset calls / ms | UNKNOWN calls / ms | annotation ms | kernel union ms | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 / 1K / c1 | 1 | 44 / 9.91 | 44 / 6.40 | 176 / 2.46 | 132 / 0.69 | 177 / 0.00 | 755 / 2.19 | 135 / 0.26 | 51 / 7.69 | 30.55 | 29.58 | 96.8% |
| S1 / 1K / c4 | 4 | 44 / 11.41 | 44 / 7.30 | 176 / 2.91 | 132 / 1.79 | 353 / 7.20 | 756 / 2.82 | 134 / 0.27 | 51 / 2.19 | 36.98 | 35.90 | 97.1% |
| S1 / 4K / c1 | 1 | 44 / 9.86 | 44 / 6.36 | 176 / 3.00 | 132 / 0.69 | 177 / 0.00 | 755 / 2.17 | 135 / 0.26 | 51 / 7.76 | 31.04 | 30.08 | 96.9% |
| S1 / 4K / c4 | 3 | 44 / 11.49 | 44 / 7.33 | 176 / 4.94 | 132 / 1.43 | 353 / 7.21 | 762 / 2.87 | 134 / 0.28 | 51 / 2.20 | 39.59 | 37.72 | 95.3% |
| c57 / 1K / c1 | 1 | 44 / 8.65 | 44 / 5.23 | 176 / 2.47 | 132 / 0.70 | 177 / 0.00 | 755 / 2.19 | 47 / 0.10 | 51 / 7.76 | 28.02 | 27.08 | 96.6% |
| c57 / 1K / c4 | 4 | 44 / 9.80 | 44 / 6.00 | 176 / 2.92 | 132 / 1.79 | 353 / 7.33 | 756 / 2.82 | 46 / 0.09 | 51 / 2.19 | 34.02 | 32.94 | 96.8% |
| c57 / 4K / c1 | 1 | 44 / 8.56 | 44 / 5.16 | 176 / 3.02 | 132 / 0.69 | 177 / 0.00 | 755 / 2.16 | 47 / 0.09 | 51 / 7.82 | 28.45 | 27.50 | 96.6% |
| c57 / 4K / c4 | 3 | 44 / 9.73 | 44 / 5.94 | 176 / 4.88 | 132 / 1.43 | 353 / 7.31 | 762 / 2.83 | 46 / 0.09 | 51 / 2.19 | 36.00 | 34.40 | 95.6% |

Candidate 4K/c4 repeat (actual bs3): stable-step category means are GPTQ gate_up 44 / 9.63 ms, down 44 / 5.81 ms, Attention 176 / 4.86 ms, Mamba 132 / 1.43 ms, Other GEMM 353 / 7.28 ms, elementwise/norm 762 / 2.81 ms, copy/set 44 / 0.09 ms, UNKNOWN 51 / 2.19 ms; kernel union 34.26 ms, annotation 35.34 ms, coverage 97.0%. This supports repeatability of the filtered trace-window category distribution, not performance with profiler off.

### Overlap and unknown bounds

- Kernel execution interval sum intentionally counts GPU activity durations; the steady-step category totals in the table are near but not equal to annotation span because kernels overlap and there are gaps/tails. For most stable windows category interval unions overlap only modestly (script emits exact pairwise overlap). Examples: c57 4K/c4 stable step has 34.28 ms all-kernel union in 35.38 ms annotation, categories' pairwise union excess only a few microseconds; S1 4K/c4 is 37.66 ms union in 38.82 ms, with larger category overlap due elementwise, other GEMM and GPTQ/copy combinations.
- Owner-exclusive time does not double-count owner overlaps. Cross-category overlap is explicitly emitted in `analysis.json`; category duration sums must not be added to infer wall time. UNKNOWN kernel symbols are 51 calls and ~7.7–7.8 ms in stable c1 steps versus ~2.19–2.20 ms at bs3/4. The unidentified GEMM/GEMV names lack a source-backed callsite/owner mapping. There may also be unobserved/annotation-tail time: kernel coverage is typically 95–97%, not 100%.
- Categories are execution-kind attribution. “Other GEMM” means identified CUTLASS/CUBLAS matrix-multiply kernel activities, not a specific Falcon module or layer. “Elementwise / normalization” similarly aggregates identifiable kernel family activities, not a unique layer owner.

## Comparison and boundaries

Observed per-step category timings depend on actual batch and backend geometry. S1 vs c57 comparisons at the same context and actual batch are reasonable for descriptive instrumented execution-category contrast. Candidate accepted path's GPTQ owner comprises partial + reduce kernels; S1 uses its one GPTQ kernel per projection, so a direct operation with two kernels vs one still compares categorized GPU execution but not one-launch kernel latency.

The 4K/c4 client workload reached only graph batch 3 in traces. Do not compare that row as graph batch4. 1K/c4 rows did reach bs4. FPM in G018 is a separate capture with per-forward CUDA event category but different capture windows/batch streams; no exact annotation-to-FPM step ID exists. It supports order-of-magnitude validation only by matching active batch and context range, not exact subtraction: G018 S1 FPM c1/c4 (around ~1K/request) means were 28.38/34.18 ms and candidate 26.10/31.67 ms; G020 profiled annotation spans in stable 1K steps are S1 30.55/36.98 ms and c57 28.02/34.02 ms. Differences are not a clean profiler overhead estimate because windows and event boundaries differ. Do not subtract to claim overhead or owner residual.

The accepted G017 uninstrumented service authority is separate. Per-repeats throughput and client ITL remain in `g017-close-clean-s1-service.json` and `g017-close-clean-candidate-service.json`; mean aggregate TPS (S1/c57): 1K c1 33.72/36.54; 1K c4 103.98/111.25; 4K c1 29.72/31.91; 4K c4 47.78/50.91. These are the serving throughput authority, not profiler-trace rate. TTFT is prefill/client timing, not per-step GPU event.

## Results and strongest bounded conclusions

1. Kernel-category execution cost is directly supported by archived GPU activity events for the two stable annotated decode steps per condition. For S1 in these captures GPTQ gate_up+down sum is about 16.3–18.8 ms per annotated step; candidate partial+reduce gate_up+down about 13.7–15.7 ms. This is trace-category execution sum under `torch.profiler` and recorded graph execution; intra-category overlap is accounted separately in the analyzer. It is not a whole-forward exclusive share and not an unprofiled kernel timing.
2. Attention activity is about 2.46–4.95 ms and Mamba's explicitly identified selective-scan + causal-conv about 0.69–1.79 ms, while identified Other GEMM kernels take 7.2–7.33 ms at bs3/4 and are absent at c1. Additional unidentified GEMM/GEMV events remain UNKNOWN (~7.7–7.8 ms at c1). Neither category is a unique model-layer owner; do not casually assign these events to attention/MLP.
3. A simple performance owner ranking is not fully defensible because unclassified GEMM/GEMV call sites lack unique callsite/Graph-node mapping. Candidate 4K/c4 actual batch3 has greater attention activity than 1K/bs4, but input context and batch both differ; compare S1/c57 within same actual-batch condition rather than infer a pure context effect. The evidence shows identified GPTQ activity and unidentified GEMM/GEMV events account for substantial execution time, but cannot establish which is the recoverable bottleneck. Recommended next owner: **UNKNOWN pending a callsite-aware offline mapping of generic CUBLAS GEMV**, not a kernel optimization recommendation. No new collection is authorized under G020.

## Whole-forward owner attribution (explicit boundary)

For each annotation step, kernel intervals explain ~95–97% of its GPU user annotation by interval union. UNKNOWN events are substantial at c1 (~7.7–7.8 ms, 51 calls), and remain ~2.19–2.20 ms at bs3/4. This is **GPU activity coverage of the profiler's `step[DECODE]` annotation**, not complete Falcon forward owner accounting. The annotation is a PyTorch profiler `record_function` span around model forward executed during graph replay. It can contain async stream tail; individual kernel windows have CUDA activity timing and no graph node ID, and model worker may use auxiliary streams. The residual annotation-span minus kernel interval union (~0.95–2.1 ms in steady rows) is unassigned GPU-span remainder, not exactly host time. Do not reconcile to G018 FPM by subtraction; FPM is separate capture, separate boundaries, and no step correlation ID. Graph IDs/nodes remain UNKNOWN.

No entire GPU interval is inferred from one sum of concurrent kernels. Owner-exclusivity table uses union-of-intervals and sweep, with overlapping GPU activities assigned no exclusive category time when two classified owners overlap. Whole-device idle gaps, memory traffic, SM/Tensor/HBM utilization remain UNKNOWN from these traces. This deliverable ends after offline trace analysis.
