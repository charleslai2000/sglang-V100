# E035-R2 — Identity-safe complete-Linear comparison

## Ruling
**The prior old/R1 A/B was contaminated by Python native-module caching.** Old and R1 shared the same pybind11 extension name. In one interpreter, the second `importlib` load returned the first module/function and first `__file__`; swapping load order swapped which binary ran. The prior `r1-all-fixtures-3pairs.jsonl` is therefore **INVALID/SUPERSEDED for old-vs-R1 timing**. The exact output file is preserved unchanged and has an adjacent `r1-all-fixtures-3pairs.status.txt` marker. Its distinct recorded file SHAs do not establish that both functions executed.

The repaired experiment ran one arm per new Python process, serially and in rotated order. Actual path and SHA were recorded inside each process; old, R1 and c57 ran under distinct PIDs. The original binaries were reused, not rebuilt or renamed. Correctness remained bound to each executed binary/layout/config. Full Eager Linear results show R1 is substantially faster than old in both projections, but still slower than exact c57 in every projection/shape. **R1 remains NO-GO for production; no Forward integration.** Independent NCU/SASS and R1/original correctness evidence remain valid.

## R2 graph-field correction
The old R2 artifact was made before `paired_isolated.py` gained the final Graph audit. In that old script, `graph_oracle_bad` was set to zero for E035 after comparing `gy` (the Graph-returned FP16 value) against `ref`, which was cast-before FP32 oracle but explicitly rounded to FP16 for the code path; this did **not** test finite FP32 Graph output scratch, FP32 `go` against frozen strict tolerance, or `gy == go.half()`. It did test eager-vs-Graph output equality (`gy` against eager `y`) and an incorrectly named/contracted `graph_oracle_bad` comparison. Therefore the old field cannot independently be represented as Graph strict PASS. It was not a graph output failure; it was an insufficient/ambiguous assertion set. Targeted repair, using unchanged R1 binary and new parallel binary at the same exact L43/down M4 cell, was executed through the corrected script SHA `6e1f39b634b6dd49bf049ea7dc90eff35952ff9b77eadfa2d2007086de229dca`. For each exact binary SHA, child result records `graph_eager_mismatch=0`, `graph_oracle_bad=0` on the actual FP32 `go`, `graph_go_finite=true`, `graph_cast_bad=0` for `gy==go.half()`, output dtype `torch.float16`. Artifacts: `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-graph-correction-r1-l43downm4.json`, `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-graph-correction-parallel-l43downm4.json`. Reuse existing R2 full Eager/Graph timings; no old binary or full matrix remeasured.

## Cache-root proof
Environment: Python 3.12, PyTorch 2.9.1+cu128, CUDA 12.8. Direct import of `pybind11` package is unavailable, but the actual compiler headers are PyTorch's vendored `torch/include/pybind11`; `detail/common.h` reports pybind11 3.0.1. Both exact E035 binaries export `PyInit_e035_grouped_q4_sm70` (same `PYBIND11_MODULE` name). `paired_bench.py` used `spec_from_file_location('e035_grouped_q4_sm70', ...)` for each file. Direct same-process reproduction:

- old (`build4`) then R1 (`build-r1`): both returned the same module object, `__file__` stayed old path, and `grouped_q4_sm70` was the identical function object.
- R1 then old: both returned the same module/function object and `__file__` stayed the R1 path.
- The imported module is not inserted in `sys.modules` by that code path; the reuse is at the CPython extension-module initialization/pybind11 internals keyed by the extension init name, not normal `sys.modules` lookup. Both functions report native owner `e035_grouped_q4_sm70`.

Thus the three-route result mislabeled the second requested E035 arm as independent. Under the old-first order old ran in both old and R1-labelled slots; the other rotations likewise alias the first-loaded E035 extension. The historical c57 result is from its distinct `common_ops` module and does not repair old/R1 symmetry. Exact c57 timing comparisons are independently supported by corrected R2.

## Minimal repair and protocol
New reusable script: `paired_isolated.py`. Each arm/cell gets a fresh child interpreter; process start, imports, CUDA setup, fixture input loads, deterministic pack/gather prep and warmup are outside event timing. Children run serially, with three rotated arm orders per fixture×M cell. No two E035 extensions coexist in a process. JSONL records PID, actual module file, binary SHA, native function identity/owner, manifest SHA, correctness flags and separate timings. The same exact original binaries are used:

- old `51cb9b32a77ee6af5c7e940acb46b812c7950fe598cc6a764a07ee49d50736b8`
- R1 `f72b764654de643384166f44acf7924fd8199a6512f9573bb921224394b2392d`
- accepted c57 `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`

Input tensors, actual M, output dtype and cache-warmed boundary are matched. Eager event timing includes full operator invocation: E035 gather + grouped computation + final FP16 cast, versus c57 native complete call. CUDA Graph is captured/replayed separately and reported separately (same complete call; graph capture excluded from timing). No NCU instrumentation in these wall-time measurements. Each E035 child validates the exact loaded eager output against the frozen cast-before FP32 oracle `abs(error) <= 1e-4 + 5e-5*abs(reference)`, strict-bad 0, and finite final FP16. The original R2 Graph check required eager/Graph equality, but its `graph_oracle_bad` field was insufficiently defined and did not validate the actual FP32 Graph scratch/cast chain; see **R2 graph-field correction** below. Corrected targeted R1 and parallel L43/down M4 runs now validate FP32 graph `go`, finite, strict oracle, `gy == go.half()`, and Eager/Graph equality. c57 emits FP16 by contract; its correctness is checked for finite output and Graph/Eager equality. R2 does not claim c57 exposes a pre-cast FP32 result.

## Background completions and clean down repeats
The reported five background tasks have now terminated; foreground audit found no remaining Python benchmark process, and target V100 is back at 7 MiB/0% with host MemAvailable above 8 GiB. Three of those tasks were deliberately terminated after duplicate starts were detected; their partial outputs were retained and are not folded into the complete result. One completed 0/down batch was legitimately used once in the raw full file; an accidental repeated output generation caused duplicate keys there. Original raw file remains untouched, adjacent audit documents duplicate selections, and reconciled primary removes them. To eliminate any concern about the previously overlapped cells, an additional clean single-driver serial matrix reran L0/down and L22/down × M1/M2/M4 × 3 rotated pairs. Raw `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-paired-down-clean-repeats.jsonl`, SHA `b19558730cc89b4ab645e2e2c54e1819c428cdda8edc2c7864c767c689f86ea7`. Corrected full strict oracle/Graph fields and exact per-arm identities are present. Clean medians support the same NO-GO: parallel remains slower than c57 in every cell. R1 timed output occasionally has one first-pair eager outlier, but pair medians and Graph values preserve a wide R1/parallel separation; no clock explanation is asserted.

## E035-R2 down group-parallel candidate stage (2026-10-16)
The full experiment design, hypothesis, precise arithmetic contract, risk, intended CTA grid, and partial traffic estimate are in `experiment.md`. Candidate source only: `grouped_q4_sm70_parallel.cu` SHA `75cd105c7ed82de68d5810417e99f2237bf4cda8edfe58b772c3c224886ce521`; harnesses `test_parallel.py` SHA `227d72548a4d26e6fafbc3bd9c88bbbb95033372e288cb2568a315f656d89937`, `ncu_parallel_launch.py` SHA `eb61410991574b20a9c9d5a1799d83d52d2a16282781037f7494394347f844b7`, `run_down_pairs.py` SHA `ef627363ed7b41293ac75d8c2b9841c1c63a4eab11d151ea9b89bc24c4f716a1`, and corrected `paired_isolated.py` SHA `6e1f39b634b6dd49bf049ea7dc90eff35952ff9b77eadfa2d2007086de229dca`. Parallel `.so` SHA `23ea9c87cbea6d6f6d491a94a5ef2446c363c7367d573d0807ee2358aa26bbcd`. Exact R1 and c57 binaries are the R2 SHA identities above.

**Resource SASS and NCU findings:** `cuobjdump` on actual candidate binary shows `group_wmma_partial REG48 STACK0 SHARED2048 LOCAL0`, reducer REG31 STACK0 LOCAL0, gather REG12 STACK0; no dynamic accumulator local-memory array. Exact R1 M4/down report confirms R1 serial group grid `[192,1,1]` block 32, 47.34% long scoreboard; candidate group grid `[192,1,96]` = 18,432 CTAs, same block/WMMA count, 26.61% long scoreboard, 183.68us vs 1.879ms NCU kernel. NCU is single kernel. Actual partial is 4,718,592 logical bytes (FP32, G×M×N), while measured partial-kernel DRAM writes are 3.718MB, not equal (transaction granularity/other stores included); read 18.982MB. R1 group kernel DRAM read/write 19.590/0.363MB. Reducer and extra kernel launch are not in this NCU target and are included in complete Linear timings. R1 report `r2-r1-down-m4.ncu-rep` SHA `4fab9fa2...`; candidate `r2-parallel-down-m4.ncu-rep` SHA `a85afd9f...`. Actual binary cuobjdump STACK0 is stronger stack claim; NCU `launch__stack_size=1024` is launch stack-size metadata and must not be read as per-thread allocated local array.

Strict correctness: actual candidate SHA passes 9/9 down fixture×M cells across L0/L22/L43; FP64 GPTQ→FP32 strict-bad=0, finite FP32/FP16, repeat and independent scratch bitwise identical; Graph1/2/4 checks use actual `go`: strict-bad=0, finite, `gy == go.half()`, and Graph eager equality. L43/down M4 ×1 and ×4 high amplitude (input max 2420/9680) strict-bad=0 and finite. R1 repeat/independent scratch checks stay established by previous correctness run; targeted R1 M4 graph correction is separately documented above.

Complete Eager/Graph median-of-three-pair-median µs, down by fixture and M (old omitted; do not repeat invalid/superseded old arm):

| fixture | M | R1 / parallel / c57 Eager | R1 / parallel / c57 Graph |
|---|---:|---:|---:|
| L0 | 1 | 1541.6 / 181.2 / 130.6 | 1542.7 / 183.3 / 124.4 |
| L0 | 2 | 1506.8 / 184.3 / 133.1 | 1503.7 / 185.3 / 126.0 |
| L0 | 4 | 1675.3 / 188.9 / 147.5 | 1677.3 / 193.0 / 144.4 |
| L22 | 1 | 1505.3 / 181.2 / 129.0 | 1501.2 / 183.3 / 126.0 |
| L22 | 2 | 1511.4 / 181.2 / 132.1 | 1506.3 / 184.3 / 124.9 |
| L22 | 4 | 1715.2 / 183.3 / 148.5 | 1716.2 / 188.4 / 141.3 |
| L43 | 1 | 1590.3 / 181.2 / 128.0 | 1582.1 / 182.3 / 124.9 |
| L43 | 2 | 1509.4 / 181.2 / 135.2 | 1508.4 / 183.3 / 129.0 |
| L43 | 4 | 1632.3 / 184.3 / 144.4 | 1628.2 / 187.4 / 148.5 |

All nine geometry cells are 3 rotated pairs; L43/M4 causal initial result also independently stored in `r2-paired-down-m4.jsonl` and full matrix in `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-paired-down-full.jsonl`. Candidate improves R1 greatly, but is slower than exact c57 on all 9 cells (parallel/c57 Eager roughly 1.28–1.41×; no M-only exception). This is a **NO-GO for proceeding to Forward integration** under the acceptance criterion. This stage does not reject arithmetic correctness or the bounded performance result. It does not establish service/Forward effect. Keep production dispatch/defaults unchanged.

## Causal M4, layer 43
Three rotated pairs, 20 event iterations/arm/pair; all recorded PIDs distinct. Values are median over three pair medians, microseconds. Eager / Graph replay shown separately.

| projection | old eager / graph | R1 eager / graph | c57 eager / graph | old→R1 eager delta | R1/c57 eager |
|---|---:|---:|---:|---:|---:|
| gate_up M4 | 2697.2 / 2725.9 | 465.9 / 479.2 | 230.4 / 229.4 | -82.7% | 2.02× |
| down M4 | 2180.1 / 2176.0 | 1634.3 / 1628.2 | 144.4 / 148.5 | -25.0% | 11.32× |

Per-run evidence: `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-causal-m4.jsonl` (gate-up) and `r2-causal-down-m4.jsonl` (down). The independently isolated chain yields a very large old→R1 gate-up reduction as expected from removing local-array SASS, a smaller down reduction, and still clear R1-vs-c57 regressions. Full matrix below is the primary table.

## Full 6 fixture × M1/M2/M4, three rotated pairs
Raw JSONL: `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-all-fixtures-3pairs.jsonl`, SHA256 `747571cf2b9b99015f8ffd5a80aa8cb0aa7fde3f6846e3911c9a149b82abfa3c`. 54 rows, 18 geometry cells × 3 pairs; 20 timing iterations per arm/pair. Table shows median of the three pair medians for each cell; format `old / R1 / c57` in µs. Graph is replay-only and intentionally separate from Eager.

| fixture/projection | M | Eager old / R1 / c57 | Graph replay old / R1 / c57 |
|---|---:|---:|---:|
| L0 down | 1 | 2086.9 / 1499.1 / 128.0 | 2095.1 / 1497.1 / 126.0 |
| L0 down | 2 | 2182.1 / 1505.3 / 136.2 | 2178.0 / 1500.2 / 131.1 |
| L0 down | 4 | 2177.0 / 1632.3 / 144.4 | 2175.0 / 1627.1 / 148.5 |
| L0 gate_up | 1 | 2681.9 / 448.5 / 203.8 | 2761.7 / 461.8 / 202.8 |
| L0 gate_up | 2 | 2697.2 / 452.6 / 214.0 | 2711.6 / 469.0 / 213.0 |
| L0 gate_up | 4 | 2690.0 / 465.9 / 230.4 | 2717.7 / 478.2 / 229.4 |
| L22 down | 1 | 2096.1 / 1500.2 / 135.2 | 2084.9 / 1502.2 / 126.0 |
| L22 down | 2 | 2093.1 / 1507.3 / 132.1 | 2087.9 / 1502.2 / 124.9 |
| L22 down | 4 | 2274.3 / 1636.4 / 152.6 | 2282.5 / 1628.2 / 148.5 |
| L22 gate_up | 1 | 2676.7 / 448.5 / 205.8 | 2733.1 / 463.9 / 203.8 |
| L22 gate_up | 2 | 2692.1 / 449.5 / 216.1 | 2761.7 / 462.8 / 210.9 |
| L22 gate_up | 4 | 2689.0 / 465.9 / 230.4 | 2762.8 / 479.2 / 231.4 |
| L43 down | 1 | 2082.8 / 1499.1 / 129.0 | 2079.7 / 1499.1 / 123.9 |
| L43 down | 2 | 2095.1 / 1507.3 / 133.1 | 2090.0 / 1501.2 / 127.0 |
| L43 down | 4 | 2180.1 / 1634.3 / 144.4 | 2176.0 / 1628.2 / 140.3 |
| L43 gate_up | 1 | 2697.2 / 448.5 / 204.8 | 2736.1 / 461.8 / 202.8 |
| L43 gate_up | 2 | 2693.1 / 449.5 / 213.0 | 2736.1 / 462.8 / 212.0 |
| L43 gate_up | 4 | 2682.9 / 465.9 / 229.4 | 2750.5 / 479.2 / 229.4 |

### Per-operator / M decision values (median across the three layers and three pairs/cell)
These are descriptive centers over corresponding layer fixtures, not substituted for the layer-specific table.

| projection | M | Eager old / R1 / c57 | old→R1 change | R1/c57 | Graph old / R1 / c57 |
|---|---:|---:|---:|---:|---:|
| down | 1 | 2086.9 / 1499.1 / 129.0 | -28.2% | 11.62× | 2084.9 / 1499.1 / 126.0 |
| down | 2 | 2097.2 / 1507.3 / 133.1 | -28.1% | 11.32× | 2089.0 / 1501.2 / 127.0 |
| down | 4 | 2180.1 / 1632.3 / 144.4 | -25.1% | 11.30× | 2176.0 / 1628.2 / 148.5 |
| gate_up | 1 | 2681.9 / 448.5 / 204.8 | -83.3% | 2.19× | 2736.1 / 461.8 / 202.8 |
| gate_up | 2 | 2693.1 / 449.5 / 214.0 | -83.3% | 2.10× | 2736.1 / 463.9 / 211.0 |
| gate_up | 4 | 2689.0 / 465.9 / 230.4 | -82.7% | 2.02× | 2745.3 / 479.2 / 229.4 |

## Correctness and independent evidence disposition
- R2 bound Eager strict-output checks to exact old/R1 extension SHA being timed; all 54 E035 arm executions had strict-bad=0 and finite FP16. The old raw Graph fields establish eager/Graph equality, but its graph_oracle_bad label is **not** a full Graph strict pass; the exact comparison was an ambiguously named cast-boundary check. Corrected R1 and parallel L43/down M4 Graph checks in the two `r2-graph-correction-*.json` artifacts establish FP32 go finite/strict, cast equality and Eager/Graph equality for those exact binaries. No full-graph matrix remeasured.
- Existing independent complete R1 correctness matrix remains valid: 18/18 fixture×M strict oracle, repeat/independent scratch, Graph1/2/4; high-amplitude L43/down ×4 strict pass and finite cast. Existing separate immutable-correctness SHA/files stay valid.
- Old and R1 NCU/SASS are independent one-binary processes/artifacts and remain valid: actual old stack-array LDL/STL and R1 no LDL/STL; matched NCU M4 gate_up 313.95MB→0.355MB DRAM writes, kernel 2.74ms→526.62us. These are instrumented single-kernel data, not substituted for Eager/Graph Linear timings.
- Withdraw only the old prior conclusion that full Linear was unchanged by R1; correct isolated execution shows strong R1 reductions, strongest for gate_up. Retain the separate c57 rejection: R1 remains slower than exact c57 in every tested complete-Linear projection/shape. No whole-Forward, service, TPS/TTFT or production claim.

## L43/down M4 full-call cost decomposition (2026-10-17, single CUDA-profiler-range trace)
A UUID-locked Nsight Systems CUDA profiler range captured exactly one warmed Eager full Linear invocation per arm (warmup/strict checks before the capture range), target-only `CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, fixture `43.mlp.down_proj`, `M=4,K=12288,N=3072`. SQLite `TARGET_INFO_GPU` binds the device to Tesla V100-SXM2-16GB, UUID `eaeb0cc6-58aa-8413-bef3-7fb075992189`, PCI `0000:26:00.0`, GV100; each `TARGET_INFO_CUDA_CONTEXT_INFO` has `deviceId=0` and the matching process is the arm Python. The `cuda_gpu_trace` CSV's display label says “NVIDIA GeForce RTX 2060 (0)” because it reports CUDA ordinal text, but this is not physical-device identity: `TARGET_INFO_GPU` records V100 as device id 1 / CUDA ordinal 0 (`cuDevice=0`) and RTX2060 as device id 0 / ordinal 1 (`cuDevice=1`). Independent `torch.cuda.get_device_properties(0).uuid` was the target V100 UUID. Therefore bind the trace to V100 using UUID + PCI metadata, not the mislabeled CSV display string. No RTX2060 workload was launched by these commands; a desktop GPU process existed there and was not disturbed.

| arm | Captured kernels and grids | summed GPU kernel time | relevant allocation/cost |
|---|---|---:|---|
| R1 | gather 3.264 µs `[192,1,1]`; `group_wmma_fused` 1,848.954 µs `[192,1,1]`; cast 2.176 µs | 1,854.394 µs | group kernel 99.7%; 48 regs/thread; R1 cubin STACK0/LOCAL0 |
| Parallel | gather 3.200 µs `[192,1,1]`; `group_wmma_partial` 183.167 µs `[192,1,96]`; `reduce_groups_ordered` 16.704 µs `[48,1,1]`; cast 1.888 µs | 204.959 µs | 18,432 group CTAs; 48 regs/thread, 2,048B shared/CTA; scratch logical 4,718,592B FP32 |
| exact c57 | `packed_fp32_partial` 124.831 µs `[24,1,96]`; `packed_fp32_reduce<half>` 28.608 µs `[48,1,1]` | 153.439 µs | two kernels; first 56 regs/thread |

These are one-call Nsys GPU kernel-duration sums, not a timing distribution or host wall time; no repeated NCU was run. Exact parallel-minus-c57 decomposition for this trace only: partial `183.167-124.831 = +58.336 µs`; reducer `16.704-28.608 = -11.904 µs`; gather/cast `(+3.200+1.888)-(3.264+2.176) = +5.088 µs`; net `+51.520 µs` (204.959−153.439). The reducer is faster, not additional cost versus c57. Although four kernels are shown for parallel and two for c57, kernel-count difference does not measure launch latency difference; no launch-latency attribution is claimed. Against R1, parallel reduces group-kernel duration by 1,665.787 µs; these are kernel-duration sums only.

Same-cell prior complete-Linear paired distribution (20 CUDA events per arm × 3 rotated pairs; median-of-pair-medians) independently measured Eager R1/parallel/c57 `1632.3/184.3/144.4 µs` and Graph replay `1628.2/187.4/148.5 µs`. Thus this trace's ~205/153 µs kernel sums are consistent with, but do not replace, those event results; their difference includes event/timing boundary and run-to-run conditions. Full-cell paired raw and SHA remain in the E035 artifact record.

Evidence identity: run_id `29b996e4-c6f9-4b2b-85db-aa9be5d014a1`; manifest `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-cost-l43-down-m4-final.29b996e4-c6f9-4b2b-85db-aa9be5d014a1.manifest.jsonl`. The preserved original manifest has 11 physical newline-delimited records, all parseable; however its three `arm_start` objects were never independently recorded because the runner emitted a literal backslash-n, concatenating each start object into the following `arm_end` physical line. The three `arm_end` records retain valid embedded `ready` stdout JSON and exact arm/binary SHA; stats records and final run_end remain. Original artifact unchanged, SHA256 `bb51d00fcc0b393fea6aff7304409f65434b63eec039a8a782f030b5d285bdcb`. A CPU-only fake-arm regression reproduces the malformed pattern and then exercises the exact shared `record_arm_start` writer used by the runner; every repaired physical line is parsed with `json.loads`. No GPU evidence was rerun. Reports (SHA256): R1 `.nsys-rep` `26a5e00ad1593729fcad0f3a6c761d5a67785b28547598c4d90e8d48a8ddcdb5`, SQLite `b088f78804e44ba5421c3948b77678c34fbc9c492bbc96c9c5f2f15916480d9e`; parallel `.nsys-rep` `4a364ab47df88c56700a08064b2fd1352a2de126acff1abe1fb1ba8621bf8ffa`, SQLite `1e969bc7f3f8e28fb272300048c1e57dcacb60f85379660dfeaa83c03f479b27`; c57 `.nsys-rep` `59afde706e9ff0e7ab1fcbf4efbfece394e5089929fad1dcb00bb0f17b7b90a9`, SQLite `3f4fd2ccd9ef5e5ea59f684257899a7962c0a8b99ca1b90d7bfbc0797fbd2785`. The manifest records candidate/R1/c57 executable SHA `f72b7646…` / `23ea9c87…` / `c57e50dc…`, lock owner and timestamps. Raw Nsys outputs are outside Git in the designated artifact store.

## Final decision and next action
**NO-GO for Forward/production.** The candidate's lower R1 cost and strict correctness do not offset its loss to exact c57: full Linear Eager and Graph lose in all nine tested down layer×M cells (about 1.28–1.41× slower). L43/down M4 decomposition pinpoints the measurable gap: parallel-WMMA partial 183.167µs vs c57 packed partial 124.831µs, plus 16.704µs ordered reduction and another kernel launch. No Forward/service/cache evidence was collected for this rejected candidate. Per the user's explicit decision, record **Owner REJECTED for Forward/production** (this is a scoped rejection of this candidate, not a claim that its arithmetic is incorrect); keep the candidate isolated and undispatched. Preserve S1 default, exact c57 opt-in/rollback and production dispatcher unchanged. The unique next action for this candidate is stop: no further tuning, no matrix repeat, no Forward/Cache run. A different design requires new explicit authorization and must first beat exact c57 under the frozen oracle before any integrated qualification.
