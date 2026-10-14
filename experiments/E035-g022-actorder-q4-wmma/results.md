# E035-R2 — down group-parallel result

## OBSERVED

### Background-task completion and post-overlap clean repeats
The original `r2-paired-down-full.jsonl` run contained duplicate keys because two invocations overlapped; preserve that raw file and its reconciled selection (`r2-paired-down-reconciled.jsonl`) rather than treating duplicates as extra repeats. After the six user-notified background tasks completed, performed a serial clean 18-row supplemental matrix for the cells whose earlier rows overlapped: L0/down and L22/down × M1/M2/M4 × 3 cyclic pairs, 20 event iterations/arm. Artifact `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-paired-down-clean-repeats.jsonl`, SHA256 `b19558730cc89b4ab645e2e2c54e1819c428cdda8edc2c7864c767c689f86ea7`. All R1 and parallel arms passed frozen strict FP32 oracle; actual FP32 Graph go was finite/strict-bad 0; final graph output equaled `go.half()`; Eager/Graph match; c57 SHA and PIDs recorded.

Clean-repeat median-of-pair-medians, µs:
| Fixture | M | Eager R1 / parallel / c57 | Graph R1 / parallel / c57 |
|---|---:|---:|---:|
| L0 down | 1 | 1716.2 / 205.8 / 146.4 | 1714.2 / 207.9 / 143.4 |
| L0 down | 2 | 1504.3 / 181.2 / 132.1 | 1500.2 / 183.3 / 131.1 |
| L0 down | 4 | 1631.2 / 184.3 / 145.4 | 1628.2 / 187.4 / 148.5 |
| L22 down | 1 | 1499.1 / 180.2 / 147.5 | 1498.1 / 186.4 / 144.4 |
| L22 down | 2 | 1504.3 / 180.2 / 132.1 | 1498.1 / 183.3 / 124.9 |
| L22 down | 4 | 1634.3 / 186.4 / 145.4 | 1628.2 / 188.4 / 149.5 |

These independent serial measurements confirm the direction and no c57 win in the overlapped cells. L43/down pair results were never overlapped: M4 has two complete triplets (the explicit causal set plus the full run); M1/M2 have three each. Original E035/R1 parallel/c57 binaries are unchanged.
- **Source/symbol/path:** experiment-only `grouped_q4_sm70_parallel.cu`, public entry `grouped_q4_sm70_parallel`, internal `group_wmma_partial` + `reduce_groups_ordered`; no production dispatcher change. Source SHA `75cd105c7ed82de68d5810417e99f2237bf4cda8edfe58b772c3c224886ce521`. Binary SHA `23ea9c87cbea6d6f6d491a94a5ef2446c363c7367d573d0807ee2358aa26bbcd`. Same R1 `f72b764654de643384166f44acf7924fd8199a6512f9573bb921224394b2392d`, exact c57 `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`.
- **Mechanism:** R1 serial grid for M4/down was `[192,1,1]` single warp CTA and group loop iterated 96 groups. Candidate launches `[192,1,96]` = 18,432 independent single-warp CTAs, each does one group dot result; stores unscaled FP32 group result for the actual M rows only; reducer per output executes g=0..G−1 `fmaf(partial[g], scale[g], acc)`. Same gather, group Q4 prepack, g_idx, zeros/scales, within-group WMMA and output cast. No FP16 partial, atomic, tree/parallel reduction or tolerance modification. Only down projection measured/touched in experimental entry point.
- **Baseline NCU evidence:** new R1 matched one-launch report, exact down M4 L43 fixture, exact binary. R1 group kernel grid `[192,1,1]`, block `[32,1,1]`, 48 regs/thread; SASS resource output from cubin STACK0/LOCAL0; 1.879ms, DRAM R/W 19.590/0.363 MB, cumulative active warps field 1.477M, issue 172,101 cycles, long scoreboard 47.34%, Tensor warp instructions 4,718,592. Raw report SHA `4fab9fa2a98d3d1eee2b03d5ce6c03d6461019b07e15604abb91e88af0c95a33`; NCU log SHA `4787ec22cc3860799b5cefe8c1a922ef2e8c0970b98668befef5c9a2680c16e5`.
- **Candidate NCU:** exact same fixture and selected group partial only, grid `[192,1,96]`, 48 regs/thread, cubin STACK0/LOCAL0, 183.680µs, DRAM R/W 18.982/3.718 MB, cumulative active warps field 1.777M, issue 166,939 cycles, long scoreboard 26.61%, Tensor instructions same 4,718,592. This confirms more scheduled CTAs, lower long scoreboard and shorter individual group-kernel duration; R/W traffic is the counter, not logical bytes. 4.719MB partial storage at M4 is logical FP32 volume; NCU measured 3.718MB writes for the profiled WMMA kernel and ~18.982MB reads. It excludes separate reducer/gather/cast traffic. Candidate report SHA `a85afd9fc28cc1ee52f937b759cf8309509e624dd6ef1da57ba7ebaf345cf253`; log `cbdd26da0d6cdb3ef106e94458a941ff73d19263e541d690ac38d5754cfd208d`. NCU reports are kernel-only and do not imply complete-call speedup.
- **Candidate correctness:** all L0/L22/L43 down M1/M2/M4 = 9/9 strict FP32 oracle pass, actual candidate binary SHA bound in rows; FP64 GPTQ reference -> FP32, abs(error)<=1e-4+5e-5*abs(ref), strict-bad=0. Final FP16 finite. Same-scratch repeat and independent-scratch exact output. Captured Graph1/2/4: actual uncast FP32 `go` finite, strict oracle bad=0, final graph `gy == go.half()`, eager and graph cast output equal. L43/down M4 input ×1 and ×4, input max 2420/9680, strict-bad=0, finite FP32/FP16.
  - L43/M1-M4 artifact `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-parallel-l43down-m4.jsonl`, SHA `464c442d5b4f55c487e4bc6e353858997c40252d1e56a053aed9c29ffd6c8912`.
  - L0/down artifact `r2-parallel-down-l0-l22.jsonl`, SHA `2efbc77d23475e5d8203f7ef6103ffc7a016939b51cbd4009d9252712843a80e`.
- **Full Linear comparison (old omitted per explicit instruction not to repeat the invalid old arm):** isolated-process script SHA `6e1f39b634b6dd49bf049ea7dc90eff35952ff9b77eadfa2d2007086de229dca`; exact R1/candidate/c57 SHA logged by child; three rotated order pairs ×20 event iterations; one arm per interpreter, same input/M/cache warm setup; complete call includes gather + group products/partial + reducer + final cast. The run orchestration accidentally overlapped two starts in the same output JSONL for L0 M1/M2/M4 and L22 M1. The source JSONL is preserved as written, SHA `13bc0b33e6fcd37b27597f57a17d2772a8da8e1a8293ffd28691ac9c7d267e76`; a separate audit lists duplicate keys. Reconciled primary keeps the earliest complete 3-pair set per `(fixture,M,pair)`, retains non-overlapped L22 and L43 data, and is SHA `f8b7a33bbd3a27c12cbec2d03ff7394a98c42b5055084677bb512098e62fbbb6`; see adjacent `.audit.txt`. No binary identity conflicts: duplicate run cells each preserve exact per-arm SHA and PIDs; only one V100 workload ran at any time after overlap was caught and stopped. Do not treat duplicate rows as more repetitions.

Median of three pair medians, microseconds, Eager and Graph replay separate, R1 / parallel / exact c57:

| Layer | M | Eager R1 / parallel / c57 | Graph replay R1 / parallel / c57 |
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

Primary result limitations: L43/down M1/M2/M4 stored in separate `r2-paired-down-all.jsonl`, SHA `f011bcd2365da9738c77c9ca54106e50b893af824b3aa8442a25d96949dbbc7f`; the first measured L43 M4 is from `r2-paired-down-m4.jsonl`, SHA `92abad4ef5691fdaf4100ad7031cb4baa4124e970eba67d104cb3833feef8546`, but full matrix captured another complete L43 M4 triple too, which is the row used above. The full reconciled file contains 27 geometry×pair observations; each row includes 3 PIDs/arm, binary SHA, script SHA, strict correctness, output dtype and Eager/Graph times. L43 M4 uses the separate full 3-pair causal file for the missing row. The uncorrected `r2-paired-down-full.jsonl` is preserved and marked by adjacent `r2-paired-down-full.jsonl.audit.txt`; it contains 36 rows and duplicate pair keys from overlapping invocations. The reconciled `r2-paired-down-reconciled.jsonl` SHA `f8b7a33bbd3a27c12cbec2d03ff7394a98c42b5055084677bb512098e62fbbb6` keeps the earliest complete tuple per key; combined primary full matrix is 27 deduplicated cells/pairs plus separate 2,3-pair L43 M1/M2 and L43 M4 files. Artifacts are analyzed without overwriting raw. R1/candidate/c57 identities remain separately bound.

**R2 graph-field correction detail:** original R2 `graph_oracle_bad` compared returned FP16 `gy` to the oracle variable rather than actual FP32 Graph scratch. `graph_oracle_bad=0` in old 54-row artifact established neither finite `go`, strict FP32 `go` oracle nor `gy==go.half()`; do not label it Graph strict PASS. It did test Eager/Graph output equality. Corrected script SHA `6e1f39b...` targeted same R1/parallel L43/down M4 and verified `graph_go_finite=true`, strict bad=0 on go, cast bad=0, Eager/Graph mismatch=0, dtype float16. Raw correction files and exact SHAs are described in `report.md`. No full matrix rerun.

Across all 9 fixture×M down cells, group-parallel is ~1.28–1.41× slower than exact c57 in Eager; it is markedly faster than R1 (about 8–9×) but this does not meet the criterion. No measured c57 win in any M or layer.

## INFERRED
- The supported mechanism hypothesis was real at this selected kernel boundary: CTA availability grows 96× and measured long-scoreboard share drops 47.34→26.61%, with group WMMA duration 1.879ms→0.184ms. However new group partial store traffic appears; 3.718MB measured writes for this selected kernel, then reducer must read logical 4.719MB and traverse 96 ordered FMAs per output. Complete Linear never recovers enough to beat c57.
- This architecture is faster than R1 but not exact c57. It has no Forward/throughput/service benefit established.

### L43/down M4 full-call cost attribution
A UUID-locked Nsight Systems CUDA profiler range captured exactly one warmed Eager full Linear invocation per arm (warmup/strict checks before the capture range), target-only `CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, fixture `43.mlp.down_proj`, `M=4,K=12288,N=3072`. SQLite `TARGET_INFO_GPU` binds the device to Tesla V100-SXM2-16GB, UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, PCI `0000:26:00.0`, GV100; each `TARGET_INFO_CUDA_CONTEXT_INFO` has `deviceId=0`, and the matching process is that arm's Python. The `cuda_gpu_trace` CSV display label says “NVIDIA GeForce RTX 2060 (0)” because it reports a CUDA ordinal label; it is not physical-device identity. `TARGET_INFO_GPU` records V100 device id 1 / CUDA ordinal 0 (`cuDevice=0`) and RTX2060 device id 0 / ordinal 1 (`cuDevice=1`). Independent `torch.cuda.get_device_properties(0).uuid` was the target V100 UUID. Bind the trace to V100 using UUID + PCI metadata, not the misleading CSV display string. An unrelated desktop process existed on RTX2060 and was not disturbed.

| arm | Captured kernels and grids | summed GPU kernel time | relevant cost |
|---|---|---:|---|
| R1 | gather 3.264 µs `[192,1,1]`; `group_wmma_fused` 1,848.954 µs `[192,1,1]`; cast 2.176 µs | 1,854.394 µs | group kernel 99.7%; 48 regs/thread; R1 cubin STACK0/LOCAL0 |
| Parallel | gather 3.200 µs `[192,1,1]`; `group_wmma_partial` 183.167 µs `[192,1,96]`; `reduce_groups_ordered` 16.704 µs `[48,1,1]`; cast 1.888 µs | 204.959 µs | 18,432 group CTAs; 48 regs/thread, 2,048B shared/CTA; scratch logical 4,718,592B FP32 |
| exact c57 | `packed_fp32_partial` 124.831 µs `[24,1,96]`; `packed_fp32_reduce<half>` 28.608 µs `[48,1,1]` | 153.439 µs | two kernels; first 56 regs/thread |

These are one-call Nsys GPU kernel sums, not a timing distribution or host wall time; no new NCU profiling was run. The selected complete-call gap is explained by two directly measured costs: parallel WMMA partial kernel is 58.336 µs slower than c57 packed partial; then parallel adds a 16.704 µs ordered reducer and another launch. Parallel total kernel sum exceeds c57 by 51.520 µs (33.6%); reducer is 8.15% of candidate kernel sum. Against R1, parallel removes ~1.666ms from group compute but adds reducer and launch; the WMMA partial remains 89.4% of its total. Do not overstate this as a hardware-counter stall attribution of the c57/parallel difference.

Same-cell paired complete-Linear results (20 CUDA events/arm × 3 rotated pairs; median of pair medians) are Eager R1/parallel/c57 `1632.3/184.3/144.4 µs`, Graph replay `1628.2/187.4/148.5 µs`. These distributions independently show parallel loses c57; Nsys is one representative cost decomposition and does not replace the A/B.

Run ID `29b996e4-c6f9-4b2b-85db-aa9be5d014a1`; capture runner SHA256 `bf5becaa828db8ed235591b2cf6cc43c821caf9836100924015e9595ae7bc562`, workload capture script SHA256 `ef8a1bb367abd662d21e10b75514d3a3f82ab76bbb9cb816e9d06c0792ef6437`; manifest `/mnt/data/pc01/sglang-v100-migration/experiments/E035/r2-cost-l43-down-m4-final.29b996e4-c6f9-4b2b-85db-aa9be5d014a1.manifest.jsonl`. SHA256: R1 `.nsys-rep` `26a5e00ad1593729fcad0f3a6c761d5a67785b28547598c4d90e8d48a8ddcdb5`, SQLite `b088f78804e44ba5421c3948b77678c34fbc9c492bbc96c9c5f2f15916480d9e`; parallel `.nsys-rep` `4a364ab47df88c56700a08064b2fd1352a2de126acff1abe1fb1ba8621bf8ffa`, SQLite `1e969bc7f3f8e28fb272300048c1e57dcacb60f85379660dfeaa83c03f479b27`; c57 `.nsys-rep` `59afde706e9ff0e7ab1fcbf4efbfece394e5089929fad1dcb00bb0f17b7b90a9`, SQLite `3f4fd2ccd9ef5e5ea59f684257899a7962c0a8b99ca1b90d7bfbc0797fbd2785`. Raw outputs are in designated artifact storage, not Git.

## UNRESOLVED / omitted
- No service, full Forward, model layer/logit/token, long decode, request-state, KV/Mamba/cache restore, Prefill, context sweep, VRAM model capacity, or production graph qualification; do not claim none regress.
- R2 old/R1/c57 graph field semantics were incomplete before update; old raw data remains preserved, field limitations are stated in `report.md` and T002 control. Targeted graph corrections exist; no full Graph matrix rerun.
- The Nsys call trace is one warmed invocation per arm; no distributions or counter data were collected in this capture. Existing NCU is selected-kernel scope; it does not measure c57 or candidate complete-call counters.

## Bounded decision
**Owner REJECTED for Forward/production** (user-directed scoped decision for this E035 down group-parallel candidate). The numerical contract passes and independent CTAs improve on R1, but exact-c57 complete-Linear A/B loses in all 9 layer×M cells; the single-cell cost trace locates the representative gap in the slower partial kernel plus ordered-reducer/launch overhead. Unique next step for this candidate: stop—no further tuning, Forward/service or Cache run; retain isolated candidate undispatched. S1 default, exact c57 opt-in/rollback, and production dispatcher remain unchanged.
