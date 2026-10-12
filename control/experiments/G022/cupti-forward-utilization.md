# G022 — Falcon Forward utilization captures

## Result and scope

Captured the requested Falcon-H1-7B-Instruct-GPTQ-Int4 S1 Forward ranges on V100 GV100, UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, for Prefill EXTEND 1K/4K and Decode Graph1/2/4 (actual batch 1/2/4). Each workload has **three complete, matched, seven-pass UserReplay metric-vector captures**. L2 throughput has an additional two complete 15-pass captures per workload. The earlier one-pass captures remain historical evidence but are superseded for repeated core metrics by the later complete vectors below.

The CUPTI Range Profiler API executes the real Forward once per requested metric pass using ApplicationReplay and an immutable token fixture; between invocations the existing harness rebuilds the Falcon request/cache state. Each pass asserts fixture identity, expected actual batch/input, pass index/configuration and final `allPassSubmitted`; completed runs report zero dropped ranges. This is **application replay**, not NCU Kernel Replay. The captured counter range is the named Forward annotation. CUDA Events are emitted around the real Forward invocation and listed separately; with multipass capture they vary by replay invocation, so the table reports the final (counter-producing invocation) event bracket and does not imply a standalone unprofiled latency. Replay pass durations are not performance baselines.

No NCU, DCGM, or Kineto was collecting simultaneously; no global profiler pause, driver/policy change, or RTX 2060 use occurred. A 2-second supervisor enforced `MemAvailable >= 8 GiB`. The lowest accepted guard sample among the completed multipass runs was above 13 GiB; every run exited successfully and target V100 returned idle. The Prefill EXTEND range remains eager (no graph); Decode range label confirms requested Graph1/2/4 key. Harness workload outputs use 8 generated tokens; these are hardware-utilization observations, not 256-token service benchmarks.

## Repeated matched core vector

Metric names and CUPTI formula outputs are preserved literally. For each row, each cell shows three observations; `mean` is arithmetic mean of the three single-range values (not a time series). Read/write counters retain the returned `dram__bytes_*` names and raw values. Their physical byte interpretation is **not validated**, so no physical HBM bandwidth or %peak is calculated from them. `DRAM throughput` is the independent CUPTI formula metric and is reported as such.

| Workload (range; actual batch) | Forward CUDA Event, ms (rep1 / rep2 / rep3) | SM Active % | SM throughput %peak | Active warps / occupancy %peak | Tensor pipe active %peak | DRAM throughput %peak | L2 sector hit % | Effective DRAM BW GB/s (rep1 / rep2 / rep3; mean) | Effective DRAM BW / nominal 900 GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill EXTEND 1K; bs1 (`FalconPrefillGraph1`) | 285.71 / 285.12 / 284.84 (mean 285.22) | 92.115 / 92.265 / 92.455 (92.278) | 41.802 / 41.854 / 41.901 (41.853) | 37.347 / 37.381 / 37.392 (37.373) | 41.228 / 41.279 / 41.326 (41.278) | 25.687 / 25.693 / 25.847 (25.742) | 86.816 / 86.843 / 86.852 (86.837) | 229.7 / 230.2 / 230.4 (230.1) | 25.57% |
| Prefill EXTEND 4K; bs1 (`FalconPrefillGraph1`) | 1116.41 / 1128.11 / 1114.83 (mean 1119.78) | 97.011 / 97.043 / 97.141 (97.065) | 45.202 / 45.203 / 45.194 (45.200) | 33.072 / 33.084 / 33.125 (33.094) | 44.788 / 44.789 / 44.780 (44.786) | 23.661 / 23.618 / 23.928 (23.736) | 85.413 / 85.417 / 85.336 (85.389) | 212.5 / 210.0 / 213.7 (212.1) | 23.57% |
| Decode Graph1; bs1 (`FalconDecodeGraph1`) | 30.07 / 30.50 / 30.50 (mean 30.36) | 69.577 / 76.058 / 77.425 (74.353) | 23.584 / 26.040 / 26.471 (25.365) | 47.598 / 47.603 / 47.605 (47.602) | 0 / 0 / 0 | 30.253 / 29.173 / 30.211 (29.879) | 66.975 / 66.984 / 66.980 (66.980) | 271.3 / 267.5 / 269.4 (269.4) | 29.93% |
| Decode Graph2; bs2 (`FalconDecodeGraph2`) | 33.10 / 33.11 / 33.85 (mean 33.35) | 74.503 / 76.413 / 76.153 (75.690) | 25.440 / 25.869 / 25.968 (25.759) | 44.767 / 44.781 / 44.759 (44.769) | 2.086 / 2.121 / 2.129 (2.112) | 27.874 / 27.821 / 28.592 (28.096) | 58.159 / 58.146 / 58.162 (58.156) | 257.8 / 257.7 / 246.6 (254.1) | 28.23% |
| Decode Graph4; bs4 (`FalconDecodeGraph4`) | 36.60 / 36.38 / 35.65 (mean 36.21) | 73.231 / 78.499 / 80.536 (77.422) | 25.319 / 27.309 / 27.765 (26.798) | 44.152 / 44.189 / 44.129 (44.157) | 1.800 / 1.942 / 1.974 (1.905) | 25.693 / 28.656 / 29.326 (27.892) | 60.197 / 60.148 / 60.189 (60.178) | 253.6 / 255.0 / 260.2 (256.3) | 28.48% |

Raw DRAM counter values (bytes returned by CUPTI), in rep1 / rep2 / rep3 order:

| Workload | `dram__bytes_read.sum` | `dram__bytes_write.sum` |
|---|---|---|
| Prefill EXTEND 1K bs1 | 40,149,017,856 / 40,164,446,272 / 40,152,429,568 | 25,482,769,248 / 25,482,391,104 / 25,482,964,448 |
| Prefill EXTEND 4K bs1 | 169,433,918,848 / 169,083,727,168 / 170,528,576,640 | 67,762,909,472 / 67,763,511,936 / 67,763,364,192 |
| Decode Graph1 bs1 | 7,995,274,400 / 7,995,196,384 / 7,995,816,096 | 162,050,272 / 162,000,704 / 162,670,432 |
| Decode Graph2 bs2 | 8,195,301,024 / 8,196,156,000 / 8,196,228,352 | 337,268,896 / 337,980,384 / 338,162,976 |
| Decode Graph4 bs4 | 8,599,623,744 / 8,599,316,768 / 8,598,016,000 | 681,033,056 / 680,559,136 / 679,701,216 |

**Interpretation caveats.** SM Active, SM throughput, active warps, Tensor pipe active, DRAM throughput and L2 hit rate are separate hardware/formula metrics, not interchangeable “utilization” measures. In particular, `sm__cycles_active...` is not `sm__throughput...`. Graph1 SM Active showed a comparatively wide three-run range (69.58–77.42%); retain observations and mean rather than selecting the highest value. Zero Tensor pipe-active is the measured CUPTI result for those ranges, not an unavailable field. All five requested workloads now have Tensor/occupancy coverage.

## L2 throughput

`lts__throughput.avg.pct_of_peak_sustained_elapsed` needs 15 passes on GV100; two complete matching ApplicationReplay captures were made per workload. Results, mean of two:

| Workload | Rep1 %peak | Rep2 %peak | Mean %peak |
|---|---:|---:|---:|
| Prefill EXTEND 1K bs1 | 37.3992 | 37.3699 | 37.3846 |
| Prefill EXTEND 4K bs1 | 37.4321 | 37.4716 | 37.4519 |
| Decode Graph1 bs1 | 21.1376 | 21.0470 | 21.0923 |
| Decode Graph2 bs2 | 18.7271 | 20.0923 | 19.4097 |
| Decode Graph4 bs4 | 19.6049 | 20.3443 | 19.9746 |

These are CUPTI throughput formula percentages. No L2 bytes/second is derived. Separate replay ranges and run IDs are provided below. The two measurements are repeat observations, not a 100-ms timeline or peak series.

## HBM/bandwidth interpretation

For each accepted core-vector run, effective range-average DRAM bandwidth is calculated as `(dram__bytes_read.sum + dram__bytes_write.sum) / (Forward CUDA Event seconds)`. NVIDIA Nsight Compute's metric glossary defines `dram_read_bytes`/`dram_write_bytes` as `dram__bytes_read.sum`/`dram__bytes_write.sum` (official CLI metric table); its profiling glossary defines a sector as an aligned 32-byte chunk. The `.sum` rollup is a sum across instances of the relevant hardware unit, not an additional sum over ApplicationReplay passes; the CUPTI CounterData result is taken from the single named range once `allPassSubmitted=true`. NVIDIA's Volta V100 data sheet specifies 900 GB/s peak bandwidth for this 16-GB V100 configuration. Sources: [Nsight Compute CLI 12.8 metric table](https://archive.docs.nvidia.com/nsight-compute/2025.1/NsightComputeCli/index.html), [Nsight Compute Profiling Guide metric terminology](https://docs.nvidia.com/nsight-compute/2024.2/ProfilingGuide/index.html), [NVIDIA Tesla V100 data sheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf).

The independently reported `dram__throughput.avg.pct_of_peak_sustained_elapsed` provides a consistency check: e.g. Prefill1K mean byte/event bandwidth 230.1 GB/s predicts 25.57% of 900 GB/s, versus directly measured CUPTI DRAM throughput mean 25.742%; Prefill4K 212.1 GB/s predicts 23.57%, versus 23.736%; Graph1 269.4 GB/s gives 29.93%, versus 29.879%; Graph2 254.1 GB/s gives 28.23%, versus 28.096%; Graph4 256.3 GB/s gives 28.48%, versus 27.892%. Close agreement across all five distinct phases corroborates that the byte counters and event denominator are coherently interpreted for these ranges. These effective values are range-average observed traffic, not a time-series peak, and the ratio to 900 GB/s is a fraction of nominal peak, not a claim of sustained peak capability.

Raw read/write inputs remain available above and in supervisor logs for audit. No latency-based kernel owner attribution is made from these range totals.

## Raw evidence and reproducibility

Target UUID above, Falcon model `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4`, model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`, S1 runtime, CUDA/CUPTI 12.8, `tilelang_fa_v100`, `--disable-radix-cache`, `--disable-piecewise-cuda-graph`, `--max-total-tokens 8192`, `--mem-fraction-static 0.75`, input IDs fixed per workload fixture, `--output-len 8`, `--profile-stage prefill|decode`, profile step 2 only. Prefill graph is disabled and Extend runs eager; Decode graphs are explicitly capped and requested at actual batch 1/2/4. Exact token fixture SHA256: Prefill1K `0453fd2f3a4ed6e3640a443b52eec9380f690831b091564eb00338147403f5b4`; Prefill4K `453a69f31a531a6c71b778c40534a3cc6b4a4f3d71e9492ace7838afef1e9dc0`; Decode bs2 `f9b090974fbc1383bc3715d07c3d750ccbadce0d3d0c0795409d53ef2cb67a64`; Decode bs4 `cca7ebdec6b2b70680510f9a3817c3ab79498e580a7ef84ba94ed0c02fbdbd93`.

Runner `/tmp/g022_cupti_runner_multpass.py`, SHA-256 `ca133dd11c70981c4acb50947fa3e438868caae56d6383db40360a0044a66f9f`; seven-metric wrapper `/tmp/g022_cupti_cell_core_multpass.sh`, SHA-256 `ec09da59780d5794fa3108cdda879555df71aae303e03651e80285416ac44b7f`; L2 throughput wrapper `/tmp/g022_cupti_cell_l2throughput.sh`, SHA-256 `66f8824c09bed65498517e544d1215e560bfc49c113941e557e1189074b516f8`.

Core vector log IDs (each final metric range has `all_passes_submitted=true`, zero dropped ranges, and `cupti_application_replay_complete=true`):

| Workload | Rep1 log — SHA-256 | Rep2 log — SHA-256 | Rep3 log — SHA-256 |
|---|---|---|---|
| Prefill1K | `g022-cupti-core-prefill1k-rep1-20261013T080000` — `9253e0abc31c8af3b3bed5a6fd5b0554247727dbe641f8c998f7d90c4998e2cf` | `g022-cupti-core-prefill1k-rep2-20261013T081000` — `ced58e650962692d1c35fc6c5e64551e58634ff2ccea86f3baebd7620255de98` | `g022-cupti-core-prefill1k-rep3-20261013T170000` — `bac9d674cabedea7693707147e2b2c703c3f5e4674df274038f49586eb0754eb` |
| Prefill4K | `g022-cupti-core-prefill4k-rep1-20261013T082000` — `af1bc8da712d4144386b0528750615dc6e51d80e7806bc9998cc22adc8e29287` | `g022-cupti-core-prefill4k-rep2-20261013T083000` — `1968c0e52481b6f0a6d82081d0be09dcff75e6e28508ccb9c3669996844a7f74` | `g022-cupti-core-prefill4k-rep3-20261013T173000` — `e053de2b6ae6cd199ef55a00f0a3e892a3fd88d4c9fe558eafe1dc6ce699fcb5` |
| Decode G1 | `g022-cupti-core-decode-g1-rep1-20261013T073000` — `33fefcf2f54def84bc4c685a05d3ef8cda85d438223ab2b6ed47e33d1030ce75` | `g022-cupti-core-decode-g1-rep2-20261013T150000` — `725f966a4af24b841649f34f089a2597e807e1602d3b9d3538264572df342147` | `g022-cupti-core-decode-g1-rep3-20261013T153000` — `f1b3339e943b0de0a882bf06248064d111a65f7d5ddf79d4a7eb3036443db7fb` |
| Decode G2 | `g022-cupti-core-decode-g2-rep1-20261013T090000` — `ab9b83745ff2d3029487d3673ff88c7998c081c80156ec3c1911545876413ab7` | `g022-cupti-core-decode-g2-rep2-20261013T091000` — `1d08d2c7ab7bd529d2d75381395d59f07c7ee8bf7c84681be84a064afd4daefd` | `g022-cupti-core-decode-g2-rep3-20261013T160000` — `a3a24bd9498a11bfaa2437075c6b1830d95dc90675d2aac3c73ebe1f6bd13e96` |
| Decode G4 | `g022-cupti-core-decode-g4-rep1-20261013T092000` — `66dab7b52cb2f67f7e71bb80aaa6b19f705e1ca94f09c2f5482a9965d08f6364` | `g022-cupti-core-decode-g4-rep2-20261013T093000` — `86bdacedc8d42f9c81a2ae774fd309fd4ad24df4a0a0b18f9905238078d019fd` | `g022-cupti-core-decode-g4-rep3-20261013T163000` — `67b3cfb0564e4fc6301da78ad38057bce387888d3131738b29d480213e307bb1` |

L2-throughput log IDs (same replay completion checks): Prefill1K rep1 `g022-cupti-l2throughput-prefill1k-rep1-20261013T100000` SHA `abd3cd7b889dbb932a06a56ae09199a0c086666ef8e0826352dd84877e48b558`, rep2 `...rep2-20261013T123000` SHA `04d6fd5682456f141567e487773dbb4533e34400e9fa41f90dada0c507535c1f`; Prefill4K rep1 `...prefill4k-rep1-20261013T103000` SHA `35ef5f86adfcd5343923b4642598ecc510b6d4780ee872cfba37a0509b8f80af`, rep2 `...rep2-20261013T130000` SHA `5bbfc6ec83bc20b5f8bad19f6b5f32d2dc5684b76c223348d3e67254309d6e4a`; DecodeG1 rep1 `...decode-g1-rep1-20261013T110000` SHA `e225658b095625dbdd0468e1d2ddd6b6d8e5dd451da1bcc748e8ed96dc7954bc`, rep2 `...decode-g1-rep2-20261013T133000` SHA `1baff7ae5c3616ff605c5e5e8c253bca16d1e53ef3067e1528905437643a9c54`; DecodeG2 rep1 `...decode-g2-rep1-20261013T113000` SHA `abc6f8386295d16023b1725e808fa110cc483074ff903b66d9bbc2fef6aca5fa`, rep2 `...decode-g2-rep2-20261013T140000` SHA `27594492c7f4ccefab3452121b00807f5fed090771e87aba996124fa1514cc71`; DecodeG4 rep1 `...decode-g4-rep1-20261013T120000` SHA `991b9997f5d24b8c00225670dcd17facabbf662a4ad75ea55e777c372f9234da`, rep2 `...decode-g4-rep2-20261013T143000` SHA `f40f4f3f0c41d6b1c2de0d8737cd670a6a722243552966bcef9e801fa6f1bf9e`.

All `.supervisor.log` and `.results.jsonl` raw files reside in `/mnt/data/pc01/sglang-v100-migration/{logs,experiments/G022}` respectively; exact log ID above omits `.supervisor.log`. Earlier partial and one-pass results are not deleted. This report is the authoritative interpretation layer; raw values remain the source data.

## Still unavailable / not claimed

- Physical HBM effective GB/s, effective % of nominal peak from byte sums: withheld pending unit/aggregation validation. DRAM throughput %peak itself is measured.
- L2 bytes/s, individual L1, FP32/FP16 pipe activity, instruction issue and stall reasons: not captured by this seven-metric group and not substituted with kernel-level NCU values.
- A device-wide 100-ms timeline, peak values, 100-ms means, or repeated time samples: not captured; these are repeated Forward-range captures with no time series.
- Per-kernel or exclusive owner budgets: this is aggregate annotation scope; it does not isolate GPTQ/GEMV/attention contributions.
- Historical single-pass and earlier incomplete/aborted replay captures are retained but do not replace the valid complete matched vectors above.
