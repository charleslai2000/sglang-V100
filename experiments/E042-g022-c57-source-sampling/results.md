# E042 results — exact-c57 SourceCounters

## OBSERVED

### Collection and raw identity
All four reports used the E040 bounded route with E042-only copied target/wrapper, `E042_SECTIONS=SourceCounters`, one exact-c57 selected `packed_fp32_partial` kernel, four total ApplicationReplay passes (2 metrics/replay passes as NCU grouped source metrics; four target restarts due sampling), strict match and file buffer. Every target emitted matching exact extension, fixture and output hashes. No sampling overflow; host MemAvailable remained >13.6GiB; cgroup RAM peak <0.75GiB; zero memory events; target returned idle. Run records:

| Cell | Report path | SHA256 | PC samples | Passes aggregated | Interval cycles | Dropped bytes | sampled long scoreboard | short | wait(not classified) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gate_up M1 | `/mnt/data/pc01/sglang-v100-migration/experiments/E042/reports/c57-gate_up-m1-20261017T235936.ncu-rep` | `8b0f85c10aeeee25bba9d2c00c59d264ebf7a8814a3be681442b56aed9c77727` | 10,451 | 2 | 2048 | 0 | 3,078 | 4 | 1,987 |
| gate_up M4 | `/mnt/data/pc01/sglang-v100-migration/experiments/E042/reports/c57-gate_up-m4-20261018T000124.ncu-rep` | `7094978707b47a9142051573b71fdbe7fe3f9a73413d3978f357e79814aa5058` | 11,389 | 2 | 2048 | 0 | 3,768 | 2 | 1,980 |
| down M1 | `/mnt/data/pc01/sglang-v100-migration/experiments/E042/reports/c57-down-m1-20261018T000138.ncu-rep` | `f4c66a976fcd9ea987ebe88d08b2f909d6f0c75116523fe09d3b53f32d9e66d1` | 5,559 | 2 | 2048 | 0 | 1,565 | 0 | 1,131 |
| down M4 | `/mnt/data/pc01/sglang-v100-migration/experiments/E042/reports/c57-down-m4-20261018T000150.ncu-rep` | `8a9fc7ef9c1fa1547783064f0812995cecdedaab0dd2498202bbc7f3c1230153` | 6,082 | 2 | 2048 | 0 | 2,099 | 2 | 1,167 |

PC sample counters are sampled events, not execution counts or wall time. The `long_scoreboard` component is large among sampled warp states, short scoreboard nearly absent. This is consistent with a long-latency data dependency but does not identify the producer uniquely.

### PC/source observations and causal limits
- Source-page exports are preserved as `/mnt/data/pc01/sglang-v100-migration/experiments/E042/export/*-source-page.csv`. With `stall_long_sb` parsed (not inferred from generic sample counts), long samples by cell are 1,565–3,768; the hottest relative instruction is `SHF.R.U32.HI` at `0x...f930` (gate_up M1 2,002/3,078, M4 2,323/3,768; down M1 897/1,565, M4 1,106/2,099). Next repeated PCs include `IMAD R44,...` at `0x...f810` and `SHF ...` at `0x...f870`. These are PC-sample/stall observations, not execution counts or latency.
- The exact full SASS at `/mnt/data/pc01/sglang-v100-migration/experiments/E040/export/c57-full.sass` has a repetitive main loop. Around relative PCs `0x04c0–0x06f0`, metadata and g_idx-derived values feed addresses and shifts. The Q4 word path is explicit: four `LDG.E.U16.CONSTANT.SYS` at `0x0560 R50,[R26]`, `0x0590 R48,[R26+2]`, `0x05a0 R49,[R26+4]`, `0x06f0 R43,[R26+6]`; values are consumed by `SHF/LOP3` extraction (e.g. 0x0570/0580 and 0x05b0–0x0600). The `SHF` PC sampled hottest (0x...f930) is not uniquely identified as one semantic producer simply by sampling; dynamic g_idx, qzero/scale, activation and Q4 streams overlap in the loop, and consumer-side stalls can arise from multiple outstanding dependencies. Address/register lineage shows plausible paths, not an exclusive latency-critical producer attribution. A selective producer intervention or per-dependency latency proof is absent.
- No temporal stall→load latency attribution is possible from SourceCounters PC samples; do not equate ~30% sampled long state with recoverable runtime. L1/L2 hit/sector and memory-workload analysis from SourceCounters are SW/source stats and earlier E040 aggregate traffic only; exact producer-cache hierarchy attribution remains unresolved.
- Source sample quality is sufficient to show reproducible hot PC regions: 5,559–11,389 samples per cell, 2 aggregated passes, 2,048-cycle interval, 0 bytes dropped. However pass aggregation and low absolute kernel duration, uncontrolled caches/clocks, plus single-launch fixture geometry limit precise PC percentages and cross-cell ordering.

### E041 audit corrections
- Corrected `experiments/E041-g022-c57-n-tile-parallelism/results.md` and T007 task: final A/B runner order was fixed c57→candidate for each repeat, not alternating. Raw JSON is unchanged and remains fixed-order observations.
- The final A/B JSON `oracle`/precast fields compare FP64 mathematical values to native FP16 output, so they are not strict pre-cast FP32 correctness evidence. Separate corrected typed FP32 diagnostic on L43 gate_up M1 (`strict_bad=0`) remains a one-cell diagnostic only. No full matrix is claimed. The E041 PERFORMANCE-NO-GO is unchanged. Source results CSVs are preserved in E042 export.

## INFERRED
- Exact-c57 has dependency-bearing memory producers and sampled consumers in the K loop; candidate prefetch would need to hoist future K's dynamic g_idx/Q4/metadata accesses while preserving order. But current evidence cannot identify a bounded 2/4-step lookahead that safely shortens the critical dependence chain; blindly adding loads risks higher register pressure, bandwidth/sector traffic, and cache pollution.
- Therefore no prefetch prototype is justified under this evidence. No software intervention or source/binary candidate was built. This is a bounded **ARCHITECTURE-NO-GO for the proposed lookahead under current producer attribution**, not a claim that all c57 optimization is impossible.

## UNRESOLVED
- Which specific source (g_idx, qzero, scale, Q4 code, activation) dominates sampled long-wait consumers; PC sampling is correlation, not exact data-dependency tracing.
- No matched cache-state / source-pointer intervention, NCU per-PC load sectors or hardware PC sampling sequence proving load→consumer latency was collected.
- No prefetch candidate, operator A/B, full strict numerical matrix, model/Graph/service test; no performance claim. Those are not warranted absent identified producer mechanism.
- Historical “15.88ms on RTX 3090” run provenance/backend/weight format/accumulator was not located in current checkout or referenced E035/G022 control artifacts. Status: **UNVERIFIED**; no 3090 scheduled.

## Bounded answer
**ARCHITECTURE-NO-GO for ungrounded lookahead prefetch at this stage.** Safe SourceCounters sampling on all four exact-c57 L43 cells shows repeated long-scoreboard samples concentrated around Q4 load/bit-extract and later activation/scale/FMA PCs, but cannot uniquely distinguish producer load chains. No supported 2/4-step lookahead distance follows from these data; do not implement one speculatively. E041 order/numerical-field corrections are documented without rerunning its matrix. T008 has no kernel implementation, no numerical qualification, and no model/service benefit claim. S1/default and exact c57 remain unchanged; no Owner ACCEPTED.
