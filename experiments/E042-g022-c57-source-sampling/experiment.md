# E042 — G022/T008 exact-c57 SourceCounters/PC sampling

## Question
For real L43 gate_up/down M1/M4 exact-c57 `packed_fp32_partial`, which dynamic loads most plausibly precede long-scoreboard sampled consumer PCs, and does the evidence justify software lookahead without changing Q4 layout, CTA shape, K accumulation/reducer, or arithmetic?

## Controlled identity and safety
- Exact extension SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`.
- Fixture-only NCU Application Replay reusing E040 target; physical V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189` only. One selected partial kernel, strict matching, file buffer, transient system cgroup `MemoryMax=4GiB`, `MemorySwapMax=0`, external abort below MemAvailable 9GiB. No Falcon process, RTX2060 or production files.
- NCU 2025.1.1 `SourceCounters` samples per-SASS PC plus warp stall reasons; this is not timing and is not a source-derived load-latency attribution by itself.

## Runs and expected evidence
Four real fixtures/cells: gate_up M1/M4, down M1/M4. Keep per-report SHA, cgroup peak/events, min MemAvailable, exact fixture/input/binary identity. For each report export `smsp__pcsamp_sample_count`, `smsp__pcsamp_interval_cycles`, `smsp__pcsamp_dropped_bytes`, long/short scoreboard and instruction-level source page. Trace consumer register uses backward to prior LDG, and producer address registers back to semantic pointers via exact source/SASS. If dependencies cannot distinguish overlapping pointer streams, state ambiguity. Do not infer recoverable time directly from stall sample fractions.

## Result
See `results.md`. No optimized kernel is authorized unless producer/load-chain evidence establishes a specific useful lookahead.
