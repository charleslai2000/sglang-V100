# E043 — T009 c57 Q4 uint4 load experiment

## Question
Does combining the exact-c57 partial kernel's four contiguous per-thread Q4 `uint32_t` source loads into one aligned `uint4` load reduce SASS load instructions and improve complete-Linear latency, without changing Q4 layout, CTA mapping, K order, FP32 arithmetic, split/reducer semantics or output contract?

## Hypothesis and falsification
For each K row, one thread owns four adjacent N outputs. The source computes `row*N+n0+j`, `j=0..3`, with N divisible by 128 and `n0=blockIdx.x*128+threadIdx.x*4`. Thus each thread's qweight byte address is `base+4*(row*N+n0)`; fixture qweight allocation is at least 16B aligned, while `row*N+n0` is divisible by four. Expected alignment is 16B. Scalar source has four uint32 loads; compiler may combine them or emit four instructions. Hypothesis fails if old SASS already has one 128-bit load, candidate does not produce an equivalent one-load operation, output differs, or load/traffic/resource/timing gates fail.

## Identity and frozen conditions
- Exact historical source: G017 commit `5fac74206b55464dca5993d943120746a24b959a`, SHA256 `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`.
- Frozen reference binary SHA256 `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`; E026 `g017-close-final-rebuild.log` ties this to the historical source and CUDA 12.8 / torch 2.9.1+cu128 build. Do not rebuild historical qualification.
- Target only Tesla V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; host MemAvailable >=8GiB. Preserve S1/default, exact-c57, E035 dirty file and G023 hold.
- Candidate is isolated under E043, SM70 only, and changes only Q4 load expression. Initial comparison uses real L43 fixtures, fixed arithmetic, partial+ascending reducer+FP16 cast. No numerical repair unless >=12% weighted M4 complete-Linear threshold and M1/M2 gates pass.

## Method
1. Decode exact binary SASS and parameter slots against host launch signature/source. Record load widths, pointer alignment and source equivalence.
2. Build isolated candidate only; hash source/binary, extract SASS/resource usage, inspect spills and occupancy.
3. Validate candidate output exactly against native-FP16 c57 (and FP32 diagnostic only against independent oracle where required); do not use rounded FP16 for pre-cast strict pass.
4. Run at least three genuinely alternating c57/candidate fresh-process pairs on gate_up/down M1/M2/M4, Eager and Graph complete Linear; include partial, reducer, and cast. Use same arithmetic path and normal clocks.
5. Bounded E040-style ApplicationReplay on decisive gate_up M4 for SASS load instructions, DRAM/L2, stalls, duration, registers/local/occupancy. Preserve the prior E042 PC sampling; do not repeat it. No Falcon-wide replay.

## Measurements
Actual source/binary hashes; SASS load sites/widths; pointer alignment; exact output equality and frozen-oracle diagnostics; load instruction counts; registers, local memory, occupancy; NCU kernel-scope DRAM/L2/stalls/duration; complete-call Eager/Graph A/B; weighted M4 result and M1/M2 regression gate. Results and bounded conclusion are in `results.md`; hashes in `artifacts.sha256`. Current outcome: exact load combined in SASS, but operator threshold failed; further qualification is conditional and closed for this implementation.