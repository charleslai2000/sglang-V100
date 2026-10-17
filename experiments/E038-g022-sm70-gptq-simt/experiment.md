# E038 — G022 T004 SM70 GPTQ group-major FP32 SIMT candidate

## Question
Can a new act-order-aware group-major Q4 format plus FP32 SIMT grouped dot product and bounded deterministic reduction beat the exact G017 c57 complete GPTQ Linear calls on V100 for actual Falcon M1/M2/M4 gate_up/down while satisfying the frozen independent FP64→FP32 oracle including L43/down M4×4 stress? If yes, does it survive Graph/model/service gates; if not, decide a bounded numerical or performance NO-GO.

## Authorization and exclusions
User directly authorized implementation in this experimental area, numerical/performance tests and, only after complete-op gate passes, isolated opt-in full-model/service qualification. No production default/dispatcher change, no E035 edits, no Cache/G023 work, no RTX2060. Compare exact S1 and exact c57 identities. Strict pre-cast FP32 contract is unchanged: `abs(error) <= 1e-4 + 5e-5 * abs(reference)` vs independent FP64 GPTQ-v1 dequant/GEMM → FP32. All failures are fatal; no tolerance weakening, FP16-only substitute, reduced precision, or output-token-only numerical gate.

## Initial mechanism audit (before implementation)
- Current source: `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py::GPTQLinearKernel.__init__` initializes `use_shuffle=True`; `process_weights_after_loading` sorts `g_idx` and calls `gptq_shuffle` only when this flag is true.
- Falcon SM70/S1 actual path in prior E028/G017/E037 evidence sets `use_shuffle=False` (checkpoint is `desc_act=true`, but SM70 fallback keeps legacy unshuffled `g_idx` semantics); thus actual active path has not done loader-time stable group-major sorting/repack. This is confirmed by current source and exact active S1 extension SHA; exact candidate CUDA runtime process introspection is still a required qualification check.
- Existing alt4 already stages packed Q4 weights into shared `blockvec` across up to 8 M rows and prepares Q4 decode table in shared memory. It also reuses four Q4 values/paired scales and zero terms per inner unrolled iteration for a K tile, but still reads each word of each output column and performs FP16 half2 math/FP16 partial atomic merge.
- Exact-c57 is an explicit guard in this unshuffled path; it loads Q4/qzero/scale/g_idx, computes 128-K FP32 partials and an ascending-partition sequential FP32 sum reducer (not Kahan). It is not a loader shuffle/repack. Preserve its exact binary and oracle-reported ×4 failure as risk. No candidate may be called new if it reuses c57 partial/reducer unchanged.
- E035 numeric FP64 prototype and E035 group-serial/group-parallel/packed-word candidates are rejected and must not be revived. E028 pinned SM70 Marlin is incompatible with Falcon desc_act runtime metadata and qzero contract. Donor concepts may be used only for Q4 layout or GEMV vector/warp reduction; no direct Marlin/WMMA design.

## Candidate status (implemented as isolated experimental source; not production-integrated)
A separately named experimental custom op exists in `actorder_group_q4_simt.cu`; CPU conversion reference `reference_group_major.py` and strict fixture harness `test_candidate_fixture.py` are local experimental files. The transformation stable-partitions original K indices by `g_idx`, repacks all 8 Q4 nibbles to group-major `[G,128,N/8]`, preserves group-indexed qzeros/scales, and gathers FP16 activation by the same permutation. Current host pack path copies raw qweight/g_idx to CPU and back; its model-loading cost and peak host/device residency have not been measured.

Kernel maps 4 K-warps × 32 contiguous N lanes, accumulates 32 K per warp in FP32, combines four partials in fixed order per group, applies original FP16 scale in FP32, and writes four bounded group-range partial planes followed by a deterministic ascending-plane FP32 final reduction. No FP16 intermediate accumulator, FP64 production math, or WMMA. This preliminary scalar-Q4/SIMT kernel was strict-correct on tested fixtures but decisively failed complete-Linear performance: see `results.md` PERFORMANCE-NO-GO. Each CTA still loops sequentially over 6 gate_up or 24 down groups within its assigned group-range plane. No NCU/model/service gate was run after this decisive failure. No further tuning is authorized under T004. No strategy depends on exact-c57's arithmetic association.

## Procedure / gates
1. Pin git/source/binary/runtime provenance; record active extension realpath+SHA, model config, all six fixture manifests, CUDA/PyTorch, V100 UUID. Inspect actual `GPTQLinearKernel.use_shuffle`, parameter shapes/strides, qzero+1 and actual `g_idx` permutation in real loaded S1 and candidate process.
2. CPU reference converter: exact reconstruction mapping checks on nonidentity group IDs and arbitrary zeros, all real fixtures; assert packed layout and activation permutation invert exactly. No CUDA/production source dispatch changes until CPU checks pass.
3. Implement isolated op/binding and process-local opt-in, not default route. Keep modified symbols/files and build flags/hashes. V100 compile only after host ≥8GiB, target UUID/idle, serialized memory-bounded resource plan.
4. Correctness first: all 6 projections × M1/2/3/4 plus L43/down M4×4; strict-bad=0; FP32 pre-cast Graph/eager scratch verified, final FP16 cast, repeatability and independent scratch. Failure details persisted before exit nonzero. If any frozen oracle failure, stop and record NUMERICAL-NO-GO; do not proceed to NCU/service.
5. If correct, full Linear A/B vs exact c57 and S1, >=3 alternating pair cycles per fixture/M; include gather/reorder, main grouped GEMV, all partial/final reductions, FP16 output cast and allocations. Targeted NCU/SASS reports actual ld/st, integer/Q4, FP32, Tensor, registers/local spills, occupancy/eligible warps/stalls and DRAM bytes. If no repeatable net improvement sufficient to plausibly reach c4 5% Forward threshold, make bounded PERFORMANCE-NO-GO and stop before model matrix.
6. Only operator pass: integrate explicit isolated SHA-gated opt-in; actual binary dispatch; Prefill and multi-step Decode; Graph1/2/4; logits/hidden states; concurrent/reuse state isolation; then matched alternating S1/c57/candidate 1K/4K ×256 greedy c1/c2/c4 service and DeviceTimer. Measure actual batch/key/KV, per-step GPU Event/TPS, service TPS, TTFT/ITL/E2E, Prefill, VRAM, supported NCU/CUPTI at precise boundaries. No cache tuning. Never call this production ACCEPTED.

## Final disposition

**PERFORMANCE-NO-GO; T004 DONE.** Exact artifact identities and complete-Linear timings are in `results.md`. S1/default/c57 and production dispatch remain unchanged; no ACCEPTED claim.

## Required results
- `results.md` OBSERVED / INFERRED / UNRESOLVED, all negative outcomes retained.
- Repro scripts and immutable raw artifact pointers + SHA; exact extension path/SHA in every row.
- Complete-op and, if reached, full model/service table with S1, exact c57, candidate; baseline arms are never omitted.
- Candidate may conclude `READY-FOR-OWNER` only if all mandated numerical/model gates pass and actual c4 1K full Graph Decode improves ≥5%, c4 service output TPS improves ≥5%, and c1/c2/4K/Prefill has no repeatable >3% regression without extra VRAM/precision/batch tricks. Otherwise explicit `NO-GO` with the first decisive falsification.
