# G022/T012 — Compressed-Q4 Tensor Core Prefill

Status: **ACTIVE — implementation not started; no performance/correctness result claimed.** Owner-authorized follow-on to T011's Prefill PERFORMANCE/NUMERICAL-NO-GO. T011's FP32 full-weight reconstruction and dual-FP16 residual routes are closed and must not be revived. T011 QPN2 Decode remains independently qualified only at its recorded binary identities; no production ACCEPTED designation.

## Objective

Implement Falcon GPTQ Prefill at M=512/1024/4096 directly from compressed Q4 on SM70 Tensor Cores. Avoid full FP16/FP32 weight materialization, avoid pre-casting a potentially out-of-FP16-range raw projection, and retain the required FP32 semantics through group scaling, group reduction and Falcon down multiplier before the final FP16 cast.

## Frozen constraints

- Work only on existing `main` and `/data`; no branch/worktree. Preserve unrelated dirty files, S1/default, historical c57 assets, T011 Decode QPN2 source/dispatch, and G023 ON_HOLD.
- Only V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; abort GPU runs if host `MemAvailable < 8 GiB`. Never use RTX 2060. No CUDA global policy/driver changes.
- Keep T012 experimental/default-off. Do not relax frozen strict oracle `1e-4 + 5e-5*abs(reference)`, alter GPTQ-v1 zero+1/group128/nonidentity `g_idx`, fold scales/multiplier early, or transfer evidence across binary SHA changes.
- Do not alter Decode QPN2. Reuse established prepack/K permutation/quadpair/multiwarp ownership where technically suitable. No full dense weight replica, no global `[G,M,N]` partial. Use output-tile ownership and bounded on-chip/group reduction suitable for large M.
- Do not start the capacity or service campaign until the Prefill numerical and performance gates pass. If the authorized design fails after a bounded evidence-led correction, stop and report decisive NO-GO; do not accumulate speculative variants.

## Required implementation semantics

1. Each GPTQ code is unpacked with the original packed ordering. Effective integer operand is exactly representable `FP16(q - (qzero + 1))`; qzero is the stored GPTQ-v1 zero nibble, with the established +1 convention.
2. `g_idx` is nonidentity and defines group ownership. Every group has 128 K elements; the prepack/gather must preserve original activation-to-weight association.
3. Tensor Core group products accumulate in FP32. Apply each original group scale in FP32, reduce group contributions in FP32, apply `down_multiplier` in FP32, and cast to FP16 only at the original final output boundary.
4. For large M, CTA ownership must avoid serial underutilization and avoid materializing all group partials globally. Document scratch bounds, shared memory/register footprint, launch count and any preprocessing.
5. Packed Q4 weights may use a validated compact prepack, but no persistent FP16/FP32 dense weight copy. Account for packed prepack and permutation memory and load-time cost separately.

## Sequential gates

### Gate A — source/build identity and mapping

- Record source diff and exact source/binary SHA; build explicit SM70 and execute only on assigned V100.
- Validate every mapping against checkpoint raw GPTQ tensors: nibble ordering, zero+1, group scales, `g_idx` permutation, Q4 prepack roundtrip. Identify scratch and memory ownership.
- Leave QPN2 Decode binary/source untouched and default path unchanged.

### Gate B — 512-token correctness

- Use identical controlled real checkpoint, positions/tokens and initialized state.
- Independently reconstruct FP64 reference using original GPTQ semantics on captured finite inputs (never FP16-reconstructed weights as oracle).
- Qualify true L0/L22/L43 down projections and the actual first-error L43/down case, then all 44 down layers. Strict bad must be zero; all intermediate and final outputs finite.
- Run CUDA Graph capture/replay with independent outputs/scratch and verify strict oracle, repeat, eager/graph and final cast behavior. Regress first observed native overflow boundary and ensure corrected output ordering avoids pre-multiplier FP16 overflow.
- Any nonzero strict-bad or mapping/Graph defect blocks progression.

### Gate C — 1K/4K correctness

- Actual real-checkpoint 1K and 4K model Prefill using same frozen arithmetic and state setup. Validate layer/hidden/logit boundaries, strict comparisons where oracle is defined, finite outputs, Graph replay, repeats, multiplier/cast and first-error regression. Record exact token/input hashes.

### Gate D — three-arm Prefill performance and resource evidence

Same host, GPU, checkpoint, shape, warmup, synchronization and sampling protocol; alternate arms and preserve raw per-sample records:
A. native FP16 route, performance reference only (known numerical failure);
B. correct FP32 route, slow numerical control (already measured; do not recompute unless a controlled artifact is invalid);
C. new compressed-Q4 Tensor Core route.

For 512/1K/4K report full Prefill CUDA Event and host submit/sync/wall distributions; per-kernel names/counts/durations; relevant DRAM bytes and Tensor instruction evidence/counters at matching boundary; allocator and NVML peak; load-time pack/preprocess wall/device time, bytes, and persistent/temporary memory. Separate instrumented profiler runs from uninstrumented latency. Confirm UUID and `MemAvailable >=8 GiB` before and during runs. Require no repeatable >3% full-Prefill CUDA Event regression against native reference; correctness is not waived because native itself is numerically invalid.

### Gate E — automatic downstream closure after A–D pass

Without seeking another routing decision, run the outstanding T011 model-state/hidden/logit checks, Prefill→Decode, Mamba/KV request isolation, cancellation and slot reuse; verify 16-GiB capacity without reducing the frozen Decode capacity target; then execute three alternating identity-matched service A/B groups at 1K/4K × c1/c2/c4 using the frozen original gates. Report actual admitted GPU batch separately from client concurrency and report Decode/Output TPS, TTFT, ITL/TPOT, E2E, Prefill TPS, VRAM and correctness. Do not transfer service acceptance across binary SHA.

## Initial prototype checkpoint (not a milestone pass)

`experiments/E044-g022-sm70-rebuilt-c57/prefill_q4_tc.py` is the first experimental implementation; no production integration/default change. The original dtype/shape mismatch is fixed: prepack emits packed bytes viewed bitwise as `int32 [N/32,K/16,32,2]`, matching the CUDA `uint2` consumer; qzeros are `int32 [G,N/8]` and `g_idx`/permutation are int32. This preserves every Q4 nibble without quantized-value conversion. New `test_prefill_q4_tc.py` checks encode/decode roundtrip and compares kernel outputs to an independent FP64 original-GPTQ reference.

On assigned V100, actual L43/down fixture (`K=12288, N=3072`, manifest SHA `7c0057edb888d0b4c3730c717a6c3359ff48e07ea439714f39653d4149c48271`) roundtripped all 37,748,736 Q4 codes bitwise; prepacked size 18,874,368 bytes. SM70 binary SHA `c8a0ade8eae01a52df4a703189fca47e9eaeb34718fa365efc41b9523699cf48`, source SHA recorded by Git after commit. SASS contains `HMMA.884` and cubin target `sm_70`; SPLITK 8/16/32 report REG64 and shared 8/16/32 KiB, stack/local zero. Fixture M4 strict FP64 eager and CUDA Graph oracle passes for ordinary and saved 4× stress input, repeat/fresh scratch/Graph cast equal, strict_bad=0; max tolerance ratio 0.412535. Independent runner log `/data/t011/artifacts/commonops128/t012-prefill-q4-build-roundtrip.log` SHA256 `e390284eddf43cfa4b4bd2bf08dc063c18f5e23b83bf1b2caf8e394ffa7a348e`; frozen QPN validator JSONL `/data/t011/artifacts/commonops128/t012-qpn-validation.jsonl` SHA256 `c24f85d20b568f6688ff4a26daafe4b56285c3c4d0f62764735b993595783f6f`. These are component/fixture results only, not 512/44-layer/model/1K/4K acceptance or performance.

### M-tile experiment update (intermediate, not full Prefill acceptance)

An initial template over-expanded the shared partial across split-K and exceeded SM70's 48KiB CTA shared-memory budget at most 32/64-row combinations. A bounded implementation change made CTA own multiple 8-row HMMA fragments while sharing packed Q4 reads within each group iteration; the M64 test uses split4 to keep shared storage 32KiB. This source currently includes this work-in-progress extension; it remains experimental and must be requalified after final source identity is frozen.

On actual L43/down Q4 (`K=12288,N=3072,G=96`), M4 (fixture input) and generated finite controlled M512/1024/4096 inputs were independently checked against original GPTQ FP64. The fixture M4 tested tile8/split8,16,32; tile32/split8; tile64/split4. At M512/1K/4K tile64/split4, eager and Graph strict-bad=0, Graph FP16 cast equals eager, max tolerance ratio about 0.509/0.519/0.519. These generated activation rows are operator stress/scale checks, not actual model activation captures or model correctness. The 512/1024/4096 oracle logs above are included as strict-bad=0 operator evidence only.

A nine-sample uninstrumented single-Linear CUDA Event diagnostic at M512 showed tile8/split16 2.518ms, tile32/split8 2.050ms, tile64/split4 1.939ms (−18.6% / −23.0% vs tile8). At M4 the same tiles measured 0.0799/0.1280/0.2345ms, so large tile is harmful for small M. This doesn't attribute bytes or establish end-to-end speed. cuobjdump for this binary reports tile8 split16 64 registers/16KiB shared, tile32 split8 127 registers/32KiB, tile64 split4 239 registers/32KiB; stack/local zero. The high register count for 64-row ownership predicts low occupancy and needs Nsight/whole-prefill validation; 32-row tile may be the safer reuse candidate. No launch synchronization or kernel fusion assumption is made.

Prototype evidence paths and SHA256:
- `/data/t011/artifacts/commonops128/t012-prefill-q4-mtile-real.log` `2d24e9c9f6bfa16b3b69d353ee989d501d781be8bc06c43bf3b1b1d81debb0dc`
- `/data/t011/artifacts/commonops128/t012-prefill-q4-m512-oracle.log` `9c5ddbf2aac68405074136d927bc6aefeb30e162435941c38174b6714f9925cf`
- `/data/t011/artifacts/commonops128/t012-prefill-q4-m1024-oracle.log` `fa4d835c42d05d55f10a4b85badd9c2debcbc85bdb4b4b28dc74d387c1bee79a`
- `/data/t011/artifacts/commonops128/t012-prefill-q4-m4096-oracle.log` `b35ba9e279a8f264db101a3f92bf975db800238f6c46587802f322e00a5be982`
- `/data/t011/artifacts/commonops128/t012-mtile-benchmark.log` `79e975996c796d7719fdfd7afb46375d6afc33cfb3d1f12ec120346d8e6dd258`

## Stop rule / deliverables

After one architecture implementation and at most one bounded, source-backed repair to that implementation, if strict correctness, >3% regression, 16-GiB capacity, or service gates fail, stop with evidence-backed NO-GO and root cause. Do not implement unrelated candidates or weaken gates.

Commit/push real source, tests, control records and evidence before each substantive progress report; verify `HEAD == origin/main`. Final report must identify actual binary SHA, source diff, full Prefill/Decode/service result matrix, numerical and performance risks, artifact paths/hashes and READY-FOR-OWNER or decisive NO-GO. Never mark Owner ACCEPTED.
