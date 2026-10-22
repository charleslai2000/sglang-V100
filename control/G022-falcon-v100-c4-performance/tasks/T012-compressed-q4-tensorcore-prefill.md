# G022/T012 — Compressed-Q4 Tensor Core Prefill

Status: **DONE — CLOSED / PERFORMANCE-NO-GO / NUMERICAL-NOT-FULLY-QUALIFIED.** Final disposition supersedes the prospective implementation gates and intermediate checkpoints below. The implemented T012 route failed the frozen full-Prefill <=3% regression gate. Full-model numerical qualification and measured-binary identity binding remain incomplete. This result applies to this T012 implementation only; it is not evidence that compressed-Q4 Tensor Core algorithms as a class are infeasible.
Work area: experiments
Inputs: Owner-authorized G022/T012 directives; existing T011 QPN2 prepack/quadpair/multiwarp evidence; real Falcon GPTQ checkpoint and frozen strict FP64 oracle; assigned V100 only.
Completion: Implement and assess opt-in compressed-Q4 SM70 Tensor Core Prefill without changing Decode QPN2/default, then pass strict full-model correctness and <=3% matched Prefill regression or close with decisive route-specific NO-GO.

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

### Real Falcon Prefill opt-in smoke (initial integration evidence)

Default-off integration was added only on FalconH1MLP's Prefill EXTEND down-projection; Decode dispatch remains the original route. It reuses `TpModelWorker`/ModelRunner and exact GPTQ layer tensors. The runtime prepack is lazy and stores a compact per-layer Q4 reordered buffer and `g_idx` permutation; activation gather and FP32 projection output workspaces are process-scoped, keyed by `(device,K,N,M)` to avoid M-slice storage overlap. Group scale and CUDA kernel's FP32 group reduction remain before the caller's FP32 down multiplier; final cast is afterward. T012 binary identity is checked through `SGLANG_T012_PREFILL_Q4_SHA256` when supplied.

A fixed real 512-token Falcon ModelRunner forward ran with all hidden/logits finite. Seven uninstrumented CUDA Event samples median 212.385ms (2,410.7 tokens/s); host wall median 212.452ms, peak allocated 10.844GB/reserved 11.799GB. Binary SHA was `1250fd70d420a8416170c3f6d2cfe1f2d6844c67991c0ce2e57a6dc4972ba6d8`; log `/data/t011/artifacts/commonops128/t012-prefill-integrated-512.log` SHA256 `9b93c449f9e3e9d6d1eaeba72b46c056f8cb823b63f556170eb9637c29c9ddd9`. Important: this run's harness does only one random shape warmup and three real input warmups, then measures; it has no matched native/FP32 arms, no independent layer/output oracle, and no graph replay. It is not correctness or performance acceptance. The first integration attempt OOMed during ModelRunner kernel warmup after a max-M workspace was allocated on each of 44 layers; that was fixed by moving workspaces to a shared keyed pool and dynamic per-M slices before the successful run. First attempt log `/data/t011/artifacts/commonops128/t012-prefill-integrated-512.log` was overwritten by successful retry; the failed OOM fact is recorded here but its output is not retained. Current committed opt-in integration itself now needs a binary rebuild and strict requalification because its workspace safety fix postdates the earlier standalone oracle.

### Real Falcon three-arm A/B — performance gate fails; stop downstream work

The existing worker/ForwardBatch harness was extended to run three fresh-process arms in alternating order `native, FP32, T012, T012, FP32, native` at each of 512/1024/4096 actual token lengths, seven warmed uninstrumented CUDA Event trials per process. Native remains performance-only (hidden nonfinite); FP32 remains the correct-but-slow control; T012 hidden/logits finite in all measured rows. Controller samples NVML every 250ms. Full raw `/data/t011/artifacts/commonops128/t012-prefill-ab.log` SHA256 `942b988b690e3a7ea7b4fdb0432eb61ccaf48bb22d3338ebd55b44d1d732e1a1`; NVML `/data/t011/artifacts/commonops128/t012-prefill-nvml.jsonl` SHA256 `991805bd563fd474f0afc5d4beb542dbd4f5daada860f431c734b6d14ce71a20`; controller completed exit 0, empty stdout log SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

| Tokens | Native pass1/pass2 GPU ms | FP32 pass1/pass2 GPU ms | T012 pass1/pass2 GPU ms | T012 slowdown vs native paired |
|---:|---:|---:|---:|---:|
| 512 | 183.602 / 188.064 | 267.472 / 267.171 | 211.800 / 212.121 | +15.36% / +12.79% |
| 1024 | 278.660 / 278.877 | 480.853 / 480.250 | 384.051 / 384.483 | +37.81% / +37.88% |
| 4096 | 1081.611 / 1084.629 | 1893.354 / 1889.486 | 1493.640 / 1493.940 | +38.20% / +37.74% |

Host wall medians tracked event time closely. T012 throughput: 2,415/2,665/2,742 tokens/s at 512/1K/4K, vs native 2,789/3,675/3,787; FP32 1,914/2,131/2,166. T012's allocator peaks were 10.844/10.863/11.304 GB allocated, 11.648 GB reserved; NVML sampled device used peaked at 11,748.5 MiB for T012, leaving 4,635.5 MiB free (native max 11,026.5 MiB). This is not 16-GiB capacity qualification. No T012 run was profiler-instrumented, so these are uninstrumented latency figures.

The T012 route reproducibly misses the frozen <=3% regression gate: approximately 13–15% at 512, 38% at 1K, and 38% at 4K. This is decisive performance NO-GO for this implementation, so the conditional T011/T012 state, capacity and service matrix was not launched. Numerical limits remain separately open at full-model scope: independent strict operator checks passed for the recorded L43/down fixture M4 and generated-input M512/1K/4K cases; actual Falcon captured-activation 44-layer oracle, L0/L22 model-boundary comparisons, first-error full-model regression, and full Eager/Graph hidden/logit checks were not completed. The route is NUMERICAL-NOT-FULLY-QUALIFIED, not a proven numeric failure and not full-model PASS. The actual measured T012 binary SHA was not reliably bound to every timed subprocess; no production-binary qualification is claimed. No complete GPU profiler attribution was collected, so no definitive performance root cause is asserted. T011/T012 joint service matrix is NOT_RUN because T012 failed its performance gate. Do not proceed with further T012 variants or service qualification under this closure.

The A/B harness wrote the T012 binary SHA from its environment override (`06ff...`) even though that override was set before process-local compilation. A post-cohort artifact hash of the build path is `bbdfcc8a24fd046b09ccf7980734d9e271db7968210e77c390933543f132d79f`; because each subprocess may relink the same extension path, this post-run digest is not proof of every measured arm's loaded binary identity. This identity limitation is recorded and no SHA-specific production claim is made. The earlier frozen fixture binary `c848...` and M64 oracle binary `026e...` are separate identities. Source at the current commit is independently SHA-indexed in the delivery report; source/binary identity pairing for these timings is incomplete.

## Final closure summary (2026-10-22)

- **Decision:** CLOSED / PERFORMANCE-NO-GO / NUMERICAL-NOT-FULLY-QUALIFIED.
- **Measured complete Prefill:** native/T012 paired GPU Event medians (ms): 512: 183.602/211.800 and 188.064/212.121; 1K: 278.660/384.051 and 278.877/384.483; 4K: 1081.611/1493.640 and 1084.629/1493.940. T012 repeated regression: ~13–15%, ~38%, ~38%, respectively, exceeding 3% gate.
- **Numerical scope:** strict independent FP64 operator checks passed on L43/down real GPTQ component and controlled M512/1K/4K operator inputs; actual full-model 44-layer captured-activation and L0/L22/L43 hidden/logit/Graph qualification not completed. Do not describe as full model PASS or numeric FAIL.
- **Identity/profiling:** exact running T012 binary SHA was not reliably bound to every timed subprocess; no production binary acceptance. No complete matching-boundary GPU profiler attribution; performance mechanism/root cause remains unproven.
- **Resource:** T012 allocator allocated peak 10.844/10.863/11.304 GB, reserved 11.648 GB; NVML sampled device usage peak 11,748.5 MiB, minimum free 4,635.5 MiB. Not 16-GiB capacity qualification.
- **Downstream:** joint T011/T012 state, 16-GiB and service matrix **NOT_RUN**, because T012 failed Prefill performance gate. T011 QPN2 Decode evidence is preserved at its recorded identities, but production qualification remains incomplete/not ACCEPTED. S1 default unchanged; G023 remains ON_HOLD.
- **Scope:** this is a NO-GO for the implemented T012 route only, not a claim that compressed-Q4 Tensor Core methods in general are infeasible. Preserve implementation/test/artifacts for traceability and rollback. No further Falcon optimization or Gemma testing in this closure.

Primary evidence: A/B `/data/t011/artifacts/commonops128/t012-prefill-ab.log` SHA256 `942b988b690e3a7ea7b4fdb0432eb61ccaf48bb22d3338ebd55b44d1d732e1a1`; NVML `/data/t011/artifacts/commonops128/t012-prefill-nvml.jsonl` SHA256 `991805bd563fd474f0afc5d4beb542dbd4f5daada860f431c734b6d14ce71a20`; completed controller log SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. T012 exact per-run binary binding is unresolved; post-cohort build-path SHA is not promoted as the tested-binary identity.

Final closure supersedes the earlier prospective gates and intermediate evidence wording above. Preserve historical results, source and tests. No Owner ACCEPTED designation.

## Result

T012 is DONE as a route-specific PERFORMANCE-NO-GO / NUMERICAL-NOT-FULLY-QUALIFIED. Three-arm Prefill performance failed the gate at all measured shapes. The 512-token 44-layer actual-activation oracle and model Eager/Graph hidden/logit qualification did not complete; run-time binary SHA was not reliably tied to every timed subprocess. The required joint service matrix was NOT_RUN for the performance-gate reason.

## Remaining

No work remains in this closed task. Do not infer the entire compressed-Q4 Tensor Core design class is infeasible. Any future Falcon optimization requires a separately authorized task; preserve this route's source and evidence, keep default disabled, and do not mark T011 production accepted.
