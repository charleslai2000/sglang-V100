# E044 — G022/T010 QPN2 SM70 act-order GPTQ adaptation

Status: **ACTIVE — donor and fixture mapping are understood; first exact-QPN2 CUDA prototype matches strict FP64 on L43/down ordinary and 4× stress; full six-fixture correctness and performance are not yet qualified.**

## Frozen identity and boundaries

- Donor: `dnv2003/v100-skinny`, commit `5b589c0dc81223e0ba65bcb3e755874723f8b515`, isolated at `/tmp/t010-v100-skinny` (donor-only location). No formal project source or implementation exists outside this repository.
- Target: Tesla V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; RTX2060 excluded. Before every V100 run recheck MemAvailable >=8GiB and target-only idle/lock. Preserve E035's dirty file.
- Frozen oracle: independent FP64 dequant/GEMM→FP32; `abs(error) <= 1e-4 + 5e-5*abs(reference)`, strict-bad=0. No relaxed tolerance/cast-only proxy.
- S1/default, exact-c57, G023 and E035 sources are unchanged by E044.

## Donor source audit and actual reuse

Reviewed actual `kernels/skinny_kernels.cu` and `fork_patches/marlin.py` at the pinned commit:

- `_qpn_prepack` (`marlin.py:87`) emits `[tile N/32][K/16][lane32][8B]`; `col=((lane>>2)&3)*8+(lane&3)+4*(lane&16)` and K order `[0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]`. This is a reversible index/byte permutation, not NVFP4 dequantization.
- `skinny_nvfp4_qpn2<SPLITK,NACC>` (kernel approximately lines 1395–1508) is one CTA per N=32, 32×SPLITK threads, QP=`(lane>>2)&3`, four 8-column quadpairs with shared A fragments, register decode, four inline `mma.sync.aligned.m8n8k4...f32` ops, optional independent NACC FP32 fragments, and K-split warp reduction through only a small shared FP32 output tile plus one barrier. No shared weight/activation staging in its main loop.
- QPN2 decoder `dequant8_tm` builds interleaved `(i,i+4)` half2 pairs. Its 2^14 rebias is folded into NVFP4 scales. Donor’s scale is E4M3 group16 and does not implement GPTQ group128 scales, g_idx, zero+1 or the frozen FP32 contract. Donor route is M<=8; later M goes to other kernels. Donor 77% read-roofline is not Falcon evidence.
- E044 prepack/decoder now follows donor’s exact lane/K mapping, but stores each original Q4 code as `q-8`, which is exactly FP16 representable for codes 0..15. Every group128 is accumulated separately in FP32, its original independent scale is applied only after the group dot, and scaled group contributions are folded into register-resident FP32 Kahan sum. This avoids E035’s global `[G,M,N]` partial array. Activation gather remains a separate launch and workspace. This is a correctness prototype, not yet an optimized performance kernel.

## E035 distinction

E035 `grouped_q4_sm70_parallel.cu` used activation gather + one CTA per `(group,M-tile,N-tile)`, global `[G,M,N]` FP32 partial writes, and a separate ordered reducer. Rejected for complete-Linear performance. The E044 QPN2 prototype instead uses one N32 CTA with 4 QPs, register fragment MMA, accumulates/scales each group and keeps final accumulation in registers; no shared tile per group and no G×M×N partial array. It does still perform a sequential per-CTA loop over GPTQ groups, so performance must establish enough occupancy/benefit and any better K/CTA partitioning must preserve group scales and not regress to the rejected design.

## Fixture provenance and CPU checks

- L43/down frozen fixture package: `fixtures/packed-word-fixture-l43-down.zip`, SHA256 `e721af60f050db6bd63a6891d4362dd23e68b6bb91744387ca986f73f4b1df16`; original fixture manifest SHA256 `7c0057edb888d0b4c3730c717a6c3359ff48e07ea439714f39653d4149c48271`. Derived ×4 stress oracle SHA256 `da7ee5fa596912cc9ec2ed61f4015b9b46427cd29acc3dd61fc42e49a0f47602`.
- Six E035 conversion dirs (manifest/perm/Q4/scales) were read-only inputs from `/mnt/data/pc01/sglang-v100-migration/experiments/E035/converter`; their asset hashes were captured and used to generate QPN test prepack. The 168MiB duplicate copies were removed after correctness testing; no E035 files were modified. Raw fixture and converter manifest hashes are preserved per-row in the six-fixture result JSONL.
- `audit_fixture.py`: 96 groups ×128 positions; nonidentity g_idx; stored qzero nibble 7 everywhere (effective v1 zero 8); N=3072 divisible by32; K=12288 divisible by64; 37,748,736 Q4 code mapping checks and all 32 QPN output lanes covered.
- `analyze_fp16_weight_rounding.py`: exact-arithmetic FP16(q−8)/FP16 activation/FP32 group scale-and-sum model bit-identical to FP64 oracle on ordinary and saved ×4 inputs. CPU-only and not HMMA evidence.

## CUDA qualification evidence to date

### E035-style WMMA control prototype

`prototype_qpn.py` was first a WMMA correctness control with group partial array and Kahan reduce. It passed L43/down ordinary and ×4 stress, but is explicitly not the QPN2 performance candidate. Initial artifacts remain in `build/`, `results/prototype.json`, `results/prototype-build-run.log`; hashes are recorded in `results/sha256.manifest`.

### QPN2 quadpair kernel

Current harness `prototype_qpn.py` and generated source `build-qpn/e044_qpn2_gptq.cu`:

- Actual SM70 `mma.sync.m8n8k4`, donor prepacked lane/K bytes, q−8 exact FP16 operands, independent FP16 scale per group128, FP32 group accumulator then register-resident FP32 Kahan group sum; no global group partial array.
- Target-only V100, cooperative E035 UUID lock, target idle before/after; MemAvailable >8GiB. RTX2060 untouched.
- L43/down M4 ordinary: strict-bad=0/12,288; max abs error 0.00048828125; max tolerance ratio 0.1277115; finite; repeated call bitwise equal; CUDA Graph FP32 scratch strict-bad=0; casted Graph output equals Eager cast.
- L43/down M4 saved ×4 stress (max |x|=9680): strict-bad=0/12,288; max abs error 0.001953125; max ratio 0.4438063; finite; repeat bitwise equal; Graph FP32 scratch strict-bad=0; cast equality holds.
- Full six-fixture × M1/M2/M4 ordinary matrix plus L43/down M4 ×4 stress: 19 cases, 0 strict-bad and 0 Graph strict-bad, all finite, repeat-equal and Graph/eager FP16 cast-equal. Maximum tolerance ratio 0.4438063 (the ×4 case). Raw JSONL and build/run output `results/six-fixture-correctness.jsonl`, SHA256 `fa9343f430e65ede6437d64656d465c22785d664c709752ceae65520ab88199a` (build/run tee identical output hash); harness `qualify_six.py` SHA256 `d4734b178779f5a44c9e7b27f823f0d87e12b697017d95e774d95a45dbbd7834`. Original fixture manifests and converter manifests are per-row in JSONL.
- `cuobjdump --dump-resource-usage` on candidate: `kern` REG56, STACK0, SHARED0, LOCAL0; `gather` REG12, STACK0, SHARED0, LOCAL0. SASS contains repeated `HMMA.884.F32.F32.STEP0..3` (Volta HMMA) and packed/vector loads, confirming Tensor Core issue. This is initial source/resource evidence, not full NCU utilization/stall profile.
- Candidate source/binary/artifact identities in `results/qpn-prototype.json`; generated CUDA source SHA256 `8c94fcf4e461e3a4d75c733b2eac15af74bf0302bad9b08520ac89921e8c126f`; binary SHA256 `c5a85de210ab1a50f9943efbaaff05ebb85fbaa4daf04a596ba518ca9b1c0035`; JSON SHA256 `55f37a15cc32c00f04d33350408fc4ab2041cc34b7781b57776ff975f33b33e0`; build/run log SHA256 `fe5816f083904494b879f40797d950683c5fd1a083aab4c250f15ea5c6a13749`. Source and binary hashes are in `results/SHA256SUMS`. Build products were removed after inspection; they are reproducible from `prototype_qpn.py`.
- Several earlier prototype iterations failed compilation or oracle (layout errors); logs/results were overwritten during bring-up, so only final passing QPN2 raw JSON/build log and retained earlier WMMA control artifacts are authoritative. Do not cite intermediate failures as candidate numerical counterexamples.

## First full-Linear A/B — L43/down only (decisive PERFORMANCE-NO-GO)

`paired_linear.py` runs each arm in a fresh child under the V100 UUID lock, with three alternating QPN/c57 pair orders for M1/M2/M4. Complete Eager includes QPN activation gather, register MMA, per-group scale/Kahan and output FP16 cast; Graph captures both kernels plus FP16 cast. Both arms use exact target V100 and binary identities (QPN `c5a85d...`, c57 `c57e50...`). QPN strict FP32 oracle and Graph scratch/cast were verified. Timings are CUDA Event medians, 30 iterations per pair.

| M | QPN Eager µs (3 pairs) | c57 Eager µs | QPN Graph µs | c57 Graph µs | QPN Graph / c57 |
|---:|---:|---:|---:|---:|---:|
| 1 | 467.0, 466.9, 466.9 | 146.4, 146.4, 148.5 | 483.3, 483.3, 483.3 | 143.4, 144.4, 144.4 | 3.35× |
| 2 | 466.9, 466.9, 466.9 | 151.6, 150.5, 152.6 | 484.4, 483.3, 483.3 | 150.5, 150.5, 150.5 | 3.21× |
| 4 | 475.1, 475.1, 475.1 | 163.8, 164.9, 163.8 | 495.6, 495.6, 494.6 | 167.9, 167.9, 170.0 | 2.94× |

The candidate is ~2.9–3.4× slower than exact-c57 across all M. The prototype has one CTA per N32 tile (96 CTAs for N=3072) and walks 96 group128 chunks serially, so group-local register MMAs/scaling underfill the V100 despite correct quadpair MMA. A viable speedup likely requires K-slice CTA parallelism and combining independently scaled group results; this carries reduction/traffic risk and would be a different major execution design. The measured route cannot approach the ≥5% whole-Forward potential gate. **T010 QPN2 single-CTA organization is PERFORMANCE-NO-GO; do not run full six-projection timing, NCU or model/service matrix.**

Measured costs: CPU/PyTorch prepack/setup 1.11–1.16 s per fixture invocation in this unoptimized prototype (not production load-time implementation); packed QPN Q4 storage is 18,874,368 bytes for L43/down and original weights remain resident. Runtime activation-gather + FP32 output scratch: 36,864 / 73,728 / 147,456 bytes M1/M2/M4. No second persistent model copy beyond same-size packed Q4 prepack, no production dispatch/default change.

Raw `results/paired-linear.fd5b1132-fb5c-478e-b9af-19eccd229fbf.jsonl`, SHA256 `d5125dd0d02e26af92d5304ca41cade6ce031d29e745155054f93930b0345160`; runlog SHA256 `e65a85873d4bbbdb72f0fa29f6fcbb8d8057ddd381ee8bd6abb1b59195ad646b`. No NCU or service/model opt-in was attempted.

## Closeout

T010 is an evidence-backed **PERFORMANCE-NO-GO** for this QPN2-derived organization. Preserve all correctness/artifact evidence; do not claim owner acceptance. Update control and artifact hashes, commit/push only scoped T010 source/evidence/control while protecting E035 dirty source and unrelated workspace modifications, then verify origin/main SHA.
