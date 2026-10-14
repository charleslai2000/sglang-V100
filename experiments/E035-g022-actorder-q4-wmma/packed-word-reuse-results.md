# E035/G022 packed-word Q4 reuse — scoped numerical rejection

## Decision

**NUMERICAL-NO-GO** for this candidate, per the frozen strict FP32 oracle. Do not wire it into SGLang, do not proceed to service/Forward qualification, and do not alter S1/default dispatch or the exact c57 rollback binary. This is a candidate arithmetic-contract failure at the explicitly requested 4x amplitude case, not evidence that ordinary model activations fail. The 4x stress remains in the harness and its failure is intentionally surfaced as a nonzero candidate exit.

## Exact scope and identities

- Intended source change: `sgl-kernel/csrc/gemm/gptq/gptq_candidate_v8.cu`, function/kernel `packed_q4_fp32_candidate` / `packed_fp32_partial`; reused four packed Q4 words across each eight consecutive K values while retaining ascending K, FP32 FMA, K-partition 128, split partial layout and reducer order.
- Rejected candidate source snapshot is preserved at `gptq_candidate_v8_packed_word_reuse.cu`; word-reuse-only diff from the tracked candidate source is preserved as `packed_word_reuse.patch`.
- Candidate extension, exact loaded path `/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-integrated-build/sm70/common_ops.abi3.so`, SHA-256 `2238344028546fb9712ebcddba60f3d9659846a2dfbaad8483db77598e434b64`, is copied to `packed_word_reuse_candidate.abi3.so` (same SHA).
- Baseline is exact accepted c57 runtime binary `/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so`, SHA-256 `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`.
- Fixture: `/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures/43-mlp-down_proj-call0000`, manifest SHA-256 `7c0057edb888d0b4c3730c717a6c3359ff48e07ea439714f39653d4149c48271`.

## Eight-point stress counterexample — same exact data in both arms

The existing test multiplies the fixture FP16 activation by 4 and rounds it back to FP16 (`xs=(x*4).half().contiguous()`), then supplies that same tensor to the kernel and computes the reference from that same `xs`, not from the unscaled activation. Input max absolute value is 9680.0. Baseline and candidate came from separate fresh Python processes, under the V100 UUID lock, with exact loaded paths and SHA recorded. The oracle independently unpacks Q4/qzeros and uses FP64 dequantization/GEMM then casts to FP32; tolerance is exactly `1e-4 + 5e-5*abs(ref_fp32)`.

| (m,n) | baseline actual FP32 | candidate actual FP32 | FP64→FP32 reference | abs error (both) | allowed |
|---|---:|---:|---:|---:|---:|
| (0,80) | -12.611640930175781 | -12.611640930175781 | -12.610892295837402 | 0.0007486343383789062 | 0.0007305446197278798 |
| (0,1034) | 1.358386039733887 | 1.358386039733887 | 1.3580901622772217 | 0.00029587745666503906 | 0.00016790450899861753 |
| (0,1442) | -0.8704090118408203 | -0.8704090118408203 | -0.8706088066101074 | 0.00019979476928710938 | 0.000143530429340899 |
| (0,2806) | -6.8851213455200195 | -6.8851213455200195 | -6.884546279907227 | 0.0005750656127929688 | 0.0004442273057065904 |
| (1,3069) | -1.5374298095703125 | -1.5374298095703125 | -1.5370930433273315 | 0.00033676624298095703 | 0.00017685464990790933 |
| (2,1119) | 6.065071105957031 | 6.065071105957031 | 6.065493583679199 | 0.00042247772216796875 | 0.00040327466558665037 |
| (3,272) | 0.9020500183105469 | 0.9020500183105469 | 0.9023032784461975 | 0.00025326013565063477 | 0.0001451151620130986 |
| (3,2962) | -2.0208206176757812 | -2.0208206176757812 | -2.0211102962493896 | 0.00028967857360839844 | 0.00020105551811866462 |

All eight exceed the unchanged tolerance. In each isolated child run, pre-cast FP32 output SHA-256 was `4a0800c14d8c49750c1582e99c546908ce9284c8ef01decf3dd23d7252a61ff8`; oracle FP32 SHA-256 `da7ee5fa596912cc9ec2ed61f4015b9b46427cd29acc3dd61fc42e49a0f47602`; explicit stress-input FP16 SHA-256 `100faa2bc448c9bbbd17e68206ade236ce65dd0394fc0a953639a1c55d6439f1`. Both FP16-cast outputs have SHA-256 `877e8f0afdae643c1aafef9b6b8376d0eeea93063fb9c4c6d24d5e57ea9d074a`. Equal FP16 hashes do not satisfy the pre-cast FP32 oracle.

Authoritative unique-pair forensic record: `/mnt/data/pc01/sglang-v100-migration/experiments/E035/packed-word-reuse-stress-forensic.json`; run ID `stress-forensic-final`. It contains both arms, exit statuses, input/oracle/output hashes, exact binary identities and all coordinates. Candidate was rerun after fixing the test to make strict stress failure fatal: exit status 1 with `candidate violates frozen 4x stress oracle at 8 outputs`; baseline exits 0 while recording its eight strict failures. The detailed paired run is not used as a pass claim.

## Root-cause boundary and historical acceptance

The experiment-local candidate preserves each K value's ascending position in the same 128-K partition and performs the same FP32 FMA expression, group/zero/scale lookup and ordered split reducer. The packed words now feed those original K positions; no reduction reordering or precision change was intended. For this case the exact candidate and c57 pre-cast output tensors are byte-identical, which shows the attempted load reuse does not repair or introduce a difference in the observed failing values. The independent existing `diagnose_l43_down_strict.py` run on unscaled 1x fixture reports zero strict-bad (max abs 0.00286865234375, relative L2 2.1539774763823516e-07); this bounds the counterexample to 4x stress rather than ordinary fixture. No new algorithm within the authorized packed-word/small-M scope was allowed to alter the frozen arithmetic/reduction contract, so do not carry on to service measurement.

This new stress case narrows, and does not retroactively erase, the historical c57 acceptance evidence: c57's prior strict checks cover the documented real-activation fixtures and prior tested stress factors, not this added L43/down 4x FP32 strict oracle. Historical token/service evidence remains valid only for those archived scopes; it must not be described as proof that c57 passes this newly explicit 4x counterexample.

## Earlier performance evidence (not a qualification waiver)

The preserved initial 72-pair operator matrix (six projection fixtures × M1–4 × three alternating pairs) at `/mnt/data/pc01/sglang-v100-migration/experiments/E035/packed-word-reuse-final.64b2737a-427a-4e84-91c2-e548baf1e3e2.jsonl`, SHA-256 `970ab5b29423457d41650899ecb28fb3c8ba3d15a1391ea6a9a2ce0c2571c2fc`, used candidate SHA above and c57 SHA above. It passed ordinary fixture strict/Graph checks and FP16 identity in 72/72 pairs, but its harness merely logged the L43/down 4x `strict_bad=8`; it did not fail the run. That is why its initial “matrix complete” label was insufficient. Per-projection Graph medians showed down gains near 5–6%, while gate-up often regressed; aggregate results were only around 0.09–1.82% across M. These timings are exploratory and cannot justify runtime integration after the strict stress failure.

The original exact G017 c57 implementation source checkout no longer exists; the binary is exact by SHA, while build logs map it to the historical G017 close source/build. Do not claim the current checkout is the binary's source provenance. Historical service output was not tested with the rejected source or binary.

## Code disposition and risks

- Production tracked file `sgl-kernel/csrc/gemm/gptq/gptq_candidate_v8.cu` has been restored to HEAD bytes; no rejected candidate code enters default build or production dispatch.
- Source snapshot, build binary, isolated probe and test harness remain inside the E035 experiment area. No opt-in service route was added.
- Harness fix in `test_packed_word_reuse.py` now records every failed stress coordinate and actual/reference/error/limit, input/oracle/output hashes, and causes candidate stress strict failures to return nonzero. It does not remove stress or relax tolerance.
- Packed-word reuse lowers source-level packed Q4 loads from four loads per K iteration to four per eight-K block; this is a load-instruction opportunity, not a claim of 8x HBM traffic reduction. Candidate compiler resource details must be interpreted from its exact cubin/SASS; no equivalent NCU or service claim is made. Keeping the 128-K split and reducer means partial buffer and reducer traffic are unchanged; register pressure/load scheduling may change. Full-model performance, state isolation, Prefill, multi-step Decode, Graph dispatch integration, service TPS/TTFT/ITL/E2E, VRAM and capacity were not run because the candidate fails the required numerical gate.
- E035 WMMA/group-parallel remains Owner REJECTED and untouched.
