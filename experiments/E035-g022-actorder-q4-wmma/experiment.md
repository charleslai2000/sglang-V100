# E035 — Act-order GPTQ Q4 grouped SM70 WMMA prototype

## Question
Can an isolated, deterministic act-order GPTQ-v1 Q4 prepack plus grouped SM70 Tensor Core linear implementation satisfy the frozen pre-cast FP32 oracle on all archived Falcon L0/L22/L43 gate-up/down fixtures at M1/M2/M4, including high-amplitude L43/down, and reduce **complete** linear latency against the exact accepted G017 c57 path after activation gather and FP32 reduction are included?

## Hypothesis
Sorting K indices by original `g_idx` allows Q4 rows to be packed group-major. For each group, gathered FP16 activations and exactly representable FP16 integer `(q - (qzero+1))` operands feed Volta WMMA with FP32 accumulation; scales multiply group FP32 dots, and groups reduce deterministically in FP32 before the only output cast. Whether numerical order and conversion exactly meet tolerance, and whether the added gather/reduction cost is amortized, are unknown.

## Identity and scope
- Repository `main`, starting HEAD `621824f82e3d9ba29c7725135d5ae61ff02e706c`; no branch/worktree.
- Immutable fixtures under `/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures/{0,22,43}-mlp-{gate_up_proj,down_proj}-call0000`; no new checkpoint-wide scan.
- Exact G017 c57 baseline package SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`, loaded by its explicit immutable path; never overwrite.
- Target only V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; host MemAvailable must remain >=8 GiB. RTX2060 is prohibited and currently independently occupied.
- Isolated extension and temporary artifacts live under `/mnt/data/pc01/sglang-v100-migration/experiments/E035/`; source/test/reproduction entrypoints live here. No production dispatcher/source, S1 default, G017 route, installed extension, server, cache, or service changes.

## Inputs and controls
For each of six real fixture matrices, use archived exact x/qweight/qzeros/scales/g_idx bytes; M=1,2,4 are prefixes of the captured M4 input. Also use frozen high-amplitude L43/down fixture where available. Prepack is deterministic and retains original-group identity and exact qzero+1 semantics. Gate/up may share sorted permutation only after tensor equality is checked, but each retains separate packed Q/scales/zero metadata. No full FP16 dense weight is materialized or retained.

## Procedure
1. Implement CPU converter and GPU operator only in E035. Converter groups all K row indices by g_idx with deterministic stable order; tiles Q4 storage group-major, carries exact per-group index order, scales and GPTQ zeros, and rejects malformed counts/metadata. Test round-trip all codes and qzero+1, including non-8 synthetic zero and group/tile boundaries.
2. Compile to a separate named extension in the artifact directory, with explicit sm_70 and target-only `CUDA_VISIBLE_DEVICES`. Never install/copy it into any existing package.
3. Run correctness in stages: actual FP32 accumulator versus frozen FP64 GPTQ oracle at `abs(error)<=1e-4+5e-5*abs(ref)` (zero strict-bad), final FP16 cast, deterministic repeat, independent scratch and Graph1/2/4 replay. Diagnostics must be actual accumulator returned by the new operator.
4. If correctness passes, compare complete end-to-end linear latency against exact c57 on identical archived x/weights and actual M1/M2/M4, with at least three alternating matched pairs. Include converter/prepack only as load-time (not each invocation); invocation cost includes gather, grouped Tensor Core work, reduction and output cast. Freeze cache/Graph/geometry/inputs and record all raw pair data.
5. Reuse existing NCU capability only for one representative exact geometry after correctness and resource preflight. Verify Tensor instructions, CTA/grid, full operator stages, kernel durations, memory traffic and stalls. NCU results are kernel scope; don't translate kernel sums to full Forward.
6. Recalculate per-layer conversion peak (not all-model dual-format sum), steady storage, only real SM70 call-chain scratch, and conditional Forward headroom from existing measured budget. Complete this phase with GO-FORWARD (candidate only for separately authorized whole-Forward integration) or NO-GO-CANDIDATE and a unique next recommendation. Never claim production/Forward/service improvement here.

## Measurements
- Exact source/binary/artifact SHA, build flags, GPU UUID, host memory pre/post.
- Per fixture, M, repeated, high-amplitude and graph cases: max errors, strict-bad, cast check, deterministic/scratch isolation, graph identity/correctness.
- M1/M2/M4 complete-linear alternating c57 pairs, stage timings, CTA/grid, NCU relevant metrics if safe.
- Static/peak/steady/scratch bytes and eligible fraction of historical GPTQ kernel-duration budget, explicitly conditional and non-exclusive.

## Stop conditions
Stop only on a genuine architecture/ownership boundary, missing mandatory external component, or diagnostic evidence establishing a hard numerical/geometry/resource blocker. Build/interface/numerical implementation failures are in scope for repair and rerun. No GPU run if target V100 is not idle or host MemAvailable is below 8 GiB.
