# Plan: S4-M3 low-concurrency c4 residual and GPTQ qualification

## Established inputs

- Prior c96 decode ledger is diagnostic only; no c96 owner shares are transferred to c4.
- M2 Attention optimization is closed rejected: production stage1+stage2 0.724793ms vs pinned native-FP16 donor 5.828608ms full-op; do not reopen without a materially changed measured profile.
- Read-only starting evidence: E023 c4 graph ledger plus isolated disposable GPTQ ABI observer. Correctness-valid c4 has 88 alt-Q4 kernel calls/forward, 18.667655 ms/forward (51.655% of graph-kernel intervals). All exact layer/role/geometry mappings are now established: 44 gate_up M/N/K 4/24576/3072, 44 down 4/3072/12288; Q4 group128 desc_act=true, use_shuffle=false, layer IDs 0–43. Raw observer `/hy-tmp/sglang-logs/s4-m3-gptq-abi.jsonl`; E023 results record details. Prior E022 and c96 evidence remain diagnostic only.
- Model config GPTQ 4-bit/group128/desc_act=true; SM70 path retains non-identity `g_idx` semantics. Exact per-call route must be recovered, not inferred from source count.

## Decisive frontier

The c4 ledger proves GPTQ is a major owner and exact MLP mapping is complete. M3A provenance and results are in E024. S1's explicit F6B packed-FP32 elementwise criterion rejects both the S1 and shuffled FP16 route on the six M=4 cases, so it is not a transferable pass criterion. S1 serving acceptance requires repeat-greedy IDs/nonzero outputs but defines no cross-route operator tolerance. Original high-amplitude activation bytes are not retained; exact historic replay unavailable. Controlled real L43/down stress, with both routes using current output initialization, gives shuffled-only FP16 inf at max|x|=6048 while unshuffled remains finite, reproducing the same known failure class; do not integrate. Isolated performance estimate is 0.30925→0.08909ms gate_up, 0.20480→0.06042ms down; not service evidence. The existing shuffled path is rejected as correctness-ineligible. External donor continuation is not authorized within this update. No token A/B/profile occurred. V100 idle and disposable sources restored.

## Conclusion

T001 complete. Existing S4-M3A shuffled GPTQ path verdict: REJECTED for production under frozen S1 constraints. Evidence and limitations are in `experiments/E024-s4-m3a-shuffled-gptq/results.md`. Do not continue to service A/B or refresh the graph ledger; no candidate was integrated. Further alternative-donor investigation is a separate continuation.

## G016 local authority handoff (2026-10-09)

Local V100-SXM2-16GB Falcon serving/state-isolation/graph1-2-4 and bounded short stability checks pass under the revised G016 acceptance; targeted root NCU GPTQ-alt kernel report is validated on local SM70. Local source identity commit is `36b35cb68fe493c202f7d6c64100de29738b2684`; model/dependency and evidence identities are recorded in `control/G016-local-v100-16g-falcon-gptq`. Falcon Nsys server CUDA trace is explicitly `DEFERRED / NOT QUALIFIED`, not a pass or gate. Resume authorized S4-M3C with the existing c4 ledger and targeted NCU; do not repeat rejected shuffled route.
