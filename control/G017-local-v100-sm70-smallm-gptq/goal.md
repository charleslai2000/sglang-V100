# Goal: G017 local V100 SM70 small-M GPTQ candidate

## Authorized outcome

Qualify a correct, CUDA-Graph-compatible SM70 GPTQ Q4/group128 kernel for the pinned Falcon-H1 GPTQ-v1 checkpoint on local Tesla V100-SXM2-16GB and demonstrate repeatable service improvement. The G017-CLOSE stage authorizes only clean source isolation, final build/qualification and an independent production checkpoint. No new kernel algorithm is authorized in this close stage.

## Frozen constraints

- Device UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; never run G017 work on RTX 2060 (`GPU-04ad2dd9-245d-0e7b-c96a-12da83a92b99`).
- Pinned model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- Preserve GPTQ-v1 Q4/g128/desc_act/original g_idx, zero+1, FP32 accumulator oracle with unchanged tolerance, and final FP16 output.
- Preserve Attention, Mamba, TileLang backend, Graph batch sizes 1/2/4, S1 fallback/default. Candidate requires explicit production enable configuration and fail-closed behavior.
- Do not reopen full-K fused kernel, shuffled FP16 route or Nsys profiling.
- Do not alter or discard the dirty original worktree. All checkpoint work occurs in isolated clean worktree from base `66abb118847ebf74c698820da003350a352d0ca2`.

## Completion

From the clean worktree: final complete extension build with source/toolchain/binary/load-path identities; final binary passes FP32 strict oracle 6/6, FP16/high-amplitude, actual dispatch, Graph1/2/4, 1K/4K c1/c2/c4 token parity, bounded state-isolation contract, >=3 paired service A/B rounds with TPS/ITL, VRAM and fallback checks. Commit only G017-scoped source/tests/reports/control record, push a dedicated remote ref, verify remote SHA and clean worktree. Mark ACCEPTED only after all conditions hold. Global default change is not implied.

## Result

The final clean-worktree binary `570f22db57dab3ff4b6185325a1ac7e205f4220509eb7a359894dba4ba7f4188` passed FP64-derived FP32 strict oracle 6/6, FP16 stress, actual dispatcher and Graph1/2/4, greedy identity 18/18, bounded seeded state isolation on candidate/S1, and 3-repeat clean service A/B on six cells with gains across c1/c2/c4 and 1K/4K. Full evidence is in `experiments/E026-local-v100-s4-m3c/results.md`. G017-CLOSE remains ACTIVE, not ACCEPTED: final integration/packaging review and independent scoped commit/push/remote SHA check remain. S1 stays default, candidate explicit opt-in; timing residual UNKNOWN. Original dirty source worktree is preserved.
