# G022 plan

Status: **ACTIVE — T010 DONE / PERFORMANCE-NO-GO only for its single-warp serial CTA; T011 ACTIVE as explicitly owner-directed group-aligned multiwarp QPN2.** Prior T004–T009 remain closed. T011 reuses the pinned donor's SPLITK group partition and bounded shared FP32 reduction, adapted to GPTQ group128 independent scales. Six-fixture M1–M4 + L43/down M4×4 strict matrix passes; first alternating M4 L43 gate_up/down graph comparisons show ~64.8%/~52.8% lower component latency. Not yet a final performance acceptance: NCU is blocked by system `ERR_NVGPUCTRPERM`, alternative split/complete coverage and E044 timing-boundary errata remain open. No model/service/production dispatch. Preserve G023 ON_HOLD, S1/default/exact-c57 rollback and protected E035 file.

T009 resolved source/binary provenance (G017 `5fac74206b55464dca5993d943120746a24b959a`, source SHA `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`; E026 build log binds frozen binary SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`). Corrected E040/E042: 0x0560–0x06f0 are scale half loads, actual Q4 loads are four scalar 32-bit loads 0x04a0–0x04d0. E043 candidate combines them into one `LDG.E.128`, outputs bit-match c57 on six L43 cells, but all six alternating complete-Linear Graph cells are slower (1.0147–1.0570×); M4 fails ≥12% reduction and M1/M2 do not improve. Matched gate_up M4 NCU: global load instructions -21.4%, registers 56→48, but DRAM read +0.3%, L2 read sectors +1.0%, long scoreboard +6.45pp, instrumented kernel +8.1%. No repair/model/service stage. Errata/results: E040/E042 reports and `experiments/E043-g022-c57-q4-vector-loads/`. No new GPTQ design authorized.

## T011 — ACTIVE multiwarp investigation

- Owner clarified T010's PERFORMANCE-NO-GO applies only to serial single-warp execution and directly authorized T011 SPLITK 8/16/32 multiwarp work; this does not reopen E035.
- `experiments/E044-g022-qpn2-sm70-actorder/experiment-t011.md` records donor SPLITK behavior, actual group128-owned implementation, exact CUDA build, correctness and preliminary A/B.
- Six original fixture×M1/2/3/4 and L43/down M4×4 (51 rows) pass strict oracle, Graph FP32 scratch, repeat, independent scratch, and Eager/Graph cast consistency with zero strict-bad. SM70 binary uses HMMA; registers 47/thread, shared 8/16/32 KiB for SPLITK8/16/32, no local/spills. Preliminary three alternating M4 L43 component pairs show gate_up 89.088 vs 252.928 µs and down 73.728 vs 155.648–156.672 µs.
- NCU access is concretely blocked by host `ERR_NVGPUCTRPERM` (both visible GPUs; no bypass/policy change attempted). Need finalize the full paired split coverage and E044 timing accounting errata, plus disposition hardware-counter failure before closing T011. No model/service/opt-in until gate and authorized evidence complete; no Owner ACCEPTED.

## T010 result — PERFORMANCE-NO-GO

- Owner directed continuous T010 execution; donor pinned to `dnv2003/v100-skinny` commit `5b589c0dc81223e0ba65bcb3e755874723f8b515`. Source/prepack/kernel audit, exact mapping, implementation and result are recorded in `experiments/E044-g022-qpn2-sm70-actorder/experiment.md`.
- Numerical matrix passes all six fixtures ×M1/M2/M4 plus L43/down M4×4 (19/19 strict FP64/Graph/repeat/cast); SASS confirms HMMA, REG56, no spill/local/shared. Three alternating L43/down complete-Linear pairs show QPN Graph 483–496µs vs exact-c57 143–170µs across M1/M2/M4, ~2.94–3.35× slower; Eager likewise ~3× slower. Prepack/setup 1.11–1.16s per prototype invocation.
- Decisively fails ≥5% whole-Forward potential; NCU, broader projection performance and model/service stages correctly not run. No production route or Owner ACCEPTED status. Preserve G023 ON_HOLD, S1/default/exact-c57 and protected E035 file.

## Established constraints and evidence

- G022 goal and frozen authority remain in `goal.md`; ordinary T002 work must preserve S1 default, exact-c57 rollback, strict tolerances, target V100 only, host MemAvailable >=8GiB, and distinguish complete Linear/Forward/service timing.
- T004–T009 outcomes are implementation/question-specific NO-GO and must not be reopened without a fresh causal premise. E042 SourceCounters showed recurring sampled consumer PCs, but no unique producer chain. T009's tested uint4 Q4 load candidate reduced SASS load instruction count but regressed all six complete-Linear Graph cells; no repair/model/service stage followed.
- User's numerical, Forward/service, Prefill and resource gates are authoritative; no Owner ACCEPTED designation.
