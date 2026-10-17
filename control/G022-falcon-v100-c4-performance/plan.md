# G022 plan

Status: **ACTIVE — T010 DONE, PERFORMANCE-NO-GO for the tested QPN2 single-CTA SM70 organization.** Prior T004–T009 outcomes remain closed. T010 was a distinct, explicitly authorized donor-dataflow investigation; correctness passed, but three alternating L43/down complete-Linear pairs were ~2.94–3.35× slower than exact-c57 at M1/M2/M4, decisively missing the ≥5% whole-Forward potential gate. No NCU/service/production dispatch was run. Preserve G023 ON_HOLD, S1/default/exact-c57 rollback and protected E035 file.

T009 resolved source/binary provenance (G017 `5fac74206b55464dca5993d943120746a24b959a`, source SHA `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`; E026 build log binds frozen binary SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`). Corrected E040/E042: 0x0560–0x06f0 are scale half loads, actual Q4 loads are four scalar 32-bit loads 0x04a0–0x04d0. E043 candidate combines them into one `LDG.E.128`, outputs bit-match c57 on six L43 cells, but all six alternating complete-Linear Graph cells are slower (1.0147–1.0570×); M4 fails ≥12% reduction and M1/M2 do not improve. Matched gate_up M4 NCU: global load instructions -21.4%, registers 56→48, but DRAM read +0.3%, L2 read sectors +1.0%, long scoreboard +6.45pp, instrumented kernel +8.1%. No repair/model/service stage. Errata/results: E040/E042 reports and `experiments/E043-g022-c57-q4-vector-loads/`. No new GPTQ design authorized.

## T010 result — PERFORMANCE-NO-GO

- Owner directed continuous T010 execution; donor pinned to `dnv2003/v100-skinny` commit `5b589c0dc81223e0ba65bcb3e755874723f8b515`. Source/prepack/kernel audit, exact mapping, implementation and result are recorded in `experiments/E044-g022-qpn2-sm70-actorder/experiment.md`.
- Numerical matrix passes all six fixtures ×M1/M2/M4 plus L43/down M4×4 (19/19 strict FP64/Graph/repeat/cast); SASS confirms HMMA, REG56, no spill/local/shared. Three alternating L43/down complete-Linear pairs show QPN Graph 483–496µs vs exact-c57 143–170µs across M1/M2/M4, ~2.94–3.35× slower; Eager likewise ~3× slower. Prepack/setup 1.11–1.16s per prototype invocation.
- Decisively fails ≥5% whole-Forward potential; NCU, broader projection performance and model/service stages correctly not run. No production route or Owner ACCEPTED status. Preserve G023 ON_HOLD, S1/default/exact-c57 and protected E035 file.

## Established constraints and evidence

- G022 goal and frozen authority remain in `goal.md`; ordinary T002 work must preserve S1 default, exact-c57 rollback, strict tolerances, target V100 only, host MemAvailable >=8GiB, and distinguish complete Linear/Forward/service timing.
- T004–T009 outcomes are implementation/question-specific NO-GO and must not be reopened without a fresh causal premise. E042 SourceCounters showed recurring sampled consumer PCs, but no unique producer chain. T009's tested uint4 Q4 load candidate reduced SASS load instruction count but regressed all six complete-Linear Graph cells; no repair/model/service stage followed.
- User's numerical, Forward/service, Prefill and resource gates are authoritative; no Owner ACCEPTED designation.
