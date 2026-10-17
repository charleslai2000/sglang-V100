# G022 plan

Status: **ACTIVE — T009 c57 Q4 vector-load milestone underway.** T001 hardware utilization PASS; T003 diagnostics/proposal DONE; T004/E038, T005/E039, T006/E040, T007/E041 closed NO-GO; T008/E042 bounded ARCHITECTURE-NO-GO. E041 audit corrections retained without rerun. Historical RTX3090 15.88ms identity remains UNVERIFIED. Preserve G023 ON_HOLD, S1/default and exact-c57 rollback, and protected unowned E035 file.

T009 first priority is source provenance: exact-c57 binary SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb` must be mapped to its source and build recipe before proposing a source edit. E040 SASS hash `c4570fbbabf1208d9a95f228d2eb57712389846bce4d019d23d71d6e3fe5e827` shows four separate 16-bit Q4 LDG sites; the relative alignment and actual coalesced sector traffic still need verification. No A/B, candidate implementation, or numerical qualification has yet run under T009. See `control/G022-falcon-v100-c4-performance/tasks/T009-c57-q4-vector-loads.md` and `experiments/E043-g022-c57-q4-vector-loads/`.

## Established constraints and evidence

- G022 goal and frozen authority remain in `goal.md`; ordinary T002 work must preserve S1 default, exact-c57 rollback, strict tolerances, target V100 only, host MemAvailable >=8GiB, and distinguish complete Linear/Forward/service timing.
- T004–T008 outcomes are implementation/question-specific NO-GO and must not be reopened without a fresh causal premise. E042 SourceCounters showed recurring sampled consumer PCs, but no unique producer chain; T009 is separately authorized to test the Q4 load-instruction hypothesis.
- T009 gate: >=12% weighted M4 complete-Linear net reduction with genuine alternating pairs and no significant M1/M2 regression. If missed, stop without numerical repair/model/service work. Only if passed may bounded FP32 repair and full qualification proceed.
- User's numerical, Forward/service, Prefill and resource gates are authoritative; no Owner ACCEPTED designation.
