# Plan: G017-CLOSE clean checkpoint

## Established

- Authoritative base: `66abb118847ebf74c698820da003350a352d0ca2`; remote `origin/milestone/s1-falcon-h1` points to `2eeead7e0ce274e8162a2c4f4e1b64d39c6a1b21`, and contains the base plus downstream commits. The candidate checkpoint must branch from the authorized base and be pushed to a new ref, not rewrite the existing branch.
- Original worktree `/home/charles/Workspaces/3rdparty/sglang-V100` remains dirty and untouched. Clean isolated worktree: `/home/charles/Workspaces/3rdparty/sglang-V100-G017-close`, branch `g017-close-prod-checkpoint`, HEAD at base.
- Prior final isolated candidate SHA `02f923ea…`; strict FP32 accumulator oracle 6/6 and final-SHA token parity 18/18 PASS. Prior exact-hash service A/B: c4 +6.3–6.7%, c1/c2 positive. Same-set NCU and FPM evidence is in the E026 result record.
- S1 remains default. Candidate is explicit opt-in. No Attention/Mamba changes or new kernel optimization in close phase.

## Decisive frontier

Reproduce final binary from clean worktree, perform final-binary qualifications, then make and push a dedicated scoped checkpoint. Final binary must not be confused with prior binary.

## Task

T002 `tasks/T002-clean-checkpoint-and-final-qualification.md` owns clean build, final binary checks, scoped commit/push, and acceptance handoff.

## Residual timing rule

Existing native FPM c4 forward comparison (34.124→31.719ms, 1,023 events/runtime) and service comparison are aggregate, not request-ID joined. Operator A/B saves ~0.156ms per forward across gate_up+down; named NCU kernel-only delta is ~0.061ms per forward. Do not force an owner allocation for the difference. Report unresolved residual as UNKNOWN. Reuse existing targeted NCU evidence unless clean-build SASS/geometry differs.

## Current state

A clean V100-only extension build exists: wheel SHA `c868b2a245354bdf0ca5d9402942a9c6fe5039450cdb0dd95b15e8b9a6b02a98`; extracted candidate SHA `570f22db57dab3ff4b6185325a1ac7e205f4220509eb7a359894dba4ba7f4188`. Exact final-hash strict oracle 6/6, FP16 stress, Graph M1/2/4, actual service dispatch, token parity 18/18, seeded state isolation candidate/control, and clean service A/B 3 repeats across six cells all passed. See `experiments/E026-local-v100-s4-m3c/results.md`. Loader missing path/wrong SHA fail closed; explicit frozen-S1 SHA `fea05241…` loaded. S1 remains default. Original dirty worktree untouched.

G017 is ACCEPTED. Checkpoint `5fac74206b55464dca5993d943120746a24b959a` is pushed to `origin/g017-close-prod-checkpoint`; independently queried remote SHA matches. G017 worktree is clean. Final source manifest and c57 binary qualification are recorded in E026. S1 remains default, candidate opt-in, residual UNKNOWN.
