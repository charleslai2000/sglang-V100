# Goal: S3 Falcon-H1 V100 vs RTX 3090 performance forensics and optimization

## Authorized outcome

Using frozen Falcon serving correctness baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e` and the pinned GPTQ checkpoint, optimize the V100 SGLang Falcon-H1 path only. Use the established RTX 3090 figures as a fixed historical performance reference; do not rebuild, port, run, profile, or requalify RTX 3090 Falcon. Quantify V100 prefill/decode owners with low-overhead Nsight Systems and targeted Nsight Compute or equivalent evidence, deliver at least one measured high-value V100 software optimization without changing model semantics (or evidence-supported hardware/new-kernel blocker), correct S2 Mamba memory accounting, and update the 16GB capacity estimate.

## Scope and constraints

- Active measurements and implementation changes are V100-only. Treat RTX 3090 reference TPS supplied by the user as fixed external historical context only; no additional RTX 3090 interaction/build/run/profile/requalification.
- Keep prefill and steady-state decode separate. Use repeated/long-enough workloads and correctness checks; profiler captures are attribution evidence only, never throughput values.
- Nsight Systems first, then targeted Nsight Compute/CUPTI/DCGM on identified kernel families. No whole-program high-overhead profiling.
- Optimize only a profiler-proven high-value software limiter, preferring mature SGLang/upstream donors and minimal changes. Measure before and after, revert if no repeatable gain.
- Do not change model semantics or frozen baseline. Preserve unrelated worktrees and shared-device ownership; coordinate GPU usage and leave devices idle after tests.
- Distinguish physical measurements from theoretical estimates. 16GB V100 remains unmeasured unless available.

## Completion condition

Clean V100-only Nsight Systems attribution for steady graph decode and eager prefill, targeted counters on actual dominant owners, separate prefill/decode before/after matrices, at least one material repeatable profiler-backed optimization or a demonstrated unavoidable V100/new-kernel blocker, quantitative remaining gap against fixed 3090 reference, corrected Mamba memory formula and 16GB throughput estimate, realistic V100 ceiling, and at most three ranked next optimizations. Stop only for the explicitly stated real hardware/architecture blockers.

## Result

In progress.
