# G019 — CUDA Graph Decode Execution Attribution

## Authorized outcome

On the local V100, obtain actual Falcon-H1 CUDA Graph decode execution activities and construct a traceable owner ledger with source-backed identities, counts, kernel durations and overlap-aware timing. Primary matrix: S1 and exact-hash G017 candidate, 1K/4K × c1/c4. Do not develop/modify kernels or alter model/config/Graph semantics.

## Constraints

- Base: G018 closed PARTIAL at `df975fc3a209bca90536d19861c3ee7404eca45d`; G017 accepted, S1 default, candidate opt-in only.
- V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189` only; RTX 2060 not profiled or disturbed.
- No global profiling/driver settings changes, no CUDA/PyTorch upgrade, no full-service Nsys.
- Use existing SGLang worker-side profiler first; if inadequate, only existing CUDA 12.8 CUPTI Graph tracing implementation. If both fail to expose valid node timing, mark PARTIAL and stop—no third profiler framework.
- Profiling service metrics are not throughput authority. No kernel/default change.

## Completion

Report Graph IDs/nodes where supported; kernel identity/source, launch geometry, invocation count, GPU duration; GPTQ gate_up/down, Attention, Mamba and other owners; copies/memsets and idle/gap timing; account overlaps and whole-forward boundary; tracing coverage/overhead; worker allocated/reserved/peak and Graph pool/free memory when directly obtainable; target hardware telemetry; matched NCU counters for top graph owners if mapping succeeds; artifact manifest; scoped commit/push and remote SHA verification. Unavailable values remain UNKNOWN.

## Initial result

G019 clean worktree `/home/charles/Workspaces/3rdparty/sglang-V100-G019` is based on G018 commit above. Fork worker-side `/start_profile` + `SchedulerProfilerManager` was reused. Eight small profiler captures (S1/c57 × 1K/4K × c1/c4, three decode forward steps requested per capture) ran on the target V100; exact artifacts are listed in `experiments/G019/artifact-manifest.json`. Trace captures include hundreds/thousands of GPU kernels per window and names/geometry/durations, but the worker Profiler export omits `graphId`/`graphNodeId`, so it cannot provide the required Graph-node ledger by itself. A vendor CUDA 12.8 CUPTI `cuda_graphs_trace` sample was built in the private worktree and executed successfully on V100, proving node IDs and kernel activities are available in a standalone controlled graph. It does not attach to an existing SGLang worker graph.

Current decisive task: reuse the CUPTI sample's activity/callback mechanism with an existing worker profiler window only if bounded process-local attach is supported without changing graph semantics; otherwise document why graph exec/node mapping cannot be recovered from current worker profiler outputs and stop under the two-route limit. Memory worker snapshot and top-owner NCU only if actual owner mapping is obtainable. Full detailed state in T001 and final report.
