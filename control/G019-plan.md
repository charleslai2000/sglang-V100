# G019 plan

## Established

- Parent authority: G018 PARTIAL closed at `df975fc3a209bca90536d19861c3ee7404eca45d`; G017 accepted base and S1/c57 SHA pins retained. G019 worktree/branch `g019-cuda-graph-attribution` is isolated at that commit.
- CUDA environment unchanged: PyTorch `2.9.1+cu128`, CUDA runtime 12.8, V100 UUID target only. V100 maps index 1; RTX2060 index 0. No driver/global profiling changes.
- Fork source contains worker-side `/start_profile`, `/stop_profile`, `profile_by_stage`, and bounded `num_steps`; actual collection starts/stops in scheduler worker. Captured only CPU+GPU activity, `with_stack=false`, shapes enabled, decode-only stage.
- CUDA 12.8 installed CUPTI official `cuda_graphs_trace` sample source demonstrates callbacks for `GRAPHNODE_CREATED`/`CLONED`, `cuptiGetGraphNodeId`, plus kernel/memcpy/memset activity fields `graphId`/`graphNodeId`, launch geometry, and start/end timestamps. Sample built privately for sm70 and ran on only the target V100, returning graphId 2, node IDs, kernel names/geometry/durations, and creation API correlation. It is a standalone sample, not the SGLang graph.
- Eight worker profiler DECODE traces, each with 3 decode forward steps requested, exist for S1 and c57 × 1K/4K × c1/c4; c57 exact binary SHA and dispatch were logged. PyTorch trace has named kernel events, geometry, duration and correlation for some events, but no `graphId` or `graphNodeId` field (checked every kernel event). Thus native worker route gives kernel activity totals but not graph node identity/association.
- `cuda_graphs_trace` sample implementation is single-process CUPTI callback/activity tracing. Attaching it to an independently running Python worker would require injection/in-process registration not provided by the worker endpoints. Integrating CUPTI callbacks into the serving process is a new profiler implementation and outside the finite two-route scope; no third framework will be built. Decide whether a direct existing PyTorch/CUPTI route can expose node IDs without new framework; if not, close G019 PARTIAL with evidence.

## Matrix raw traces

All under `/mnt/data/pc01/sglang-v100-migration/experiments/G019/`:
- `s1-1k-c1/*DECODE.trace.json.gz`, `s1-1k-c4/*DECODE.trace.json.gz`, `s1-4k-c1/*DECODE.trace.json.gz`, `s1-4k-c4/*DECODE.trace.json.gz`.
- `c57-1k-c1/*DECODE.trace.json.gz`, `c57-1k-c4/*DECODE.trace.json.gz`, `c57-4k-c1/*DECODE.trace.json.gz`, `c57-4k-c4/*DECODE.trace.json.gz`.
- Candidate 4K c4 has a second capture for trace repeatability.
- Each capture has corresponding EXTEND file; keep prefill and decode separate.
- Trace event inventory/timing calculations and limitations: `experiments/G019/report.md`.

## Final disposition and next action

Report and artifact manifest are `experiments/G019/report.md` and `experiments/G019/artifact-manifest.json`. Route A and B have been exercised; neither yields node-level SGLang graph attribution within the finite authorized implementations. Mark G019 PARTIAL. Do not claim worker allocator, top-owner NCU, Graph memory pool, complete owner shares or whole-model SM/Tensor/HBM without direct evidence. All G019 service workers are stopped and the V100 returned to idle. Validate the scoped report/assets, commit/push to a dedicated ref and verify remote SHA/clean tree; no third profiler infrastructure.
