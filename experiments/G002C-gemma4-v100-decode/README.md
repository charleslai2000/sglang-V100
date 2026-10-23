# G002-C: Gemma 4 V100 Decode Graph recovery

## Result
On the pinned Gemma 4 E4B W4A16-CT and target V100, the prior 126 ms-class Decode forward was caused by benchmark serving flags explicitly disabling both standard and piecewise CUDA Graphs. The existing standard SGLang `CudaGraphRunner` captures Decode batch sizes 1/2/4 and replays them when batch size is supported. No Marlin kernel or compute source was changed.

## Reproduction and evidence
- Complete A/B result, identities, GPU Event, scope limitations: `/data/t011/artifacts/g002c_decode_graph_ab_summary.md`.
- Three Eager rounds: `/data/t011/artifacts/g002c-ab-eager-r{0,1,2}.jsonl`.
- Three Graph rounds: `/data/t011/artifacts/g002c-ab-graph-r{0,1,2}.jsonl`.
- FPM: `/data/t011/artifacts/g002c-eager-fpm.jsonl`, `/data/t011/artifacts/g002c-graph-ab-fpm.jsonl`.
- Graph startup/replay log: `/data/t011/artifacts/g002c-graph-ab-server.log`.
- Graph 4096/256 c4 concurrency/slot-reuse/serial isolation run: `/data/t011/artifacts/g002c-isolation-graph.run.log` (12/12 success, matching output hashes).
- Original G002-A Eager qualification remains unchanged and separate.

## Acceptance boundary
The Graph configuration gives 3.3–3.5x lower measured GPU Decode Event than Eager and raises measured service Output TPS at all six points, with no observed Prefill GPU regression. The exact measured server parameter change is removing `--disable-cuda-graph --disable-piecewise-cuda-graph` from the prior launcher. This evidence demonstrates restored Decode Graph service behavior, not per-kernel owner fractions, hardware-counter utilization, or a new kernel algorithm. Graph remains an explicit serving configuration; default CLI semantics are unchanged.
