Status: DONE
Work area: research (archived local evidence only)
Objective: Reconstruct conservative kernel-category activity distribution from G018/G019 Falcon decode traces offline and publish reproducible artifact/report.
Inputs: G019 trace archive and source checkpoint; G018 FPM/service references; G017 accepted uninstrumented service artifacts.
Completion: CPU-only analyzer, trace hashes, per-step owner call/duration/union/exclusive/overlap/coverage/UNKNOWN results, careful service/FPM comparison, PARTIAL limits, independent commit/push and verified remote SHA.
Result: Implemented `experiments/G020/analyze_traces.py`, `analysis.json`, `report.md`, and `artifact-manifest.json`. Nine archived V100 traces were reanalyzed. GPU decode annotation spans provide temporal per-step windows; Graph node IDs and exact whole-forward owner shares remain unavailable. Four 4K/c4 traces annotate actual bs3. Unknown symbols are material (~7.7–7.8 ms) in c1; other classified owners remain conservative. Rerun matches byte-for-byte and script compiles. Scope remained offline; no GPU test/profiler ran.
Remaining: None. Isolated branch was committed/pushed as `572bcd751991fd72d7b12f5b1ee3e446861d377a`; SHA verified. At user direction, analysis deliverables and control records were copied into the existing sglang-V100 checkout.
