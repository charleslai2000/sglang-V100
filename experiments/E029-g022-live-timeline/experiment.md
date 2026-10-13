# E029 — Bounded Falcon GPU timeline restoration

## Question
Can accepted SGLang `/start_profile`/`/stop_profile`, the G018 forward identity/DeviceTimer design, and the pinned G017 Candidate runtime produce bounded, per-forward Falcon CPU/GPU activity timelines at real c1/c2/c4 workloads without a new profiler collector?

## Identity and scope
- Checkout `/home/charles/Workspaces/3rdparty/sglang-V100`, start HEAD `333af65242529f37b18170a154a7c2ce48b1fdf6`.
- Falcon-H1-7B-Instruct-GPTQ-Int4, local pinned model revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189` only.
- Exact G017 Candidate extension SHA-256 `02f923ea7329912a4413f179f292acb444a5e001caebf110b24f531cabf70d8b`; Candidate remained opt-in; no default/dispatch change.

## Procedure
1. Compared current source with accepted G018 commit `631cfc89c2644d398a4555fcfbcb509919275bef`; restore only `forward_iter`, request/batch sidecar, CUDA Event host brackets, and exact Graph key propagation.
2. Use SGLang native Kineto and existing `/start_profile`/`/stop_profile`, warm one request outside the capture, then issue a bounded real Falcon workload at c1/c2/c4 and 1K/4K.
3. Add an opt-in `profile_range=prefill|decode` request field using the existing legacy CUDA-profiler API, to begin/stop after target Forward completion. This avoids the normal legacy stage trigger leaking adjacent stage work into a trace.
4. Analyze raw trace offline. Join annotation `iter=N`, request IDs and Graph key using sidecar. Pair Kineto CPU record_function clock-marker span with `time.monotonic_ns` bracket and use robust unit-slope offset. Never subtract independent CUDA Event timer or another clock domain.
5. NCU is separate from trace/service capture. Attempt only a finite existing root NCU route if supported attach syntax can be established without changing runtime ownership.

## Measures
Per forward: mode/annotation/request IDs, actual scheduled batch/tokens, graph_bs/key, event elapsed, step annotation duration, kernel and memcpy activity intervals, activity union, intra-forward holes, inter-forward GPU activity envelope gaps, CPU scheduler boundaries in aligned Kineto time, operator family kernel-duration sum and clock alignment residual. Event, annotation, activity union and service response time remain distinct.
