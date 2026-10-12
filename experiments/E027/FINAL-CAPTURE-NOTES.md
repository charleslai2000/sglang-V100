# G021 FPM production capture notes (2026-10-11)

## Bounded capture

- Exact S1 publisher endpoint: `ipc:///tmp/g021-fpm-s1.0`; subscriber READY written before service launch.
- Target V100 only: `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; S1 route, Candidate not loaded.
- Request plan: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/fpm-production-delivery-requests.json`, 12 waves, 30 user requests, 1K/4K × c1/c4, three waves per cell, every request returned 96 output tokens.
- Service log: `/mnt/data/pc01/sglang-v100-migration/logs/g021-fpm-delivery-window.log`.
- Per-Forward research sidecar: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/fpm-request-sidecar.jsonl`.
- Subscriber output/READY: `/mnt/data/pc01/sglang-v100-migration/experiments/E026/fpm-subscriber-delivery.json` and `.ready.json`.
- Subscriber script: `experiments/E027/fpm_reconcile_subscriber.py`.

## Reconciled counts

Scheduler Forward/timer/active emit/queue-accepted: 2357 each. Internal queue drop, duplicate/unmatched timer, timeout, pending batches and event intervals at shutdown: zero. Publisher sent: 2506, send-drop/fail: zero. Subscriber received: 2506 = 2357 active + 149 idle; sequence 0..2505 contiguous, zero gaps/errors. This is a single observed bounded local IPC PUB/SUB window; no durable delivery guarantee is inferred.

The sidecar contains 2357 unique forward_iter values (2355 user-workload rows and two health-probe rows), request IDs, original unpadded input token lengths, outputs-so-far, aggregate scheduler scheduled-prefill token count, graph key, CUDA Event elapsed, scheduler wall duration and queue disposition. v1 payload bytes/schema were unchanged and do not expose identity or CUDA Event time. Per-forward Event elapsed is retained separately from scheduler wall; interval sums do not yield a GPU active union.

User-workload GPU/Event diagnostics: 43 EXTEND forwards, scheduler-reported scheduled-prefill-token sum 75,021 and Event elapsed sum 21.682844 s = 3,459.92 aggregate tokens/s; 2,312 DECODE forwards, 2,888 output-token steps and Event elapsed sum 68.065327 s = 42.4298 token steps/s, Event latency p50/p95 28.794/33.646 ms. Decode graph selection: key 1 2,024 forwards, key 4 288 forwards with actual scheduled batch 3 padded to graph4. `scheduled_prefill_tokens` is an aggregate PrefillStats field: this run did not persist each request's `extend_input_len` or `cached_tokens_before_forward`; don't claim per-request exact scheduled extend tokens or cache-exclusive recompute.

## Shutdown handshake detail

TokenizerManager drains HTTP/tokenizer requests, sends private `GetInternalStateReq(action="fpm_shutdown")`, scheduler acknowledges, and the scheduler exits at a fully idle boundary so its existing `finally` cleanup writes timer/publisher final counters. v1 wire is untouched. The handshake/control code was present in the capture build and exists in the uncommitted source diff; the final source review removed an unrelated temporary signal-handler experiment. Relevant source edits should be isolated/reviewed before any commit.

## Validation and cleanliness

- FPM unit suite: 13 tests pass.
- `py_compile` of affected Python code and `git diff --check` passed prior to final removal of the temporary scheduler signal-handler experiment; rerun both before commit.
- After service cleanup, V100 reported 7 MiB, 0% utilization; MemAvailable 16,370,576 kB. The unrelated RTX2060 still had 1,893 MiB/3% and was not touched.
