# G022 plan

Status: **ACTIVE — T010 PERFORMANCE-NO-GO only for serial single-warp CTA; T011 operator, real 132-layer loader/Graph route, and matched 1K/4K c1/c2/c4 service output-TPS matrix have passed their measured stages. Final GPU-forward attribution, per-request latency/Prefill metrics, logits/hidden-state and multi-request state qualification remain open.** Preserve G023 ON_HOLD, S1/default/exact-c57 and protected E035 file.

## T011 — operator stage complete; model/service frontier active

Owner clarified that T010's PERFORMANCE-NO-GO applies only to the serial single-warp CTA and authorized T011. T011 implements donor-informed SPLITK 8/16/32, whole GPTQ group128 ownership, independent scale, FP32 group sums and bounded deterministic shared FP32 reduction. This is distinct from both T010 and rejected E035 global G×M×N scheme.

- Alias-free correctness: all six fixture×M1/M2/M3/M4 and valid split cases plus L43/down M4×4 stress (51 cells) strict oracle/Graph scratch bad=0; repeat, independent-scratch and Eager/Graph cast all pass. Corrected validation source/artifacts and hashes in `experiments/E044-g022-qpn2-sm70-actorder/experiment-t011.md` and `results/SHA256SUMS-T011`.
- Identity-bound complete-Linear alternating A/B across all six fixtures, M1/M2/M4, Eager/Graph, down S8/S16/S32: all have 3 alternating pairs per fixture-cell. Gate_up S8 gains 62.5–64.6%; down S16 is best of tested 8/16/32 with 48.5–53.1% gain. Final L43 M4 gate_up+down Graph sums ~163.5µs candidate vs ~409.9µs exact c57 (~60.1% lower). Native-FP16 c57 schema and equivalent output-boundary timing corrected; old E044 data boundary-mismatched and retained historically.
- SASS confirms HMMA.884; NCU succeeded under existing E040 sudo Application Replay with 4GiB/no-swap scope, no host policy changes. For measured QPN CTA: 47 registers/thread; shared 8/16/32KiB; local/stack=0; one barrier. Representative M1/M4 gate_up/down issue, eligible warp, stalls, DRAM/L2 and exact fixed geometry recorded in T011 report/index; Tensor active percent and whole-forward counters UNVERIFIED.
- Real loader lifecycle and full-layer dispatch: SGLang opt-in packs 88 gate_up + 44 down Q4 modules, fallback=0. All 88 dispatch at M4 Graph warmup; Graph1/2/3/4 capture passed. Isolated buffer ownership retains original GPTQ layout and gather applies stable `argsort(g_idx)`; service generation succeeds. Binary identities and load logs are indexed in `results/SHA256SUMS-T011`.
- Matched capacity/service: T011 finalized at 10,864 tokens (requested 16,384 was reduced); resident weight memory 10.42GB, Mamba=.64GB, KV=.46GB, post-pool free=3.74GB, post-Graph free=3.08GB. At same 10,864-token config, three-wave 1K/4K c1/c2/c4 exact-c57/T011 service tests have identical 256-token outputs in all cells; median Output TPS improves +25.3% to +39.5%. Full results/hash in `results/t011-service-matched-summary.json`. Existing GPU FPM event sidecar matched 3,592 Decode forwards per arm: batch distribution b1=1032/b2=2048/b4=512, with median full-forward GPU event b4 31.609ms c57 vs 22.275ms T011 (−29.53%); b3 absent. Prefill 1K/2K events were near parity (about +0.83%/−0.49%), 4K Prefill not observed. Sidecar graph_key null and per-cell timing join absent. 4K/c4 request E2E initially worsened (8.24s vs 16.47s candidate for 2 requests), requiring repeat. These are significant measured wins but latency/state isolation and repeated 4K/c4 E2E must close before readiness.
- Decisive remaining frontier: joined FPM/device timing with actual admitted batch and Graph dispatch, layerwise hidden/logit comparison (first layer/token on mismatch), multi-request Mamba/KV state+slot reuse, Prefill fallback and per-request latency/Prefill metrics. Full 16K token request capacity did not fit this configuration; no capacity NO-GO is declared because matched 10,864-token operation succeeds. No default route changed or Owner ACCEPTED.

## T010 result — PERFORMANCE-NO-GO

T010 tested serial one-warp/CTA QPN2 only. Six-fixture×M1/M2/M4 + stress correctness passed; three alternating L43/down complete-Linear pairs were ~2.94–3.35× slower than c57. No NCU/model/service was run for that serial design. T011 does not reopen or modify T010 artifacts.

## Established constraints and previous results

- G022 frozen authorities and constraints in `goal.md`: S1/default, exact-c57 rollback, strict tolerance, target V100 only and host MemAvailable≥8GiB; no RTX2060; keep service/Forward/kernel time distinct.
- T004–T009 remain closed; do not reopen without fresh causal premise. No Owner ACCEPTED designation.
