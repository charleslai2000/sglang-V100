# T011 full-model capacity/lifecycle ledger

## Checkpoint
- `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4` two safetensors shards total **7,808,627,634 bytes = 7.27 GiB on disk**. It is a mixed checkpoint: GPTQ packed INT32 qweight + qzeros, FP16 scales, INT32 g_idx; other dense weights are BF16 on disk and load as FP16 on SM70.
- Exact-c57 SGLang service (2026-10-18, target UUID only, `mem_fraction_static=.88`, FP16, max-running 4, max-total 16384, Graph keys 1/2/4, radix off) startup log `results/t011-model-c57-server.log`:
  - load begin available 15.38 GB; load end resident **8.09 GB**, free 7.29 GB.
  - Mamba state allocated **0.64 GB**.
  - KV: 16,384 tokens, K 0.34 GB + V 0.34 GB = **0.68 GB**.
  - memory pool end free **5.78 GB**.
  - Graph 1/2/4 capture succeeded, logged graph memory 0.08 GB and available 5.13 GB after capture. Actual candidate path is exact-c57 and service logs dispatch c57 SHA.
- Source model directory size is not a GPU peak; safetensors are memory-mapped and converted/loaded. These observed startup numbers are a c57 baseline, not a QPN2 service memory measurement.

## T011 full-model opt-in measured runtime (2026-10-18)

SGLang production `DefaultModelLoader.load_weights_and_postprocess` ran with isolated `SGLANG_G022_QPN2_BINDING=1`, target V100 only, exact extension SHA `4e2848fb0ca990ff873d696100e2f2fbd0f9f7afb2ffeb4d4879451b12cdbbfb`, and `E035_RUNTIME_MODE=fp64` to suppress sitecustomize's inherited c57 candidate opt-in. It packed all **88 gate_up + 44 down** modules, logged no fallback, and dispatched all 88 at Graph warmup M4. CUDA Graph 1/2/3/4 capture succeeded. Loader elapsed 152.46s; end weight memory 10.42GB; after KV/Mamba allocation free 3.74GB; Graph capture left 3.08GB. Requested 16,384 max tokens was automatically reduced to **10,864 tokens** by SGLang's memory budget; Mamba=.64GB, K/V=.23+.23GB. A real 64-token prompt/64-token greedy decode request completed (E2E 1.244s). This proves loader and launch-route viability, not output equivalence or state isolation. Full report: `results/t011-model-runtime.json`; full process log retained outside repo at `/tmp/t011-model-live15.log`.

The conversion adds both original and packed Q4 to the resident model. This empirical ~2.33GB current-run added budget plus loader staging/high water is consistent with, but not exactly equal to, the earlier byte estimate; staging/allocator effects are included in `mem usage`. The optimized process fitted at 10,864 max tokens, not 16,384. Do not compare capacity against c57 while changing concurrency/token capacity; the paired A/B therefore uses the common 10,864 budget. At common settings exact-c57 free after pool was 6.06GB; T011 free after pool was 3.74GB, after Graph 3.08GB. T011 successfully served c1/c2/c4 at 1K and 4K with 256 output tokens, though 4K/c4 E2E time showed a slower candidate median despite higher aggregate throughput and needs repeat confirmation.

## T011 candidate memory requirement
- QPN prepack adds one nibble-per-weight packed fragment tensor plus group permutation. For six representative L43 gate_up/down linears, each QPN layout roughly equals source Q4 qweight storage; it does **not** justify assuming entire model duplicate is affordable. Across all GPTQ projection qweights an approximate extra packed copy is about half the GPTQ packed-weight footprint; exact model-wide bytes must be enumerated at load time.
- Loading a second complete transformed copy while original checkpoint weights remain resident could exceed current ~5.78 GB post-pool free after temporary allocator/KV/Mamba/Graph; no claim that it fits. Per-step packing is forbidden.
- Required integration before production-like qualification: opt-in-only load-time pack per GPTQ layer; preserve original weights for Prefill (and unsupported shapes); maintain packed weights at stable model lifetime for Decode, no re-pack on decode; if prefill uses original then decode packing must be correct for same module and graph capture. Must benchmark peak at loader overlap (original + packed + staging), after cleanup, then with KV/Mamba/Graph and c1/c2/c4 workloads. If over 16 GiB or forces >3% other-path regression/no headroom, NO-GO.
- Current blocker is implementation/measurement, not arithmetic: no T011 dispatch is wired into `GPTQLinearMethod` / Falcon module lifecycle. No 44-layer service candidate exists yet. Do not mislabel exact-c57 server as T011 model evidence.

## Service probe attempts
- Isolated exact-c57 SGLang server was launched with `run_t011_model_server.sh`, loaded weights and captured Graph1/2/4, but first failed FPM IPC directory bind; directory was fixed and rerun. The healthy server was later gracefully stopped after the exploration. It never ran a T011 route.
- `qualify_t011_model.py` direct probes are exploratory only; direct model forward lacks scheduler-owned ForwardBatch/Mamba state and is not an accepted whole-model correctness test. Preserve failed logs in `/mnt/data/sglang-V100/t011-model/`; they are harness failures, not model numerical counterexamples.

## Verdict
Capacity compatibility and Prefill fallback remain **UNVERIFIED**. No c1/c2/c4 candidate service A/B was run, and no production-default change or Owner ACCEPTED designation has occurred.
