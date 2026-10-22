# T011 Prefill first-error closure and FP32 cuBLAS candidate

Status: candidate remains experimental and opt-in only. This records one controlled fixed-input comparison, not final qualification.

## Controlled first-error comparison

- Target GPU UUID: `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; host MemAvailable before runs: 31.1 GB.
- Checkpoint: `/data/t011/hf/Falcon-H1-7B-Instruct-GPTQ-Int4` (same directory/revision for both fresh processes); frozen QPN2 Decode binary SHA256 `6f3ee5732ca571149c9209e403c1d03330a2a151a56541ca40b38c2f849af90f`.
- Input IDs and positions are both exact `0..511`; each arm ran in a fresh Python process with a newly loaded model and newly initialized worker/request/Mamba/KV state. Independent random seeds are reset before construction. QPN2 source/binary, config, parameters and expected model are the same.
- Native run log: `/data/t011/artifacts/commonops128/t011-prefill-first-error.log`; candidate run log: `/data/t011/artifacts/commonops128/t011-prefill-first-error-candidate.log`. Script: `/data/t011/artifacts/t011-prefill-first-error.py`.
- Both arms: every observed layer mixer in/out, MLP gate-up and down projection boundary through L42 finite. Native first nonfinite producer is `model.layers.43.mlp.down_proj`, output FP16 `[512,3072]`, first Inf `[229,175]`. Candidate has no nonfinite observed boundary; final hidden and logits finite.
- The L43 activation was independently captured immediately before `act_fn` (actual down-projection input). It is FP16 `[512,12288]`, finite in each run. Native activation SHA256 `62ffb16875f6ef7d9a5f51d0e49b7c4f504ab1fd46d0729f389fde6a7a04368d`; candidate activation SHA256 `1b63e53a36b89b8088bcdf4e3ec3771281a602ef34c45251d818888f4a55ff8f`. These differ because the two preceding executions already have different hidden states from numerical changes; both coordinates used for their own references are finite. Do not treat their references as same-activation A/B comparisons.
- Within each arm the independent FP64 dot used that exact run's captured activation and the exact model's qweight/qzeros/scales/g_idx. Native `[229,175]`: raw FP64 `77708.75079141936`; scaled FP64 `2529.5817314915157`; direct FP32 raw `77708.75`; scaled FP32 `2529.58154296875`; final FP16 `2530`. Native raw FP16 output overflows, matching observed Inf. Candidate `[229,175]`: raw FP64 `77837.93990025949`; scaled FP64 `2533.7871061282385`; direct FP32 raw `77837.9375`; scaled FP32 `2533.786865234375`; final FP16 `2534`.
- The old reported FP64 values `77871.5990610119` and `77708.75079141936` are not contradictory: the first comes from the separately initialized/previously hooked activation (max about 5196, artifact `/data/t011/artifacts/commonops128/t011-fp32-helper-oracle.log`), the second from a different fixed-512-token run's native activation (max about 5176; artifact `t011-fp64-prefill-forensics.log`). The present controlled run reproduced the latter input hash, and directly showed candidate-arm hidden divergence. Do not repeat this investigation absent changed inputs or code.
- The earlier E044 experiment reported a pointwise dot of `77708.75079141936` even while its full activation contained NaN at `[229,175]`: the dot uses a finite FP64 activation coordinate from the different controlled capture, not the NaN-containing activation. It was not evidence that NaN had been multiplied. Current diagnostic independently checks finiteness before producing a dot oracle.

## Current implementation

`prefill_gptq_gemm.cu` now reconstructs Q4 `(q - (qzero+1))*scale` directly to FP32, converts FP16 activation to FP32, uses FP32 input/output and `CUBLAS_COMPUTE_32F` `cublasGemmEx`, multiplies the output in FP32 in Falcon's existing down multiplier location, then casts to the layer parameter dtype. It does not modify global cuBLAS policy, checkpoint data, or Decode QPN2. Weight scratch is invocation-local and released after the call; persistent per-layer FP32 weights are not kept.

## Full real down-layer operator oracle (not model acceptance)

A separate fresh worker ran fixed 512-token Prefill and compared each actual down projection to an independent FP64 GPTQ weight reconstruction using that layer's exact finite captured activation, then applied the unchanged multiplier in FP32. Oracle tolerance was the established strict fixture form `1e-4 + 5e-5*abs(reference)`; strict-bad was zero for all 44 down layers, each 1,572,864 outputs, with all FP32 raw/scaled and final FP16 outputs finite. Across layers max tolerance ratio ranged up to `0.9079942` (L43); the 44-layer run's overall final hidden remained nonfinite because this oracle script deliberately did not replace the production down operator. It did not run Graph or measured timing. This indicates the FP32 reconstruct/GEMM arithmetic passes its local strict output oracle for the recorded finite inputs, not that the full model is qualified.

- Oracle log `/data/t011/artifacts/commonops128/t011-prefill-operator-oracle.log`, SHA256 `5494385ba73db5e198665cbfdeccec8f5f14e91eb7ac8fad005aed15df057707`; script SHA256 `fff63e49eb7679d61e94fbf96e6c231d965bdbfd89c7fca9a737f6742bd3f0d4`.
- Native log SHA256 `d0145faa37af280e903342e3ca5204772bd7f65a2c109ece67f04363fc636ccf`; candidate first-error log SHA256 `a4d26ecf0bdcd398081992a8c78dbc4b8ffde34f978c12ee2af36d2f8910ee72`.
- Candidate shared extension SHA256 for both controlled candidate and oracle processes: `faa2cf9c7cc0230c2e13e7aad0223e53f01e7ad3d77984a7e75294209e7c011e`; source SHA256 `1a09fd0fe727ed14f216ecc3af5dfabea00b3744a3b1cd4a4fcfd59f32287558`.

Full corrected model integration, same-input hidden/logit comparison under identical arithmetic, Graph replay, kernel-count collection, and service qualification remain undone. A warmed (one shape warmup plus two repeat warmups, five CUDA-event trials per shape) single-worker candidate diagnostic run on the target V100 measured:

| Prefill tokens | median GPU event ms | token/s | peak PyTorch allocated | peak reserved |
|---:|---:|---:|---:|---:|
| 512 | 265.312 | 1,929.8 | 9,998,886,912 B | 10,915,676,160 B |
| 1,024 | 478.797 | 2,138.7 | 10,080,884,736 B | 10,915,676,160 B |
| 4,096 | 1,881.118 | 2,177.4 | 10,609,440,768 B | 10,915,676,160 B |

All three trial outputs had finite hidden/logits. Log: `/data/t011/artifacts/commonops128/t011-prefill-fp32-gemm-benchmark.log`, SHA256 `2f979b4ac7890644061bcb116ef7a8878670d5ef7fb61015091bc9d85a57c6ad`. These are candidate-only measurements without matched native baseline, no profiler kernel count, and don't establish the 3% threshold. They follow model initialization and unmeasured shape warmups; no first-use timings are included. This is a forward diagnostic, not full service Prefill TPS.

## Alternating native/FP32 Prefill A/B (in progress; no decision from partial matrix)

A fresh process per arm, fixed checkpoint/config/input tokens and deterministic state seed, same 512/1K/4K shapes, three unmeasured warmups and seven uninstrumented trials per process. Order per shape: native, FP32, FP32, native. The 512 FP32 arms also collect one profiler trace; those profiled runs are excluded from timing, and the sequence is native, FP32-profile, FP32-profile, native. Native is performance-only because its hidden output is nonfinite. The runner measures GPU event time, host submission and synchronization wait, PyTorch allocator peaks; controller separately samples target-UUID NVML memory/utilization every 250 ms.

Completed 512-token arms so far:

| Arm | median GPU ms | TPS | host submit ms | sync wait ms | peak allocated | peak reserved | hidden finite |
|---|---:|---:|---:|---:|---:|---:|---|
| Native pass 1 | 184.135 | 2780.6 | 181.770 | 2.493 | 9,954,846,720 B | 10,890,510,336 B | no |
| FP32 pass 1 | 270.948 | 1889.7 | 255.939 | 15.091 | 9,998,886,912 B | 10,915,676,160 B | yes |
| FP32 pass 2 | 271.055 | 1888.9 | 260.642 | 10.457 | 9,998,886,912 B | 10,915,676,160 B | yes |
| Native pass 2 | 182.992 | 2797.9 | 180.455 | 2.552 | 9,954,846,720 B | 10,890,510,336 B | no |

These controlled order-paired measurements indicate ~47.6–48.1% full-Prefill GPU-event slowdown for FP32 versus native at 512 tokens. Existing profiler breakdown (instrumented, not performance-timing evidence) shows per 44 down calls approx FP32 Q4 reconstruction kernel 151.3 ms total, activation conversion/cast kernels small relative, and FP32 `volta_sgemm_128x128_nn` 1,122.4 ms aggregate across 440 calls (profiler sum may include concurrency/serialization and must not be equated to event interval). Native's FP16 weight reconstruction sums 121.2 ms and half GEMM 328.9 ms across 440 calls. This identifies GEMM as dominant incremental work, with FP32 weight reconstruction also material. Double FP16 residual Tensor Core experiment is therefore authorized by the owner instruction, but must pass the complete strict FP64 oracle before any qualification. No uncontrolled additional candidates will be tried.

Profiler trace: `/data/t011/artifacts/commonops128/t011-prefill-profiler-fp32-512.json` (single instrumented 512 forward, 18,484 traced kernel events across setup/warmups/measured iterations/diagnostics; trace includes overhead and is not a per-forward kernel count). Full CUDA Event/host log: `/data/t011/artifacts/commonops128/t011-prefill-ab-alternating.log`; NVML samples: `/data/t011/artifacts/commonops128/t011-prefill-nvml-samples.jsonl`. The controller measures only 512 shape at this point; 1K and 4K are still in progress.

Current call allocates 144 MiB FP32 weights per projection (K=12,288,N=3,072), plus activation and output; allocator peak delta is about 41.9 MiB for whole run relative to native at 512, but PyTorch `max_memory_allocated` may not show overlapping scratch lifetime accurately across calls. Scratch reuse could reduce allocator churn, but does not eliminate Q4 reconstruction, FP32 writes or GEMM. NVML peak sampling has not been summarized yet.

The FP32 candidate remains experimental/default-off. Do not interpret native's faster but nonfinite output as correctness, and do not enter full T011 service qualification until an approved arithmetic path passes numerical and Prefill cost gates.
