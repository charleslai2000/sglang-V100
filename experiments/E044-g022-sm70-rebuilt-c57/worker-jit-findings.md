# SM70 JIT and first real REBUILT-C57 dispatch

## RoPE compile failure root cause

The failed JIT was the FP16 Falcon RoPE specialization `sgl_kernel_jit_fused_rope_true_128_false_fp16_t`, generated at `/home/charles/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_128_false_fp16_t_78319dbc84cca7ee/cuda.cu` (two exported functions include the repo `python/sglang/jit_kernel/csrc/elementwise/rope.cuh`). Its `sgl_kernel/type.cuh` instantiated BF16x2 conversion functions for a template which only executes FP16 operations. CUDA Toolkit 12.8's `/data/t011/toolkit/cuda-12.8.1/include/cuda_bf16.h` declares `__bfloat1622float2` and `__float22bfloat162_rn` behind `__CUDA_ARCH__ >= 800`; nvcc targets `compute_70,sm_70`, so they are deliberately unavailable. This was not a target architecture selection error and not a mixed header version: system CUDA 12.0 `/usr/include/cuda_bf16.h` SHA `59d8e4eac5aa5f148f75a006334dedfd6c6b687d656755c73be4be26b75ef45b`; CUDA 12.8 toolkit and PyTorch CUDA runtime copies have identical SHA `4d5a2ad88adb17983aef0505ed6ed2a0603497c79c103bba82c928301ea12310`. The actual first failed attempt used `/usr/bin/nvcc` CUDA 12.0, shown by its command, despite the later corrected run using `/data/t011/toolkit/cuda-12.8.1/bin/nvcc`.

`python/sglang/jit_kernel/include/sgl_kernel/type.cuh` now has SM<80 conversion helpers made from two scalar BF16 conversions and compile-time selects these instead of unavailable packed intrinsic spellings. This is valid on SM70 (CUDA BF16 scalar conversion exists from SM70) and does not fabricate intrinsics or change generated hardware target. The full FP16 RoPE JIT then passes compilation and the actual model forward traverses the RoPE section. The helpers are also used by any future BF16x2 conversion in the same SM70 JIT header, so run broad JIT and numerical tests before considering merge/production qualification.

## Separate FlashInfer sampling `math.h` error

The earlier sampling JIT log is a distinct failure: `/usr/bin/nvcc` included `/usr/include/cuda_runtime.h`, then `/usr/include/crt/math_functions.h`, which reached `/usr/include/c++/12/cmath`; `#include_next <math.h>` failed. The CUDA command lacked an explicit `/usr/include` host include, and configured `CUDAHOSTCXX` was not consumed by that FlashInfer helper (command uses `/usr/bin/nvcc`, no `-ccbin`). CUDA 12.8 toolkit is under `/data/t011/toolkit/cuda-12.8.1`, while `/usr/local/cuda-12.8/include` was empty. This host-header/toolkit selection failure is separate from BF16 intrinsic isolation. The successful Falcon identity run explicitly used `sampling_backend='pytorch'`, so FlashInfer sampling JIT is not claimed fixed or validated.

## Real c57 dispatch evidence

A fresh process loaded full Falcon worker with explicit REBUILT-C57 path/SHA, then `_dummy_run(batch_size=1)` reached `GPTQLinearKernel.apply` on `model.layers.0.mlp.gate_up_proj` with Prefill M=512. It failed closed because the rebuilt c57 native op only supports M<=4. The log contains `REBUILT-C57 dispatch active ... M=512 K=3072 N=24576` followed by the CUDA op's `candidate supports aligned M in [1,4] geometries`, proving the registered native c57 kernel was actually invoked (and rejected unsupported M), not merely loaded. The SGLang startup code catches this error as a Prefill warmup warning and continues; the later diagnostic `_dummy_run` emitted the same native rejection but the harness did not propagate that exception and printed `FORWARD_OK None`. That print is not a successful Prefill result. Fix the harness/warmup propagation and route Prefill only through intended common GPTQ fallback before calling the c57 Decode path.

- Full log: `/data/t011/artifacts/commonops128/rebuilt-c57-worker-identity.log`, SHA256 `0158ac86a0fb460b411318d56fa480b65153e2ba824efd1106dcdfdd0bdcc5b0`.
- Harness: `/data/t011/artifacts/rebuilt-c57-worker-identity.py`, SHA256 `e9e9263cfa1335742612cb120a335ae33bb68fa56d3918f45ef8af2c35d003fa` (uses pytorch sampling; sampling not measured).
- REBUILT-C57 binary SHA remains `ab5db27bade8cb868faed6ee4b72c9f7047f838b0ac23c4cdfa4c2b895944d95`; original source SHA `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`.
- Peak observations from this partial worker: allocated `9,753,574,912` bytes and reserved `10,714,349,568` bytes at worker-loaded checkpoint, before verified Forward. These are not capacity peaks.
- No successful M1/M2/M4 Decode, numerical oracle, Graph, state, or service result. T011 remains ACTIVE.

## Uncommitted status

JIT `type.cuh` source fix is validated only by the compiled RoPE path observed in the failed M512 model attempt (full forward failed/reported warning due unsupported Prefill route) and has not yet been separately tested against a numerical RoPE oracle. Dispatch changes in `gptq_kernels.py` remain exploratory and unqualified. Their patch is `/data/t011/artifacts/rebuilt-c57-dispatch-exploratory.patch`, SHA256 `2cf308612ba431b56e02731a2ba2a7432851e78216802c840aa40aa84a28a0a6`.
