# SM70 JIT and first real REBUILT-C57 dispatch

## RoPE compile failure root cause

The failed JIT was the FP16 Falcon RoPE specialization `sgl_kernel_jit_fused_rope_true_128_false_fp16_t`, generated at `/home/charles/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_128_false_fp16_t_78319dbc84cca7ee/cuda.cu` (two exported functions include the repo `python/sglang/jit_kernel/csrc/elementwise/rope.cuh`). Its `sgl_kernel/type.cuh` instantiated BF16x2 conversion functions for a template which only executes FP16 operations. CUDA Toolkit 12.8's `/data/t011/toolkit/cuda-12.8.1/include/cuda_bf16.h` declares `__bfloat1622float2` and `__float22bfloat162_rn` behind `__CUDA_ARCH__ >= 800`; nvcc targets `compute_70,sm_70`, so they are deliberately unavailable. This was not a target architecture selection error and not a mixed header version: system CUDA 12.0 `/usr/include/cuda_bf16.h` SHA `59d8e4eac5aa5f148f75a006334dedfd6c6b687d656755c73be4be26b75ef45b`; CUDA 12.8 toolkit and PyTorch CUDA runtime copies have identical SHA `4d5a2ad88adb17983aef0505ed6ed2a0603497c79c103bba82c928301ea12310`. The actual first failed attempt used `/usr/bin/nvcc` CUDA 12.0, shown by its command, despite the later corrected run using `/data/t011/toolkit/cuda-12.8.1/bin/nvcc`.

`python/sglang/jit_kernel/include/sgl_kernel/type.cuh` now has SM<80 conversion helpers made from two scalar BF16 conversions and compile-time selects these instead of unavailable packed intrinsic spellings. This is valid on SM70 (CUDA BF16 scalar conversion exists from SM70) and does not fabricate intrinsics or change generated hardware target. The full FP16 RoPE JIT then passes compilation and the actual model forward traverses the RoPE section. The helpers are also used by any future BF16x2 conversion in the same SM70 JIT header, so run broad JIT and numerical tests before considering merge/production qualification.

## Separate FlashInfer sampling `math.h` error

The earlier sampling JIT log is a distinct failure: `/usr/bin/nvcc` included `/usr/include/cuda_runtime.h`, then `/usr/include/crt/math_functions.h`, which reached `/usr/include/c++/12/cmath`; `#include_next <math.h>` failed. The CUDA command lacked an explicit `/usr/include` host include, and configured `CUDAHOSTCXX` was not consumed by that FlashInfer helper (command uses `/usr/bin/nvcc`, no `-ccbin`). CUDA 12.8 toolkit is under `/data/t011/toolkit/cuda-12.8.1`, while `/usr/local/cuda-12.8/include` was empty. This host-header/toolkit selection failure is separate from BF16 intrinsic isolation. The successful Falcon identity run explicitly used `sampling_backend='pytorch'`, so FlashInfer sampling JIT is not claimed fixed or validated.

## Real c57 dispatch evidence

A full Falcon worker process loaded REBUILT-C57 in-process, checked exact schema and CUDA registration, executed a real 512-token Prefill through the existing native GPTQ fallback, and then ran Falcon model Decode forwards at M1/M2/M4. The candidate is selected only for M<=4; M=512/1024 requests fail closed at `GPTQLinearKernel.apply`, so Prefill cannot silently route to the candidate. The log has a successful dispatch record on `model.layers.0.mlp.gate_up_proj` with M=1, K=3072, N=24576 and the exact library path/SHA. Subsequent M2/M4 model forwards returned finite full `(batch,130049)` logits. The original common_ops fallback is unchanged for prefill, and the test harness explicitly restores it for M>4. These are real c57 calls and model-forward smoke, but not independent FP32 numerical correctness or Graph capture/replay.

- Full latest worker log: `/data/t011/artifacts/commonops128/rebuilt-c57-worker-identity.log`, SHA256 `dd539ba1141b3a58b02797cf0ce4ae8b39305fcd75b50c16f9d9206475087e8d`.
- Harness: `/data/t011/artifacts/rebuilt-c57-worker-identity.py`, SHA256 `4827ac99c6277ae90376efdac82fe8fb2d5d9d4f7dca985682a8157b1d73ad8d` (pytorch sampling; no sampling-service claim).
- REBUILT-C57 binary SHA: `ab5db27bade8cb868faed6ee4b72c9f7047f838b0ac23c4cdfa4c2b895944d95`; original CUDA source SHA `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`.
- Prefill logits shape `(1,130049)`, hidden shape `(512,3072)`; Decode M1/M2/M4 logits shapes respectively `(1,130049)`, `(2,130049)`, `(4,130049)`, all hidden tensors finite. This is not an oracle comparison.
- Allocator snapshot after worker creation: allocated `9,753,187,840`, reserved `10,714,349,568`; after forward `9,757,878,272`, reserved `10,714,349,568`. Not a load/Graph/runtime peak measurement or capacity proof.
- No Graph capture/replay, state lifecycle, independent numerical oracle, or service result yet. T011 remains ACTIVE.

## Isolated RoPE numerical test

The exact previously failing RoPE specialization was rebuilt by `_jit_fused_rope_module(True, 128, torch.float16)` on the designated V100. A 16-token Q/K test compared output against explicit FP32 NeoX rotation followed by FP16 cast: max error 0, differing elements 0, output SHA256 `e44eb34a4866aa178c27c92342f3f6dfd946c8f626b5a3bbbaa3aca356e0e7ed`. Test log SHA256 `61f57122fd682bd5136b4dab34229ea469dfd7def2b6970fb6df2cbeffe268a2`; harness `/data/t011/artifacts/test_rope_sm70.py`. This qualifies only this RoPE specialization/test shape.

## Uncommitted status

The committed JIT fix and RoPE numerical evidence are in commit `38dc99deaa8c00b18d33afb054a7413d9d3bb0cf` (later commits advance main). The REBUILT-C57 dispatcher changes in `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py` remain exploratory and uncommitted; patch path `/data/t011/artifacts/rebuilt-c57-dispatch-exploratory.patch`, SHA256 `2cf308612ba431b56e02731a2ba2a7432851e78216802c840aa40aa84a28a0a6`. It has now been exercised in worker for M1/M2/M4 but not independent-numerically qualified. The WNA16 compressed-tensors one-line import edit remains unverified and unrelated to RoPE/c57 decode; preserve and do not include without separate review.
