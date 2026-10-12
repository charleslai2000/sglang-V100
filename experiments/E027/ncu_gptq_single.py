#!/usr/bin/env python3
"""Bounded S1/c57 real GPTQ operator fixture used by targeted NCU.

This exercises one SGLang GPTQ Q4/G128 operator at explicit M=1,K=N=4096,
not Falcon service shapes or a full model forward. Run under root NCU with
CUDA_VISIBLE_DEVICES pinned to the V100 UUID. Output validates dispatch only.
"""
import torch
from sgl_kernel.gemm import gptq_gemm

m, n, k, group_size, bits = 1, 4096, 4096, 128, 4
x = torch.randn((m, k), device="cuda", dtype=torch.float16)
w = torch.randint(0, 2**31 - 1, (k // 32 * bits, n), device="cuda", dtype=torch.int32).view(torch.uint32)
groups = k // group_size
zeros = torch.randint(0, 2**31 - 1, (groups, n // (32 // bits)), device="cuda", dtype=torch.int32).view(torch.uint32)
scales = torch.rand((groups, n), device="cuda", dtype=torch.float16) + 0.01
gidx = torch.zeros((k,), device="cuda", dtype=torch.int32)
out = gptq_gemm(x, w, zeros, scales, gidx, False, bits)
torch.cuda.synchronize()
assert out.shape == (m, n) and torch.isfinite(out).all()
print("G018 GPTQ fixture complete", tuple(out.shape), out.dtype)
