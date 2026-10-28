# G002-N 真实输入数值与 kernel A/B

## 判定

**NO-GO：不接入生产，也不推进服务 c1/c4。** WMMA 的真实输入 numerical check 通过；kernel-only CUDA Event 有明显优势，但尚未完成实际模型 Forward A/B。不能将离线七层 attention 时间差直接当作 Forward 净收益。production dispatch 保持不变。

## 生产 Triton 入口与数值

G002-M 的 NVTX/CUDA correlation 证实 D512 Full 层调用 `TorchNativeAttnBackend.forward_extend -> extend_attention_fwd -> _fwd_kernel`。V100 D512 配置从 `_get_block_sizes_for_extend_attention(512,512)` 得到 `BLOCK_DMODEL=512, BLOCK_DPE=0, BLOCK_DV=512, BLOCK_M=16, BLOCK_N=16, num_warps=8, num_stages=1`。本实验通过真实 `extend_attention_fwd` 入口离线重放，没有使用会申请 256 KiB shared memory 的 context kernel。

对捕获 Q、owner K/V、scale=1、全序列 causal mask 同输入三方比较，输出误差格式为 `(max_abs, MAE, p99_abs)`：

| 长度 | 层 | 生产 owner | WMMA vs FP32 | Triton vs FP32 | WMMA vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1K | 5 | 5 | (0.002475, 1.045e-4, 5.25e-4) | (0.002475, 1.044e-4, 5.22e-4) | (0.003906, 7.93e-6, 2.44e-4) |
| 1K | 11 | 11 | (0.002594, 1.023e-4, 7.14e-4) | (0.002594, 1.022e-4, 7.14e-4) | (0.003906, 6.62e-6, 2.44e-4) |
| 1K | 17 | 17 | (0.002237, 9.64e-5, 5.13e-4) | (0.002282, 9.63e-5, 5.10e-4) | (0.003906, 6.00e-6, 2.44e-4) |
| 1K | 23 | 23 | (0.002398, 9.35e-5, 4.96e-4) | (0.002398, 9.34e-5, 4.94e-4) | (0.003906, 5.33e-6, 1.22e-4) |
| 1K | 29 | 23 | (0.002522, 9.22e-5, 5.34e-4) | (0.002522, 9.21e-5, 5.32e-4) | (0.003906, 4.73e-6, 1.22e-4) |
| 1K | 35 | 23 | (0.002572, 9.25e-5, 5.10e-4) | (0.002572, 9.25e-5, 5.08e-4) | (0.003906, 4.52e-6, 1.22e-4) |
| 1K | 41 | 23 | (0.002398, 8.80e-5, 4.95e-4) | (0.002398, 8.79e-5, 4.93e-4) | (0.003906, 4.52e-6, 1.22e-4) |
| 4K | 5 | 5 | (0.002475, 1.067e-4, 5.50e-4) | (0.002475, 1.058e-4, 5.22e-4) | (0.003906, 1.51e-5, 4.88e-4) |
| 4K | 11 | 11 | (0.002594, 1.068e-4, 7.51e-4) | (0.002594, 1.058e-4, 7.51e-4) | (0.003906, 1.61e-5, 4.88e-4) |
| 4K | 17 | 17 | (0.002486, 1.020e-4, 5.77e-4) | (0.002486, 1.011e-4, 5.62e-4) | (0.003906, 1.51e-5, 4.88e-4) |
| 4K | 23 | 23 | (0.002398, 9.75e-5, 5.43e-4) | (0.002398, 9.68e-5, 5.14e-4) | (0.003906, 1.26e-5, 4.88e-4) |
| 4K | 29 | 23 | (0.002450, 9.95e-5, 6.50e-4) | (0.002450, 9.88e-5, 6.50e-4) | (0.003906, 1.27e-5, 4.88e-4) |
| 4K | 35 | 23 | (0.002400, 9.69e-5, 5.69e-4) | (0.002400, 9.62e-5, 5.57e-4) | (0.003906, 1.25e-5, 4.88e-4) |
| 4K | 41 | 23 | (0.002574, 9.32e-5, 5.30e-4) | (0.002574, 9.25e-5, 4.99e-4) | (0.003906, 1.23e-5, 4.88e-4) |

FP32 reference uses identical Q/K/V, repeats KV heads 4x for GQA, scale 1, and `key_position <= query_position`. Shared layers 29/35/41 use owner cache layer 23; capture metadata confirms production `layer_id=23`, `save_kv_cache=False`. WMMA and Triton each consume the same captured owner tensors.

## CUDA Event kernel A/B（非 Forward）

Alternating/interleaved per-layer measurements, medians; seven layer sum:

| 长度 | Triton seven-layer sum | WMMA seven-layer sum | WMMA preparation sum | 离线 net delta (Triton-WMMA-prep) |
|---:|---:|---:|---:|---:|
| 1K | 46.423 ms | 12.842 ms | 1.027 ms | +32.554 ms |
| 4K | 627.790 ms | 146.507 ms | 1.011 ms | +480.272 ms |

WMMA preparation measured Q layout conversion, K/V page view/materialization, block table and sequence metadata construction. This does not include real SGLang cache pool writes, CPU launch/setup overhead or model-forward side effects. These are kernel/replay measurements only.

## 有效性边界

Capture 是真实 E4B 1K/4K Forward 中、Q/K/V norm 与 RoPE 之后的 attention pre-hook 张量。诊断 Forward 使用同 cos/sin cache、positions、NeoX 公式的 PyTorch-native RoPE，非生产 CUDA RoPE kernel。单序列 cache slot 连续；Full-to-SWA mapping 手工恒等映射。验证覆盖单序列全量 prefill，不覆盖多序列、分页碎片、线上 allocator/Graph 或 scheduler。不得推广为服务或真实 Forward 收益。

`type.cuh` 未提交任何改动并保持原样。没有生产 dispatch 改动。
