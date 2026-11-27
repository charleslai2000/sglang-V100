# E045 — 结果

## OBSERVED

运行身份：Tesla V100-SXM2-16GB，PyTorch 2.9.1+cu128，FP16 权重，Triton attention，
禁用 CUDA graph 与 radix cache（prefix cache 复用单独一轮启用）。

### 逐长度一致性（本轮最终运行）

| prompt tokens | greedy IDs | 生成 logprob 最大差 | next-token top-5 |
| --- | --- | --- | --- |
| 12 | 完全一致 | 0.0114 | 顺序一致 |
| 200 | 完全一致 | 0.00035 | 顺序一致 |
| 300 | 完全一致 | 0.00056 | 顺序一致 |
| 1000 | 完全一致 | 2.6e-5 | 顺序一致 |
| 4096 | 完全一致 | 1.0e-5 | 顺序一致 |

### 长上下文 hidden（4096 token，prefill 采样位置）

修正 RoPE 前：位置 2048 cos 0.9722 / RMSE 0.409；位置 3840 cos 0.9565 / RMSE 0.539。
修正 RoPE 后：所有采样位置 cos ≥ 0.999994，RMSE ≤ 0.0064。

### 其它

- 128 步 greedy 解码：IDs 完全一致，logprob 最大差 3.6e-5。
- prompt 全量 logprob（1000 token）：MAE 6.3e-4，max 0.024。
- chunked prefill 1024/2048 与不分块：IDs 与 logprob 一致。
- prefix cache：重复请求命中 999 cached token，输出与冷启动及 Oracle 一致。
- Chat Completions API 正常返回；4 路并发同提示输出一致。
- 吞吐（521 token prompt，单请求）：16/64/128 输出 = 12.0/19.7/20.1 tok/s。
- 服务常驻显存 14,173 MiB；KV 池 399,436 token。

### 失败与偏差（保留）

- SGLang 端 4096 token 全量 prompt logprob 采集触发 16GB OOM（logits 全量 detach），
  改用 1000 token 全量 + 4K 末尾 top-5。
- `torch_native` attention 后端在本 Hybrid 模型的 warmup 阶段抛
  `get_cuda_graph_seq_len_fill_value` NotImplementedError，未能作为对照后端。
- 当前环境无 `pytest` / `ruff`，仅执行 `compileall`、`git diff --check` 与断言脚本。

## INFERRED

- 位置相关的 hidden 漂移由 RoPE 频率表截断导致：HF 用完整 head_dim 生成 `inv_freq`，
  cos/sin 保持 FP32；按 `partial_rotary_factor` 截断会使误差随位置累积。
- 短 prompt 的 0.011 logprob 差属 FP16 末位量化：官方 FP16 与 FP32 参考本身差
  ≤0.005，SGLang 与官方 FP16 的非首选 top-5 差呈 1/128 量化步长。

## UNRESOLVED

- 未验证 8K/32K 以上上下文（受 16GB 显存与已测长度限制）。
- 未做 CUDA graph 与 radix cache 默认开启下的性能基准（当前仅功能性验证）。

## 答案

在声明条件下（V100、FP16、Triton、≤4096 token），原生 SGLang LFM3 与官方实现
greedy 输出逐 token 一致，长上下文 hidden 余弦 ≥ 0.999994，残余 logprob 差异处于
FP16 量化噪声量级；超出 4K 上下文与开启 CUDA graph 的结论不在本实验范围内。
