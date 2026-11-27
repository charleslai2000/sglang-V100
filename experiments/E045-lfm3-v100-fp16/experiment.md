# E045 — LFM3-1.7B 原生 FP16 推理在 V100 的正确性验证

## Question

原生 SGLang LFM3 文本推理（SM70、FP16、Triton attention、Hybrid SWA+conv 缓存）在
V100-16GB 上，是否与官方 Transformers 参考实现在 token、logprob 与长上下文 hidden
上数值一致？

## 实验对象

- 权重：`LiquidAI/LFM3-1.7B`（`/var/tmp/lfm3`），FP16。
- 生产路径：`sglang.launch_server` / `sgl.Engine`，`--attention-backend triton`、
  `--sampling-backend pytorch`、`--disable-cuda-graph`。
- 参考：独立 venv 的 `transformers.models.lfm3`（`transformers>=5.12`），同权重 FP16。

## 受控输入

同一段印刷史英文文本按 token 截断为 5 组：12 / 200 / 300 / 1000 / 4096，
另含 128 步 greedy 解码续写、chunked prefill（1024/2048）、prefix cache 复用。

## 过程

1. 官方参考端做 prefill + teacher-forced greedy 解码，记录 IDs、逐 token logprob、
   全 prompt logprob（按 256 位置分块，规避 16GB OOM）、4K hidden 采样。
2. SGLang 端以相同输入分别跑完整 prefill（`chunked_prefill_size=-1`）与分块 prefill，
   采集同一组量。
3. 逐位置对比，定位差异层，修正实现后重跑。

## 测量项

greedy token IDs 相等性；生成与 prompt token logprob 差值；next-token top-5；
4K hidden 余弦相似度与 RMSE；解码 128 步一致性；吞吐与显存。
