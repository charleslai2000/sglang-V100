# G023 opt-in exact-c57 配对结果

## 配置与产物
- 同模型/配置：Falcon-H1-7B-Instruct-GPTQ-Int4，FP16，c57 chunk=2048，`max-running-requests=4`，`extra_buffer`，默认调度/ CUDA Graph；两侧均 opt-in 禁用 SM70 FP16 reduced-precision reduction。
- Worker 源文件 SHA256 两侧一致：`f5e50ae0566215e0a5f84f0bfc51271166a6e10f528bd862438e08b39c1ba13f`；策略实际值均为 `False`。
- Hit prompt token hash 与 force-miss 一致；hit cached_tokens=768，force-miss=0。
- 服务轨迹：`/data/t011/artifacts/g023-optin-bounded-{hit,miss}-model.jsonl`、`...-sampler.jsonl`；API 结果 JSON 同前缀的 `.json`，服务日志同前缀 `-server.log`。

## 实测
- 四个 branch 请求各生成 256 tokens；alpha 首次、重复、bravo、bravo 后 alpha 的 greedy token IDs 均 hit/miss 完全一致，首分叉均不存在。
- Hit Prefill 为 768-token builder 后接 21-token 新 token + 768 cached；force-miss 每个 alpha/bravo 完整 Prefill 为 789 新 token + 0 cached。
- 两侧 Decode 均按默认 CUDA Graph 调度；服务日志报告 `cuda graph: True`。单请求 decode row、配置 BS=1（Graph key）；未强制 eager/graph。
- 对齐首 alpha Prefill 的 sampler 输入 hash 不同；Top-1 均为 token 1221。选中 token logprob hit `-1.0181455612`、miss `-1.0024816990`，差 `0.0156638634`。完整生成序列所有 greedy IDs 一致；selected-token logprob 仍不等价（alpha 首次/重复/after-bravo 的最大绝对差 `0.0519813299`，位置 90；bravo 最大 `0.1145814657`，位置 17）。
- 该 bounded trace 没有完整 logits 向量，因此无法计算 Top-2 margin；也无模型级逐步 Decode trace（Graph replay 路径未被逐步 hook 记录）。这些限制不影响发现 logprob mismatch。

## 判定
**NO-GO**。贪心输出一致，但必要 logprob 不一致，故不满足冻结 correctness；不开展生命周期扩展或性能矩阵。相同 Graph 调度下 Prefill shape 不同，且首步 sampler logits hash 已不同；不推断超出证据的原因。
