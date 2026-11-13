# G003 long-4096 数值归因

## 配对身份
- 输入：4096 token；SGLang capture 内 IDs 与 `/data/t011/tmp/g003-paired/long4096.ids` 完全相等，SHA256 `96626efea1c5ecd94ab8ac36f9b47e34fc4cafed2560584a9377a7a9cfa716fd`；最终 position=4095。
- GGUF SHA256 `7d6269096c9acbce4422fd1aa3a249436b01288868961e48ca9bb2541997f669`；sidecar config SHA256 `1b00515f93c683596285152325b961b068dd31af995770f03d38c0b9976b6b47`；final softcap 两边均为 30。
- SGLang 为 eager EXTEND 的 post-final-softcap、pre-sampling next-token logits；CPU 导出器 SHA256 `c9b3186353782e03804d75cc8ed48fa6d8dbb2740bed63f70c8e927911025e58`，已修正为只导出最终 batch 的最终 logits。两边 logits 均为 262144 维。CPU raw logits SHA256 `fbc2634d74cf0d854c7545e8d10c486441f6a891ba11de5894215834072b6c16`。
- 证据文件位于本地 `/data/t011/tmp/g003-paired/long4096.live.pt`、`long4096.ids.llama.f32`；该模型/输入数据集未纳入 Git。

## 4K 数值
| 项 | SGLang | CPU | SGLang−CPU |
|---|---:|---:|---:|
| top-1 token / logit | 200306 / 13.6970863 | 84701 / 10.2602777 | — |
| top-2 token / logit | 37850 / 11.5318975 | 236890 / 10.0787754 | — |
| top-1/top-2 margin | 2.1651888 | 0.1815023 | — |
| token 200306 logit / rank | 13.6970863 / 1 | 8.6113310 / 5 | +5.0857553 |
| token 84701 logit / rank | 11.0085497 / 4 | 10.2602777 / 1 | +0.7482719 |

`score(200306)-score(84701)`：SGLang +2.6885366，CPU −1.6489468，双方差 +4.3374834。全词表 MAE=0.6821130、RMSE=0.8080589、max-abs=5.0857553、rel-L2=0.0966571。Top-K overlap：5/5、10/9、20/17、50/42、100/83。可复算完整排名和候选统计：本地 `long4096-attribution-metrics.json`。

## 可证实结论与限制
确认同输入/模型配置/终点阶段下存在显著 4K 数值差异，并造成 greedy 分叉。误差并非只涉及 top-1/top-2 的微小 margin：CPU 的候选排序与 SGLang 明显不同，且 token 200306 相差 5.0858。

没有可比的 CPU、SGLang 逐层最终位置激活数据。本地旧 captures 只含 logits；CPU exporter 也只导出 logits。故本轮不能定位误差首次放大层，亦未能对同一激活、同一解码 GGUF Q4_0 projection 完成最小 FP32 MatMul Oracle。无法区分 Q4_0 表示误差、激活量化、累加/实现差异、注意力或其他前置计算误差。该差距不被忽略，也不足以判定具体生产算子错误。

**裁决状态：差异已证实，根因未确认，G003 未满足原门槛，不报告 PASS。** 本轮停止进一步扩展实验，提供上述配对证据供用户裁决下一步是否授权逐层双端 instrumentation 与 matched-activation oracle。
