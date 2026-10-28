# G002-N Dense WMMA ModelRunner 资格结果

## 最终判定

**NO-GO：生产保持 Triton，不进行 service c1/c4。** 固定上游 dense D512 ABI 已在真实 ModelRunner 接通；owner pool gather、same-input shadow、1K/4K Forward A/B 均已运行。但完整 logits 不满足既有数值验收：1K max abs `0.06045`、MAE `0.01168`；4K max abs `0.05073`、MAE `0.008984`。greedy token 相同（623）不足以覆盖完整 logits 差异。

## 固定上游依赖与 dense binding

来源 `1CatAI/1Cat-vLLM` revision `fb38933cf62f1fdb6e99da15b82f93b930fae076`。`fused_mha_forward.cu` 原样包含 D512 `flash_attention_forward<512>`，D512 tile `(BLOCK_M,BLOCK_N,WARPS)=(16,32,16)`；`fused_mha_api.cpp` 原注册 `fwd -> flash_attention_forward`。实验依赖位于 `vendor/upstream-dense/`，dense kernel 未改数学算法；最小 pybind shim 只导出该 fixed upstream `fwd`，build 仅编译该 kernel 与 wrapper。CUDA 12.8、SM70 构建成功，Python import 确认 `fwd` 可用。保留上游 BSD-3-Clause LICENSE。构建日志：`/data/t011/artifacts/g002n/dense-build.log`。

## 生产路径和 same-input shadow

在真实 `TritonAttnBackend.forward_extend` 已执行 KV 写入、已形成 metadata 后，dense adapter 读取 `pool.get_kv_buffer(layer.layer_id)`（含共享层 owner 解析），按 prefix metadata slots + `out_cache_loc` extend 顺序 gather K/V，布局转换成 dense ABI `[1,Hkv,N,D]`；Q 转换为 `[1,Hq,M,D]`。未更改 allocator、pool、owner mapping、RoPE、生产 metadata 或默认 dispatch。`SGLANG_G002N_MODE=shadow` 只读比较 dense upstream 输出与生产 Triton 输出；`dense` 模式可 opt-in 调用 dense 输出，默认不开启。

日志 `/data/t011/artifacts/g002n/dense-ab.log`：slots 顺序和 gathered K/V 与生产输入相同；scale=1、causal=true。七层 owner 为 5→5、11→11、17→17、23→23、29/35/41→23。首个非零差异是 layer 5。1K 七层 MAE：`7.88e-6, 6.51e-6, 6.04e-6, 5.26e-6, 4.74e-6, 4.58e-6, 4.55e-6`；4K：`1.51e-5, 1.61e-5, 1.51e-5, 1.26e-5, 1.27e-5, 1.25e-5, 1.23e-5`。两者各层 max abs 均 `0.00390625`，与 FP16 舍入误差一致。

## 完整 Forward A/B

交错顺序 Triton/Dense/Dense/Triton/Triton/Dense；Event 覆盖真实 `ModelRunner.forward`，含生产写 cache、dense gather、布局转换、binding 调用及其它 Forward 工作。Dense adapter CUDA Event 七层分项 medians 总和：1K gather `1.133 ms`、Q/K/V layout `1.254 ms`、kernel `9.315 ms`、output layout `0.208 ms`；4K 分别 `1.230/1.666/103.464/0.703 ms`。Raw 代表样本：

| 长度 | Triton Event | Dense Event | Dense 峰值 allocated | 峰值 reserved |
|---:|---:|---:|---:|---:|
| 1K | 302.5–305.5 ms（其中首样本 404.5 ms 为 probe 首轮抖动） | 268.7–271.5 ms | 11.81 GB | 13.37 GB |
| 4K | 1761.2–1763.5 ms（shadow 轮次有额外开销） | 1246.1–1249.5 ms | 12.37 GB | 13.37 GB |

Dense 的 Forward Event medians 为 1K `271.320 ms`、4K `1247.561 ms`；Triton 分别 `302.697 ms`、`1762.758 ms`。观测改善约 `10.4%`/`29.2%`，peak allocated 约 `11.81/12.37 GB`（Triton 约 `11.81/12.47 GB`），reserved 均 `13.37 GB`。然而完整 logits 差异明显不合格，因此不把改善认作合格净收益，不绕过数值门槛，未运行 service c1/c4。

## 范围与复现

源码修改在 `experiments/G002-N/` vendor/build/probe 和 `python/sglang/srt/layers/attention/{g002n_wmma.py,triton_backend.py}`；生产默认仍走 Triton。`type.cuh` 未改。构建环境为 `/data/t011/venv/sglang`、CUDA `/data/t011/toolkit/cuda-12.8.1`、授权 GPU UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`。仍待把本轮变更通过最终检查后提交推送；不要将 NO-GO 解释为 upstream ABI 不支持。
