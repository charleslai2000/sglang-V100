# G002-N 真实生产池 WMMA 对照

## 判定

**当前 NO-GO：生产保持 Triton，不做 service c1/c4。** 真实同输入 attention shadow 已完成并定位首个差异；数值误差小且逐层结果稳定，但固定上游 WMMA binding 只有 page-size-16 paged ABI，没有目标 dense API。真实 Forward 候选 WMMA A/B 未建立，故不能据 shadow 宣称性能或完成原里程碑。

## Shadow 方法与生产数据

在 `TritonAttnBackend.forward_extend` 的 D512 Full Attention 边界，生产 Triton 先运行并生成将被模型使用的输出；opt-in shadow 随后用相同 post-RoPE Q、owner cache、metadata/causal 语义调用 WMMA，并独立重跑 Triton 生成比较输出。Shadow 不写缓存，也不改变返回结果。Extend attention 的 prefix `kv_indices` 在零 prefix 时为空，因此有效序列由空 prefix 加 `out_cache_loc` 顺序构成；shadow 验证 gathered K/V 与真实生产 K/V 输入逐元素相等。Hq/Hkv 映射由固定上游 kernel 实现为 `q_head_id / (H / Hkv)`，scale=1，causal 位置对齐，序列长度使用实际 Q 长度。共享层经 `RadixAttention.layer_id` 读取 owner：5→5、11→11、17→17、23→23、29/35/41→23。

证据日志：`/data/t011/artifacts/g002n/shadow-ab.log`。此前真实 Forward WMMA 尝试的日志保留于 `/data/t011/artifacts/g002n/production-ab.log`，其 Forward 数值差异不再视为 WMMA 与 Triton 的 kernel 同输入差异；Shadow 证明它们使用了不同的真实层输入，因为 WMMA 改变早期层结果会改变后续层 Q。

## 逐层数值结果

比较的是真实层输入下的 attention 输出，误差为 FP16 输出差异；真实后续模型仍使用 Triton 输出。

| 长度 | 层 | Owner | Max abs | MAE |
|---:|---:|---:|---:|---:|
| 1K | 5 | 5 | 0.00390625 | 7.88e-6 |
| 1K | 11 | 11 | 0.00390625 | 6.51e-6 |
| 1K | 17 | 17 | 0.00390625 | 6.05e-6 |
| 1K | 23 | 23 | 0.00390625 | 5.26e-6 |
| 1K | 29 | 23 | 0.00390625 | 4.73e-6 |
| 1K | 35 | 23 | 0.00390625 | 4.59e-6 |
| 1K | 41 | 23 | 0.00390625 | 4.56e-6 |
| 4K | 5 | 5 | 0.00390625 | 1.51e-5 |
| 4K | 11 | 11 | 0.00390625 | 1.61e-5 |
| 4K | 17 | 17 | 0.00390625 | 1.51e-5 |
| 4K | 23 | 23 | 0.00390625 | 1.26e-5 |
| 4K | 29 | 23 | 0.00390625 | 1.27e-5 |
| 4K | 35 | 23 | 0.00390625 | 1.25e-5 |
| 4K | 41 | 23 | 0.00390625 | 1.23e-5 |

1K 与 4K 的首个非零差异均在 layer 5；误差与 WMMA/Triton 累积计算次序造成的 FP16 rounding 量级一致。每次真实 Forward 的 WMMA shadow 与独立 Triton baseline 比较，所有七层最终 capture 与最终 logits 完全相同，next token 相同（623），因为 shadow 输出未进入后续模型计算。

## 真实 Forward Event 及显存

本轮启用 shadow 所测 Forward 仍全程由 Triton 驱动，且额外运行 WMMA、重复 Triton attention，不能视为候选 A/B：1K Triton arm 366.3–367.4 ms、标作 WMMA 的 arm 365.8–367.2 ms；4K 分别 2548.2–2552.1 ms、2546.4–2555.0 ms。峰值 allocated 为 1K 11.84 GB、4K 12.49 GB，reserved 13.39 GB。这里的“arm”仅是遗留 probe label，dispatch 始终 Triton。service 未运行。

之前的生产 WMMA dispatch尝试由 `/data/t011/artifacts/g002n/production-ab.log` 留存：观测到 1K/4K Forward 时间较低但层输出和 logits 出现显著差异。Shadow 闭环表明该结果受各层输入逐层分叉影响，不能作为数值合格的同输入 kernel 性能证据，故保留但不采纳。

## 固定上游 ABI 限制

固定上游 revision 的唯一绑定为 `prefill_paged_fwd`，源码要求 Q `[1,8,M,512]`、KV `[pages,16,2,512]`，page size 必须 16。实际生产池为 token-major page size 1；其余 dense WMMA 源码并未绑定为 dense ABI。当前 shadow 为离线数学比较临时打包 scratch pages；生产 dispatch没有调用 WMMA。按约束不改 allocator、pool、默认路径，也不通过假设 page size 16 伪装目标接口。

## 下一步与保护项

要完成原性能里程碑，必须在固定上游 WMMA 数学实现上增加并审查真实 dense `[1,2,N,512]` KV binding，再重建扩展，重跑同输入 shadow 与真实 1K/4K WMMA dispatch A/B，核算完整 gather/布局/分配成本和峰值显存。只有数值合格且稳定净收益为正，才进行 service c1/c4。当前无 service 结果，无生产 GO。未改 `type.cuh`；未提交或推送。