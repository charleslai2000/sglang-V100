# G002-N Production Pool Gather 接口结论

## 生产调用与布局

真实路径为 `TritonAttnBackend.forward_extend(q,k,v,layer,forward_batch,save_kv_cache=True,sinks=None)`：先按 `forward_batch.out_cache_loc` 调用 `self.token_to_kv_pool.set_kv_buffer(...)`，再调用 `self.extend_attention_fwd(...)`。Gemma4 `Gemma4Attention.forward` 将共享层通过 `RadixAttention.layer_id` 指向 owner，并以 `save_kv_cache=False`、`k=v=None` 调用 attention；故 backend 用 `pool.get_kv_buffer(layer.layer_id)[cache_loc]` 读取 owner 的生产 cache。这里已有完整的生产 owner 解析，不应再写死 owner layer。

`ModelRunner` 为 hybrid SWA 配置 `SWAKVPool`（`model_runner_kv_cache_mixin.py`，Full/SWA 层分别持有 `MHATokenToKVPool`）；cache 主体为 `[size + page_size, Hkv, D]` 的 token-major K/V tensors（`memory_pool.py:1073`）。`SWAKVPool.get_key_buffer/get_value_buffer` 根据 `layers_mapping[layer_id]` 选择正确的 Full 或 SWA pool，返回相应物理 token buffer；Full-to-SWA 逻辑索引映射由 `translate_loc_from_full_to_swa` 实现。生产 Attention metadata 的 `kv_indices` 是逻辑 token slot 索引；索引内容来自 `req_to_token_pool`，而 prefix 与 extend 的 KV 分别通过 metadata 和 `out_cache_loc` 表示。零 prefix 时，全序列 key 索引来自 extend 的 slot 集合，按生产语义可将 `out_cache_loc` 作为 dense slot gather 的顺序来源；key/value 则必须通过 `pool.get_kv_buffer(layer.layer_id)` 获取已解析 owner 的实际 Full pool。

V100 启动参数 `attention_backend=triton` 不触发 `_handle_sm70_backends` 中 TileLang/FlashAttention 自动 page-size 分支；通用 `_handle_page_size()` 会将未指定值设置为 1。本轮已启动真实 `ModelRunner`（model `/data/t011/models/gemma4-w4a16-ct`，Triton、disable-radix-cache、chunked prefill 4096，mem fraction .76）并直接打印池对象：`SWAKVPool`, Full pool K/V 均为 `(17232,2,512)`、stride `(1024,512,1)`、FP16；allocator `SWATokenToKVPoolAllocator`; page size=1。Full layers `[5,11,17,23,29,35,41]` 的 owner 分别为 `[5,11,17,23,23,23,23]`，`layers_mapping` 为 `(0,false),(1,false),(2,false),(3,false),(3,false),(3,false),(3,false)`。这确认 owner 从 `Gemma4Attention` 的 `kv_shared_layer_index` 以及 `SWAKVPool.layers_mapping` 可动态读取，无需硬编码。模型/Pools 已加载约 11.62 GB allocated、13.34 GB reserved；这只是 runner 初始化基线，不是对比 Forward 峰值。

## 当前固定 WMMA binding 不兼容位置

源码固定扩展 `experiments/G002-N/vendor/flash-attention-v100/kernel/prefill_paged_sm70.cu` 导出的接口为：

```cpp
at::Tensor flash_attention_prefill_paged(
  const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
  std::optional<at::Tensor>& out, const at::Tensor& block_table,
  const at::Tensor& seq_lens, float softmax_scale,
  const std::string& kv_cache_dtype, float k_scale, float v_scale,
  bool is_causal, int window);
```

Python 导出名为 `prefill_paged_fwd`，不是 `fused_mha_forward`。该 binding 的 C++ 检查明确要求 Q `[1,8,M,512]`、KV `[pages,16,2,512]`、`k_cache.size(1)==16`、页表及序列长度；而本轮实际授权的历史 Gemma4 服务配置显式选择 `attention_backend=triton`（`/data/t011/artifacts/g002i_launch.sh`），因此不会走 SM70 的 `tilelang_fa_v100` 自动 page-size 设置；若不指定 `--page-size`，通用逻辑默认为 1。该配置的池形状是 3D token-major `[size+page_size,Hkv,D]`（每 slot 地址连续，strides 为 `[Hkv*D,D,1]`）；在 page_size=1 时并非 page-major 的 `[pages,16,Hkv,D]`。用户要求“实际生产”配置，不能擅自改启动参数伪造 16 页池。接口也没有接收任意 token 索引或 dense K/V 的入口。把 metadata `kv_indices` gather 到 `[M,2,512]` 后仍不能直接调用该 ABI；需要新建支持 dense K/V（如 `[1,2,N,512]`）的 WMMA binding/kernel，或扩展并重新验证固定上游 binding。现有 `fused_mha_forward_paged.cu` 同样实现 paged kernel，接口检查相同，不是 dense `fused_mha_forward` 接口。

## 同输入 Shadow 结果

Opt-in `SGLANG_G002N_SHADOW=1` 仅在真实 Triton Attention 完成后，读取相同已 RoPE 的 Q、owner cache 和生产 metadata，计算 WMMA 影子结果；Triton 结果仍被返回，shadow 不写 KV。真实 1K/4K Forward 均检查到 metadata prefix 为空，extend token 顺序必须接 `out_cache_loc`；shadow 验证其与生产 K/V 输入逐元素一致。GQA 由上游 kernel 的 `q_head_id / (H / Hkv)` 映射，且使用 scale 1、causal 全序列位置。完整七层结果在 `/data/t011/artifacts/g002n/shadow-ab.log`：1K 首个非零差异为 layer 5，MAE `7.88e-6`；4K 首个非零差异同为 layer 5，MAE `1.51e-5`；各层 max abs `0.00390625`。这是相同输入下的浮点舍入差异，真实模型继续走 Triton，最终 logits 与 Triton 基线完全一致。

## ABI 与判定

固定 revision 只提供 `prefill_paged_fwd`，强制 KV `[pages,16,2,512]`，没有 dense WMMA binding。shadow 暂以 scratch page 打包来比较数学结果，但它不是批准的 production-pool→dense ABI，Forward A/B 没有启用 WMMA dispatch。真实 shadow Forward CUDA Events（包含对照开销）1K Triton 366.3–367.4 ms、4K 2548.2–2552.1 ms；这些不是候选性能数据。生产默认仍 Triton。数值边界对照 GO；dense ABI 与 WMMA production Forward 净收益 **NO-GO/未完成**；不进行 service c1/c4。
