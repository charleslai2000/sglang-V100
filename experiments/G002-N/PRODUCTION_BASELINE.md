# G002-N 生产 Triton 对照契约

## 生产 kernel 与配置

生产 `TorchNativeAttnBackend.forward_extend` 在 `k/v=None`（Gemma4 KV-sharing）时，从 `token_to_kv_pool.get_kv_buffer(layer.layer_id)` 按 `out_cache_loc` 取 owner cache；否则先将本层 K/V 写入 pool。D512 Full 层进入 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py::extend_attention_fwd`，而非单独的 `prefill_attention.context_attention_fwd`。后者虽名为 context prefill，但在 SM70、D512 下固定 `BLOCK_M=BLOCK_N=64, BLOCK_DMODEL=512, num_warps=8, num_stages=1`，会要求 256 KiB shared memory，并非有效生产 baseline。

G002-M 真实 Forward 的 NVTX/CUDA correlation 将七个 Full 层范围直接关联到 Triton `*_fwd_kernel`；生产 extend kernel 的动态 tile 由 `_get_block_sizes_for_extend_attention(Lq=512,Lv=512)` 选择，SM70 路径为 `BLOCK_M=16, BLOCK_N=16, BLOCK_DMODEL=512, num_warps=8`（需由该 revision 上真实 Triton kernel metadata / CUDA trace 再确认）。离线 `extend_attention_fwd` 需要 paged cache indptr/indices 和 prefix/extend 结构；将它替换成 page-size-1 context kernel不能声称等同生产路径。

## 结果边界

capture 中生产 attention pre-hook 的 Q 已经完成 Q RMSNorm 和 RoPE；owner K/V 已完成 K/V norm 和 K RoPE。共享层 Q 按同一层运算，KV owner 是 23，生产 `layer_id` 对 29/35/41 都是 23，且 `save_kv_cache=False`。测试 harness 的连续物理 slot/恒等 Full-to-SWA 映射只覆盖本次单序列全量 prefill，不覆盖分页碎片与线上 allocator。需要在真实 Forward 中捕获 Triton 原始 kernel 输出/或重建精确生产 metadata，才有请求绑定的三方比较证据。
