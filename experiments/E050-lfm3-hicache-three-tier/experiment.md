# E050 — LFM3 三级 HiCache 适配（V100-16GB / FP16 / CUDA Graph ON）

## 目标

在官方 v0.5.23 `UnifiedRadixCache` + `HybridCacheController` + `StorageAttachment` 架构上，
为 LiquidAI/LFM3-1.7B 打通 **FULL KV + SWA KV + Conv(SSM) State** 三级持久化与恢复：

- L1 = 设备 radix cache（GPU）
- L2 = host pool（CPU，`--hicache-host-memory-mode cache`）
- L3 = file storage backend（磁盘，独立目录）

不改缓存架构、不引入 FP8 / 原生 Probe / 新调度策略；兄弟仓库 `../sglang-hicache` 仅作只读参考。

## 冻结配置

| 项 | 值 |
|---|---|
| 模型 | `/var/tmp/lfm3`（LFM3-1.7B，`layer_types` = 13 conv / 6 full_attention / 5 sliding_attention，kv_heads=4，head_dim=128） |
| 精度 / 后端 | FP16，`--attention-backend triton`，`--sampling-backend pytorch` |
| 长度/并发 | `--context-length 8448`，`--max-running-requests 32`，`--cuda-graph-max-bs-decode 32`，`--mem-fraction-static 0.60` |
| Radix | `--mamba-radix-cache-strategy extra_buffer`（`mamba_track_interval=256`，LFM3 `mamba_chunk_size=256`） |
| HiCache | `--enable-hierarchical-cache --hicache-host-memory-mode cache --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-storage-backend file`（`kernel` 后端同样实测可用，见结果 5） |
| 页大小 | **`--page-size 256`**（对照实验：`--page-size 1`） |
| 预取策略 | `--hicache-storage-prefetch-policy timeout` + extra_config `prefetch_timeout_base=20.0, prefetch_timeout_per_ki_token=5.0` |
| 观测 | `--enable-metrics`（`/metrics` 提供 HiCache 计数） |
| 环境 | `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`（sglang-kernel 0.4.3 < 0.4.9） |

池规模（page=256 启动日志）：FULL host 328,960 tok / 4.04 GB，SWA host 262,912 tok / 2.69 GB，
MAMBA host 18,965 slots / 6.06 GB（`page_first_direct`），设备 `max_total_num_tokens=164,352`
（page=1 时 164,390）。

## 关键机制（实测确认）

1. **有效前缀边界**：可复用前缀 = prompt 内**严格小于尾部**的 256 整数倍（`mamba_track_interval` 与
   `max(mamba_chunk_size, page_size)` 的最小公倍数）。例：prompt 1024 → 复用 768；prompt 1029 → 复用 1024。
2. **L3 是全有或全无**：混合缓存要求 FULL/SWA/Mamba 三池覆盖同一段前缀，任一池不足即整段丢弃
   （日志 `HiCache hybrid prefetch discarded ... reason=deadline`），并重算。
3. **`wait_complete` 会挂死**：L3 文件缺失/被淘汰时该策略无超时等待；必须用 `timeout` + 足够大的
   `prefetch_timeout_base`（本机 20 s）。
4. **`page_size=1` 不可用于 L3**：每 token 一个文件（一次 8K 请求产生 ~1.6 万文件），磁盘路径退化为
   比 GPU 重算更慢；256 页把文件数降到 1/256 后恢复为净收益。

## 兼容性改动

唯一代码改动：`python/sglang/srt/mem_cache/cpp_utils/native_hash.py`

- `hicache_hash_cpp` 需要 OpenSSL 头文件（本机无 `libssl-dev`），构建失败会让 HiCache 存储查询整条链路不可用。
- 新增纯 Python SHA-256 回退 `python_get_hash`，逐字节复刻 `hash_binding.cpp`：LE uint32 页字 →
  逐页链接 prior digest；wide 路径（token > 2^32）先写 `0xff` 标签再写 LE uint64；bigram 以 (prev,next) 成对入哈希。
- 后端选择 `SGLANG_HICACHE_HASH_BACKEND=auto|native|python`（默认 auto：能加载扩展就用扩展，否则回退并告警）。
- 单测 `test/registered/unit/mem_cache/test_hicache_python_hash.py`：7 例，用手算摘要锁定语义。

其余“适配”均为启动参数（页大小、预取超时、存储目录），未改上游代码路径。

## 测量方法

- 级别归因：请求显式带 `return_cached_tokens_details`，响应 `meta_info.cached_tokens_details`
  直接给出 `device / host / storage / storage_backend`，不靠时延推断。
- 传输量与用时：`/metrics` 的 `hicache_backup_bytes_total`、`load_back_bytes_total`、
  `hicache_backup_tokens_total`、`load_back_tokens_total`、`hicache_host_used_tokens`、
  `hicache_host_mamba_used_states`（阶段前后取差）。
- 磁盘文件账：按后缀分类统计（FULL `.bin` / `.swa_` / `.mamba_`）文件数与字节数。
- 时延：SSE 流式首包 = TTFT，`(总时长 − TTFT)/(n−1)` = TPOT。

脚本在本目录：`e050_probe.py`（三级命中）、`e050_compare.py`（冷/热/盘对比）、
`e050_matrix.py`（1K/4K/8K × 5/8/10 append 矩阵）、`e050_phases.py`（L2/并发/重复恢复 + 指标）、
`e050_timing.py`（TTFT/TPOT）。

## 判定标准

- 三个级别必须分别可验证命中，且 L3 能覆盖**服务重启后**的冷恢复；
- 恢复后的 greedy ids 与候选 logprob 与“无 HiCache 基准”一致（同 prompt、同采样）；
- 资源可回收：flush 后设备池归零、host 池不随轮次增长、无 `host_pressure` 丢弃。
