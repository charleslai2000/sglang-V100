# E050 结果 — LFM3 三级 HiCache（V100-16GB / FP16 / Graph ON）

结论：**FULL + SWA + Conv State 三级持久化与恢复全部打通并通过验收**；L3 是唯一需要谨慎配置的一环
（页大小与预取超时），否则会慢于 GPU 重算。

## 1. 三级命中分别可验证

`cached_tokens_details` 直接来自服务端，非时延推断。

| 场景 | 触发方式 | cached | device | host | storage | 客户端墙钟 |
|---|---|---|---|---|---|---|
| L1 设备 | 预热后重复请求 | 1024 / 4096 / 8192 | 全中 | 0 | 0 | 187 / 289 / 422 ms |
| L2 主机 | 灌水 200×1024 唯一前缀把设备页挤出 | 768 / 3072 | 0 | 全中 | 0 | 586 / 2024 ms |
| L3 磁盘 | `/flush_cache`（清 L1+L2，留盘）后请求 | 1024 / 4096 / 8192 | 0 | 0 | 全中（`HiCacheFile`） | 215–220 / 364–368 / 559–573 ms |
| L3 冷启动 | **进程重启**（新进程、空 L1/L2） | 1024 / 4096 / 8192 | 部分 | 0 | 其余全中 | TTFT 385 / 254 / 460 ms |

- L1 首次重复请求（第 2 次）可能仍为 0 命中：写穿（write_through）的 host 备份尚未落定，节点还不可复用。
- 8192 冷启动时 `device=4096` 是因为同一进程里刚恢复过的 1024/4096 前缀被重新驻留，非 L3 缺口。
- L3 冷启动 greedy ids 与无 HiCache 基准 **完全相等**（`ids_equal_8192: true`）。

## 2. 恢复内容正确性（对无 HiCache 基准）

基准服务 = 同一命令去掉 `--enable-hierarchical-cache`（`/var/tmp/e050_matrix_baseline.json`）。
HiCache 侧 = page 256 全矩阵（`/var/tmp/e050_matrix_hicache_p256.json`）。

- 9 个用例（1K/4K/8K × append 5/8/10）× 5 条路径（first / L1a / L1b / L3 / L3b）= **45/45 条记录 greedy ids 与基准完全一致**。
- 逐 token 输出 logprob 最大偏差 ≤ **7.5e-3**（阈值 2e-2）；同一用例内 L1/L3 之间偏差为 0。
- page=1 运行（`e050_matrix_hicache` 原始运行）中 L3 与 L1 **逐位相等（0.0）** 9/9；
  唯一 5.2e-3 来自“冷重算 3077 token 预填 vs 命中后只预填 5 token”的 batch-shape 差异，ids 不变。

## 3. 有效前缀边界（实测）

| prompt 长度 | 复用长度 | 说明 |
|---|---|---|
| 1024 | 768 | 尾部 256 无检查点 |
| 1029 (+5) | 1024 | 1024 是内部 256 边界 |
| 4101 / 8197 | 4096 / 8192 | 同上 |

边界 = prompt 内严格小于尾部的 256 整数倍；page=256 时粒度即 256。

## 4. 时延（流式 TTFT，32 输出 token）

| 前缀 | 无缓存重算 | L1 设备 | L3（page=1） | L3（page=256） |
|---|---|---|---|---|
| 1025 | 321.1 ms | 77.0 ms | 1146.8 ms | **124.2 ms** |
| 4097 | 2493.7 ms | 167.0 ms | 4008.0 ms | **280.4 ms** |
| 8193 | 8025.6 ms | 294.0 ms | 8037.4 ms | **473.6 ms** |

同比提速（L3 page=256 vs 重算）：**2.6× / 8.9× / 16.9×**；vs L1 仅多 47 / 113 / 180 ms。
TPOT 三档不受级别影响：7.0–7.1 / 7.7 / 8.6 ms。

原始数据：`e050_timing_page1.json`、`e050_timing_p256.json`、`e050_cold_restart_p256.json`。

## 5. 传输量、磁盘时间、字节账

单 token 张量账（由 config 推导并与落盘文件逐一核对一致）：

| 类别 | 每单位字节 | 核对 |
|---|---|---|
| FULL KV | 12,288 B/token | 6 层 full_attention × 4 kv_head × 128 × 2(KV) × 2 B |
| SWA KV | 10,240 B/token | 5 层 sliding_attention × 同上 |
| Conv(SSM) 检查点 | 319,488 B/检查点 | 13 层 conv × 2048 × (7−1) × 2 B，每 256 token 一个 |

- L2 命中 1024：`load_back_bytes_total` = **12,687,360 B**（1025 token，FULL 池计数）、
  `load_back_tokens_total` = 1025；墙钟 586 ms（含 257 token 重算）。
- L2 命中 4096：`load_back_bytes_total` = **40,998,912 B**、3072 token 复用，墙钟 2024 ms。
- L3 8192：约 **195 MB**（184.5 MB KV + 10.5 MB 检查点）；扣除 L1 基线 294 ms 后 IO+H2D ≈ 180 ms，
  有效带宽 ≈ **1.08 GB/s**；page=1 同样 ~195 MB 却需要 7.7 s ≈ **25 MB/s**。
- 落盘规模：page=1 目录 **502,006 文件 / 5.75 GB**（247,808 个 `.swa_`、253,952 个 FULL、246 个 `.mamba_`）；
  page=256 目录跑完 9 个用例只有 **47 文件 / 123 MB**。

## 6. 并发、驱逐、重复恢复、资源回收

- 并发共享前缀（8 线程，同 1024-token 前缀各加 1 个不同 token）：
  7 条 `device=1024`、1 条 `storage=1024`，`prefetched_tokens_total` +8192、`cached_tokens_total` +8192；
  4096 版本 7 条 device + 1 条 storage，`prefetched_tokens_total` +32768。
- 驱逐后恢复：灌水 200×1024 期间 `hicache_backup_bytes_total` +4.65 GB、`backup_tokens_total` +407,751，
  host 池占用升到 204,800 token / 200 状态；随后目标请求 `host=768`（L2）且 `cached_tokens_total` +768。
- 重复恢复：5 轮 `flush → restore`，全部 `storage=768`，墙钟 1.52–1.60 s，每轮 `load_back_bytes_total` +12,687,360 B。
- 资源回收：`/flush_cache` 后设备池 `kv_used_tokens=0`、`kv_available_tokens` 回到池容量（page=1 运行读数 164,390）；
  host 池保留最近一次请求的占用（512 token 用例为 512 tokens / 1 状态），连续 10 轮 flush **不增长**，
  `hicache_dropped_tokens_total{reason="host_pressure"}` 始终为 0。

## 7. 回归

- 目标单测（12 文件，含新增哈希回退测试）：
  **96 passed + 2004 subtests passed**（`test_hicache_python_hash`、`test_hicache_file_lru_unit`、
  `test_hicache_file_storage_path_unit`、`test_hicache_copy_rounds`、`test_hicache_storage_format_tag`、
  `test_hicache_write_back_ordering`、`test_hicache_load_back_timing`、
  `test_hicache_mamba_slot_side_states`、`test_unified_radix_hicache_dispatch`、
  `test_buffer_swa_tail_only`、`test_hicache_auto_size`、`test_hicache_kvcacheio_gate`）。
- 整个 `test/registered/unit/mem_cache/` 套件在本环境仍不可作门禁：收集期 5 个文件缺 `nixl`/rust 扩展，
  运行到 44% 处 core dump（与 E049 记录一致）；崩溃前出现的失败与本案改动无交集（改动仅 `native_hash.py`，
  唯一消费者是 `mem_cache/utils.py` 的 `get_native_hash`）。

## 8. 已知限制

1. `--page-size 1` 下 L3 慢于重算，生产必须 `--page-size ≥ 128`（本机 256 最佳）。
2. `--hicache-storage-prefetch-policy wait_complete` 在 L3 文件缺失时会无超时挂起；用 `timeout` + 20 s 基准。
3. L3 是“全有或全无”：任一池（FULL/SWA/Mamba）覆盖不足即整段丢弃重算，日志 `reason=deadline`。
4. 未启用 int8 Mamba 检查点压缩（`--enable-int8-mamba-checkpoint` 默认关），host Mamba 池按 FP16 conv 窗口态存储。
5. L2/L3 命中均要求 256 对齐；非对齐尾部照常重算（≤256 token）。
6. 本轮不含 FP8、原生 Probe、HiCache 新调度/异步预取优化（P2.2）。
