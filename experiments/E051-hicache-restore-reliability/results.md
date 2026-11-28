# E051 结果

基座：`v100-harness`（本提交），兄弟参照 `../sglang-hicache@c1feed4d2`。
所有 GPU 数字来自单卡 Tesla V100-SXM2-16GB，配置见 `experiment.md`。

## 1. 正确性：与无 HiCache 基线逐 token 一致

`e051_matrix_final.json`（最终代码）：1K/4K/8K × append 5/8/10 × 5 路径 = **45 条记录**。

| 项 | 结果 |
|---|---|
| greedy ids vs E050 无 HiCache 基线 | **45/45 完全一致** |
| 逐 token logprob 最大偏差 | **7.55e-3**（阈值 2e-2；E050 为 7.5e-3） |
| 实际走 L3 的路径数 | 17 / 45（其余为 device 命中或冷启动） |
| 空响应 / 异常 | 0 |

`e051_repeat_final.json`：同一 1024 前缀 `flush → restore` **5 轮全部 `storage=1024`** 且 ids 一致。

## 2. 生命周期：缺页、取消、迟到 ACK

| 场景 | 结果 |
|---|---|
| L3 缺页（删除 22 个文件中的 3 个，`wait_complete`） | 请求 2.6 s 完成，`cached=0`（整段重算），ids 与参考一致；**无挂起** |
| 取消（6 个请求在预取途中被客户端断开） | 后续 3 次请求全部 `device=8192` 正常；无残留等待、无错误放开 |
| 重复恢复（5 轮） | 全部 `storage=1024`，ids 稳定 |
| 进程内多次 flush + 连续恢复 | host 占用不随轮次增长（`hicache_host_used_tokens` 稳定） |
| anchor 驱逐压力（32 个 8192-token 互异前缀 + 6 个 L3 命中请求并发，`--mem-fraction-static 0.30`） | 38/38 完成，6 个命中请求 ids 全对（1 个 4096 storage + 4096 device，其余 device 直接命中）；日志无 `anchor_invalidated` |

anchor 失效窗口在单卡 page=256 下未能用外部请求构造出来：同一调度轮次就会完成匹配与恢复，
anchor 在 issue→commit 之间被驱逐的可控性不足。该路径因此以单测覆盖
（`TestPrefetchAnchorIdentity` 6 例 + `TestStaleAnchorDiscard` 2 例：驱逐、id 复用、未哈希回退、
split 保身份、只释放本操作 staging、死 anchor 不归还锁、重复终结不二次释放）。

## 3. 容量阶梯：裁剪 / 有界重排队 / 撤销

`e051_capacity_tiny.json`（`--hicache-host-memory-fraction 0.02`，0.3 GiB host）：

| 并发 | 完成 | ids 一致 | 证据 |
|---|---|---|---|
| 16 | 16/16 | 16/16 | `storage_prefetch_unfulfilled_tokens_total{reason="host_capacity"}` = **12,800** |
| 64 | 64/64 | 64/64 | `storage_prefetch_deferred_tokens_total{reason="host_capacity"}` = **12,800** |

含义：容量不足时先按页对齐策略尝试裁剪；LFM3 的 SWA/MAMBA 属 `TRAILING_PAGES`，**前沿一律不裁剪**
（裁剪会移动已校验的尾部端点），因此走"有界重排队（2 次）→ 仍不足则整体撤销并重算"。
无挂起、无错误输出、无重复释放（账目只记一次：unfulfilled 与 deferred 相等）。

0.8 GiB host 对照（`e051_capacity_small.json` / `e051_capacity24.json`）：8/8、32/32 全部完成，
ids 全部一致，host_capacity 计数 3,328。

`e051_stress.json`：c32 × 4 轮 × 8192 token 内容互异，**4 轮均 32/32 完成**，零失败。

## 4. `wait_complete` 活性（E050 挂起缺陷）

预修复事实：`_can_terminate_prefetch` 对 `wait_complete` 恒 `False`，且只有 `timeout` 策略会启动
预算计时，因此任何未到达的完成 ACK 都会永久卡住 admission。

现行为（`--prefetch-timeout-base 1e-3` 强制暴露）：日志出现

```
HiCache prefetch stalled req=... policy=wait_complete completed=0 requested=8192
waited=0.0s budget=0.0s; retrying instead of blocking admission
```

三次请求全部在 8.18 s 内以整段重算完成，ids 与基线一致；默认 20 s 预算下正常恢复不受影响
（见 §1、§5）。单元回归 `TestWaitCompleteLiveness` 固定两条不变量：**有进度的传输永不被打断**、
**无进度的等待在预算后被切断并重试**。

## 5. 性能（`wait_complete` + page=256，32 token 输出，流式）

`e051_timing_final.json`：

| 前缀 | 重算 TTFT | L3 恢复 TTFT | L1 TTFT | TPOT |
|---|---|---|---|---|
| 1024 | 603 ms | **111 ms**（5.4×） | 79 ms | 7.1 ms |
| 4096 | 2209 ms | **244 ms**（9.1×） | 168 ms | 7.7 ms |
| 8192 | 5837 ms | **423 ms**（13.8×） | 288 ms | 8.6 ms |

TPOT 与缓存级别无关（7.1–8.6 ms），与 E050 一致；`wait_complete` 相对 E050 的 `timeout` 策略
（8K 474 ms）在此配置下更快（423 ms），因为不再有 20 s 预算的边界抖动。

并发（1024 前缀 + 64 token 输出，4 轮，奇数次 flush 强制 L3 恢复）：

| 并发 | 吞吐 | p50 延迟 | 恢复 |
|---|---|---|---|
| c8 | 562–636 tok/s | 0.79–0.90 s | flush 轮 1 次 L3 + 7 次 device |
| c32 | 1179–1304 tok/s | 1.54–1.71 s | flush 轮 1 次 L3 + 31 次 device |

## 6. 回归

- 新增 `test/registered/unit/mem_cache/test_hicache_restore_reliability.py`：**23 passed**。
- 逐文件回归（`mem_cache/unified_radix_cache/*` + `mem_cache/*` 的 hicache/prefetch/staging/
  buffer/swa/storage 子集，**39 个文件逐个跑**）：**1379 passed + 2201 subtests**，无失败。
- 环境限制（与本改动无关）：`test_unified_hicache_transfer_lifecycle.py::TestSwaLoadBackReservation`
  在本机 core dump（`git stash` 后在未改动代码上同样崩溃）；`nixl` / `rust_tree_core` / `umbp`
  缺依赖，跳过。

## 7. 因果 trace（可核对）

| 观察 | 位置 |
|---|---|
| 裁剪长度判定 | `hybrid_cache_controller.staging_crop_length` / `storage_hit_crop_is_safe` |
| 有界重排队 | `unified_radix_cache._defer_staging_attempt`（预算 `prefetch_staging_deferrals=2`） |
| anchor 身份 | `_OngoingPrefetch.anchor_last_hash`（发起点记录）→ `_prefetch_anchor_is_valid`（提交前） |
| 失效撤销 | `_discard_prefetch_stale_anchor`（只释放本操作 staging；anchor 锁仅在节点仍存在时归还） |
| 停顿检测 | `_start_storage_prefetch_wait` / `_wait_complete_stalled` / `_mark_prefetch_stalled` |
| 观测口径 | `storage_prefetch_{hit,unfulfilled,deferred}_tokens_total{reason=...}`；日志 `prefetch cropped` / `prefetch stalled` / `prefetch discarded ... anchor_invalidated` |
