# 兄弟仓库 HiCache 成果审计（对 v100-harness@5a9061f0c3）

审计对象：`../sglang-hicache`（分支 `3090-24g-hicache`，HEAD `592a9ef99`，官方基线 tag
`v0.5.6.post2`=`5c8bd8b51`，其后自有 HiCache 主线）。
对照对象：本仓库 `v100-harness@5a9061f0c3`（官方 v0.5.23 前向合并，见 E049）。
本轮只读源仓库、不跑新基准、不改生产代码、不决定移植范围。

## 0. 结论摘要

1. **兄弟仓库 = 旧基线 + 自有 HiCache 二期程序**，不是"v0.5.23 之上再加一层"。
   其 HiCache 相关自有提交 56 个（作者 `charles`/`charleslai`/`opencode`，
   触及 `mem_cache`/`scheduler`/`scheduler_components`/`observability`/`cache_controller`），
   热点：`unified_radix_cache.py` 28 次、`scheduler_components/hicache_restore.py` 12 次、
   `storage/extent/extent_file_backend.py` 10 次、`hybrid_cache_controller.py` 8 次。
   `SOURCE FACT`
2. **多数"兄弟新机制"在 v0.5.23 已有不同实现**：我方 `storage_prefetch.py`（`StoragePrefetchRetries`）、
   更完整的 `buffer_mode/pipeline.py`（2025 vs 961 行，含 `staged_span_covered`/`anchor_lost`）、
   更完整的 `storage_existence_cache.py`（199 vs 89 行）、`restorable_prefix_pages`、
   `unservable_prefix`/`device_covered` 拒绝原因、`declined_too_short` 抖动闸门均为我方独有或更全。
   反向缺口：extent L3 后端、恢复状态机、Ready Horizon、host 可分配代与唤醒、per-request 遥测。
   `SOURCE FACT`（符号/规模矩阵见 §2）
3. **兄弟仓库自身有一处"保护语义退化"**：`5b61e6399` 的裁剪保护
   `_storage_hit_crop_is_safe` 在 HEAD 只剩定义、**无任何调用点**；其最终分配路径改为
   "按验证前沿全量分配，失败即拒绝（`revoke_pending_prefetch(alloc_failed=True)`）"，
   即用更强的"不裁剪"替代了"有条件裁剪"。移植应以该**最终语义**为准，而非照搬函数。
   `SOURCE FACT`
4. **没有可宣称的性能净收益**：R16C 未能确认源码回归（参数匹配 c8 316.55 tok/s vs 历史
   371.2，−14.7%，归因 `UNKNOWN`）；Ready Horizon 默认关闭且三组 ON/OFF 配对明确
   "不是吞吐收益证据"；extent 后端只有介质级与 syscall 级数据，无端到端验收。
   `SOURCE FACT`（记录原文），`INFERENCE`（对 LFM3 的收益需我方自测）

## 1. 审计方法与可追溯性

- 提交集：`git log --author={charles,charleslai,opencode} -- python/sglang/srt/{mem_cache,managers/scheduler.py,managers/cache_controller.py,managers/scheduler_components,observability,environ.py,server_args.py,io_struct.py}`
  → 56 个提交；按路径细分 36 个直接落在 `mem_cache`+`cache_controller`。
- 文档：`control/frontier.md`、`control/queued-prefetch-host-ready/{goal,plan}.md` 与
  `runs/T004-FALCON-20261016/FINDINGS.md`、`control/falcon-h1-3090-hicache-perf/runs/R16C-*/`、
  `control/falcon-h1-3090-online-hicache/report.md`、`.memory/hicache-l3-extent-storage/ledger.md`、
  `.memory/phase2-hicache-design/`。
- 语义对照：不按函数名比对，改为"机制—不变量—可观测结果"三问，并对我方做符号/行为核对
  （`grep` 实测 + 源码阅读）；我方结论均在 §2 标注依据。
- 源仓库状态：分支 `3090-24g-hicache`，`git status` 有 1 个既有 dirty 文件；本轮未切换分支、
  未改动其工作树。

## 2. 功能差异矩阵（主表）

判定：**等价** = 我方已有语义等价实现；**部分** = 我方有相邻机制但覆盖不全；
**缺失** = 我方无对应能力；**冲突/硬件相关** = 与 v0.5.23 架构或 SM70/PCIe 现实冲突。

### A. FULL/SWA/MAMBA 混合缓存

| # | 能力 | 兄弟仓库（机制/提交/位置） | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| A1 | 混合联合恢复：FULL+sidecar 同前沿、尾部键重对齐 | `PoolHitPolicy.TRAILING_PAGES`、`_sync_trailing_keys`、min-clamp（`batch_exists_v2` 各池命中取 min） | 代码 + T004 逐笔 `pc_storage_query`/`pc_get_v2`/`pc_l3_host` | 生产 | 同机制（`_sync_trailing_keys`、`restorable_prefix_pages`、`all_or_nothing`） | 等价 |
| A2 | 全部或全无语义 | `all_succeeded`（我方同款）+ 其结果必须发布给调度器 | `67b39b5ac`、`6881529ac` | 生产 | 我方在 `_finish_storage_prefetch(fulfilled=0)` 后释放请求 | 部分 |
| A3 | **混合前沿裁剪保护** | `5b61e6399` 加 `_storage_hit_crop_is_safe`；HEAD 退化为"全量分配或拒绝" | 提交 + 我方核对（HEAD 无调用点） | 生产（以更强形式） | `allocate_storage_hit(allow_partial=True, min_tokens=threshold)` 仍会裁剪 | **缺失** |
| A4 | 混合就绪拒绝：丢弃结果不得当 READY | `record/pop_storage_prefetch_completion` + 调度器把 `discarded/rejected` 映射为 `DENIED_SHORT`+`short_tail_recompute` | `67b39b5ac`（测试 `test_rh_contract_dependency_free.py`） | 生产 | 无逐请求完成记录；丢弃即按 reason 放行重算 | 部分 |
| A5 | 存储空洞 D→H 备份守卫（设备值存在性） | `cc718c3c2`：storage-gap 时按 device value 存在性决定是否备份 | 提交 + 新增测试 | 生产 | `publish_node_ids`/`ongoing_write_through` 有 ACK 解锁，无该守卫 | 缺失 |

### B. 缓存池、容量分配与驱逐

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| B1 | `--hicache-size` 在 KV 与 MAMBA 间按比例切分 | `_split_hicache_size()`；实测 4 GiB 总额只给 FULL KV 0.22 GiB、MAMBA 3.90 GiB；ratio=10 试图分配 62.85 GiB MAMBA 而启动失败 | `.memory` 记录 + T004 FINDINGS | 生产（限制已知） | 同款 `_split_hicache_size`（`hybrid_pool_assembler.py:239`） | 等价（同一限制） |
| B2 | host 可分配代（epoch）+ 唤醒仲裁 | `_bump_host_allocability`、`_host_allocability_generations`、wake arbiter、`[HAE]` 遥测 | `37cdf219e`、`ebaf9e5c4`、`a3d1dd631`、`82ab23560`；`test_hicache_host_allocability.py` | 生产 | 无（0 命中） | **缺失** |
| B3 | Host 锁与 LRU 驱逐正确性 | 参数化 `test_load_back_rematch_host_eviction_before_host_lock_falls_back`（17 passed/2 skipped） | 测试 + plan | 生产 | `inc_lock_ref/dec_lock_ref` 存在，语义未逐条比对 | 部分 |
| B4 | 投机 staging 不驱逐（M2） | RH 规划里 speculative 只吃 headroom，`alloc` 失败即静默拒绝 | `ready_horizon.py` 头注 | 生产（flag 关闭时无此路径） | `prefetch_capacity_limit`/`prefetch_rate_limited` 限流，非"不驱逐" | 部分 |

### C. L2/L3 持久化与恢复

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| C1 | **Extent L3 后端**（单稀疏文件 + RAM 索引 + pwritev/preadv 合并 + 代验证 + WAL） | `storage/extent/{extent_allocator.py 202 行, extent_file_backend.py 597 行→后续再增}`；`--hicache-storage-backend extent` | `e281df363`、`036fb8945`、`63f7eff61`、`036ce197a`、`b26bd39ec`；`test_extent_file_backend_unit.py` 811 行 | 生产 | 无 `extent` 后端（有 `file/mmap/umbp/...`） | **缺失** |
| C2 | write_back 链补写（anchor→leaf 连续覆盖） | `4f0632446`：`unified_tree_core.py` + `unified_radix_cache.py` 补链写 L3 | 提交 + `3090-dev/test_*` | 生产（write_back 策略下） | `write_through` 状态机 + `publish_node_ids`，无"补链"概念 | 部分 |
| C3 | 持久化不阻塞 ACK drain（避免递归/HOL） | `2bdfcb532` 延迟 L3 持久化、`cd6c7d4c2` demote 后再 flush | 提交 + `test_drain_reentry.py` 424 行 | 生产 | 未见对应机制 | 缺失 |
| C4 | 删除节点跳过 stale persist | `caeadcf49` | 提交 | 生产 | 未见 | 缺失 |
| C5 | per-anchor 持久性版本 + no-material 快进 | `183001f86`、`0f8542c1a`、`0f8...` | 提交 + 回归锁测试 | 生产 | `storage_existence_cache` 用 beliefs/heal（199 行），非 anchor+version | 部分 |
| C6 | WAL checkpoint 水位与持久性检查点 | `036ce197a`（extent 内） | 提交 | 生产 | 不适用（无 WAL） | 缺失（随 C1） |

### D. 异步 IO、预取与传输

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| D1 | GIL-free native 写路径 + 槽回收迁移活跃引用 + 单键 set 计量 | `b26bd39ec`、`63f7eff61`、`036fb8945` | 提交 | 生产（extent 内） | 不适用 | 缺失（随 C1） |
| D2 | 预取合约关联（attempt_id/issue_origin）+ 就绪辅助完成 join | `7831d45dd`、`6881529ac`、`c4ad01ef5` | 提交 + `test_rh_contract_dependency_free.py` | 生产 | 有 `_prefetch_outcome_stats` 与 operation 记录，无 attempt 身份 | 部分 |
| D3 | 延迟 ACK 对账（frontier 队头阻塞） | `30a5a3698`（P3 deferred-ACK reconciliation） | 提交 + `test_storage_regression.py` | 生产 | `_storage_prefetch_wait`+`ongoing_backup` 部分覆盖，阻塞条件未实测 | 部分 |
| D4 | buffer-only staging（锚点/splice/覆盖检测） | 基线已有，其 961 行 | 源码规模 | 生产 | 我方 2025 行，含 `try_lock_anchor`/`staged_span_covered`/`anchor_lost`/`device_covered` | 我方更全 |

### E. Scheduler admission、容量等待与重试

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| E1 | 恢复状态机 T1–T15（NONE/WAIT_CAPACITY/RESTORING/MISS_RECOMPUTE/READY） | `scheduler_components/hicache_restore.py`（12 次提交），纯函数 `next_restore()` | 源码 + `test_hicache_restore.py` | 生产 | 无状态机；`StoragePrefetchRetries`+`reason` 分类提供粗粒度等价 | 缺失（能力部分覆盖） |
| E2 | 容量不足**有界等待**（WAIT_CAPACITY，T13 `hicache_restore_wait_timeout_s` 超时转重算，T11/T12 唤醒） | `hicache_restore.py` + `server_args.hicache_restore_wait_timeout_s` | 源码 | 生产 | 容量不足**立即**重算 + arm 有界重试（`hicache_storage_prefetch_retry_poll_interval/max_attempts`） | 语义不同（策略差异） |
| E3 | 真实 miss / 容量 / IO 失败三分类 | `genuine_miss` 粘性、`DENIED_ALLOC` vs `DENIED_SHORT`、`REVOKED_*` | 源码 | 生产 | `host_capacity`/`below_threshold`/`deadline`/`storage_transfer`/`unservable_prefix`/`device_covered` 分类 | 部分 |
| E4 | too_short 抖动闸门 | `7155cd897` | 提交 | 生产 | `declined_too_short`（不 arm 重试，见注释"polling here would…"） | 等价 |
| E5 | 正命中跨容量等待保留 | `93e6f5487` | 提交 + 3 个测试文件 | 生产 | 容量失败即丢弃本次命中，靠重试重新查询 | 部分 |
| E6 | Ready Horizon 预准入排队预取 | `ready_horizon.py`（5 次提交）+ `--enable-ready-horizon`（**默认 False**）、speculative fraction 0.6、frontier N、负缓存因果 epoch | 源码 + `control/queued-prefetch-host-ready/` | 生产可用但**默认关闭** | 无 | 缺失 |

### F. 并发、资源所有权与生命周期

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| F1 | 取消/异常/迟到 ACK 的所有权 | 取消即释放 staging；迟到 ACK 不得二次释放；`_storage_prefetch_terminal/completion` 一次性记录 | 提交 + `test_hicache_restore*.py` | 生产 | 有 `_retire_ongoing_prefetch`/`prefetch_loaded_tokens_by_reqid`；E050 实测 `wait_complete` 挂起说明仍存在死等路径 | 部分（我方缺陷待修） |
| F2 | 逐请求 attempt/代身份 | `hicache_prefetch_active_attempt_id`、generation | 源码 | 生产 | 无 | 缺失（影响遥测，不影响正确性） |

### G. 性能优化、遥测与实验工具

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| G1 | per-request 有界遥测（`hicache_request.v1`：storage_query/prefetch_to_ready/ready_to_admission/admission_to_prefill_finish，complete/partial/invalid） | `observability/hicache_telemetry.py` + `output_streamer` + `tokenizer_manager` + `io_struct` | `3b7320a55`、`a207b567f`；873 行测试 | 生产 | 仅服务级 metrics + `cached_tokens_details` | 缺失 |
| G2 | host 分配遥测（`[HAE]` 事件、代/变异序列、reclaim 深度） | `_bump_host_allocability` 等 | `82ab23560` | 生产 | 无 | 缺失（随 B2） |
| G3 | 实验车/驱动（gate2/gate3/p3 车、T004 driver、R16C 分析） | `3090-dev/*`、`control/*/runs/` | 目录 | 实验资产 | 我方 `benchmark/perf_telemetry` + E0xx 脚本 | 不迁移（非生产语义） |
| G4 | 性能结论 | R16C：参数匹配 c8 316.55 tok/s（83-state 67.84/209.95/316.55，45-state 68.06/209.70/314.54），vs 历史 371.2 低 14.7%，归因 `UNKNOWN`；NCU `ERR_NVGPUCTRPERM`、nsys 硬件 trace 不可用 | `PROFILING-AND-REGRESSION-REPORT.md` | — | — | 无可宣称净收益 |

### H. 已知缺陷与失败尝试（兄弟仓库自述）

| # | 事项 | 结论 | 出处 |
|---|---|---|---|
| H1 | `--hicache-size 4` 只给 FULL KV 0.22 GiB；`--hicache-ratio 10` 试图分配 62.85 GiB MAMBA 而启动失败 | 容量模型对"FULL 小 + MAMBA 大"的模型不友好 | T004 FINDINGS |
| H2 | extent 首版 64 MiB 几何被拒（FULL 页写失败） | 已改为 150,994,944 B/extent + 32 GiB 文件 | `.memory/hicache-l3-extent-storage/ledger.md` |
| H3 | `_simulate_backup_tree()` 合成 host 张量不归分配器所有 | 测试无效，已删除；拒绝放松 TreeCore 不变量 | T004 FINDINGS |
| H4 | 容量释放/回填链始终未观察到（`capacity_release_observed=false`）；H2D→GPU 逐请求 join、物理盘 vs page-cache 归因缺失 | 未验证，不作结论 | T004 FINDINGS / plan |
| H5 | `0005b61e6399` 的裁剪保护在 HEAD 失去调用点 | 保护以"全量或拒绝"形式存在，函数成死代码 | 我方核对 |

## 3. 与 E050 实测的交叉验证

| 现象（我方 E050） | 兄弟仓库对应证据 | 关系 |
|---|---|---|
| `page_size=1` 文件后端：502,006 文件 / 5.75 GB，8K 恢复有效带宽 ≈25 MB/s | NVMe 16 KiB 小文件：读 41,323 files/s（677 MB/s）、写 630 files/s（10 MB/s）；其 L3 目录 484,747 文件 / 20G；extent 后端设计目标即"20K 页 ≈63K open/read/close" | `SOURCE FACT`；同型瓶颈，extent 正是针对它 |
| `page_size=256`：47 文件 / 123 MB，8K 恢复 TTFT 474 ms（重算 8026 ms） | 无对应配置数据 | 我方独有结论；与 extent 方案互补而非互斥 |
| `wait_complete` 缺页挂起（4 分钟无响应） | 兄弟把"hybrid 全部或全无失败结果"作为 `discarded` 发布，调度器映射为 `DENIED_SHORT`+重算（`67b39b5ac`），并补齐辅助完成计数（`6881529ac`） | `INFERENCE`：我方 `_can_terminate_prefetch` 对 `wait_complete` 恒 False 且不 arm deadline，缺页/无 ACK 即无限等待；需 E051 实测确认 |
| 有效前缀 = 严格小于尾部的 256 整数倍 | 兄弟以 TRAILING_PAGES 尾部键对齐 sidecar | `INFERENCE`：同一约束的两种表述 |

## 4. 对 LFM3 的候选清单（不构成移植决定）

**P0 — 与本轮 P2.2 验收项直接对应，且我方已确认缺口**
1. 混合前沿**不裁剪**语义（兄弟最终实现：全量分配或拒绝；非其死代码函数）。
2. "丢弃的混合就绪必须转重算"的显式判决 + 逐请求完成记录。
3. `wait_complete` 的终止/超时兜底与迟到 ACK 所有权（对应 E050 挂起）。
4. 容量不足 / 真实 miss / IO 失败三分类的有界重试（我方已有 reason 框架，补语义即可）。

**P1 — 长上下文与高频短增量 Extend 直接受益**
5. Extent L3 后端（消除小文件 syscall 瓶颈；与我方 page=256 结论互补，需在 SM70/本机盘上实测）。
6. per-request 恢复遥测（恢复量、重算量、等待归因）——E050 已证明现有 metrics 不足以拆分阶段。

**P2 — 多会话/排队场景，需策略验证**
7. Ready Horizon 预准入预取（兄弟默认关闭、无吞吐证据；我方需自证 LFM3 收益）。
8. host 可分配代唤醒与 `[HAE]` 遥测（消除"忙等重试"与不可解释等待）。
9. 恢复状态机（与 P0 的 4 项重叠；若移植 P0，可不引入全状态机）。

**P3 — 不建议**
10. write_back 补链 / per-anchor 持久性版本 / WAL checkpoint：依赖 extent 与自有持久性模型，
    与 v0.5.23 `write_through` + `storage_existence_cache` beliefs 架构冲突。

**明确不迁移**：实验车与驱动、qwen3_coder 检测器等缓存无关项、RH 之外的调度器改写。

## 5. 未决与限制

- 兄弟仓库所有"生产"判定来自其自身 control 记录，本轮未复跑其基准；其 `frontier.md` 仍有
  `ACTIVE` 项（R16C 归因、T001 L3 生命周期）。
- 我方"缺失/部分"判定基于符号与代码路径核对，未做行为级 A/B（本轮不允许改生产代码）。
- 兄弟仓库 1 个既有 dirty 文件未纳入审计。
- 绝对性能只在各自硬件上成立（兄弟 RTX 3090 / NVMe；我方 V100-16GB / SATA），
  跨硬件数字不可直接比较。

## 6. 溯源附录

### 6.1 关键提交（按功能）

| 功能 | 提交 |
|---|---|
| 混合前沿裁剪保护 | `5b61e6399`（退化于 HEAD，见 §2-A3） |
| 丢弃就绪拒绝 / 辅助完成 join | `67b39b5ac`、`6881529ac` |
| 正命中跨容量等待 | `93e6f5487` |
| 延迟 ACK 对账（frontier HOL） | `30a5a3698` |
| 持久化脱离 ACK drain / demote 后 flush / stale persist | `2bdfcb532`、`cd6c7d4c2`、`caeadcf49` |
| write_back 链补写（T38c） | `4f0632446` |
| 空洞备份守卫 | `cc718c3c2` |
| per-anchor 持久性版本 / no-material 快进 | `183001f86`、`0f8542c1a` |
| too_short 抖动闸门 | `7155cd897` |
| host 可分配代 / 唤醒 / 遥测 | `37cdf219e`、`ebaf9e5c4`、`a3d1dd631`、`82ab23560` |
| Extent L3 后端 | `e281df363`、`036fb8945`、`63f7eff61`、`036ce197a`、`b26bd39ec`、`792d89a05`、`5afb7a8e6` |
| 恢复重试 + frontier 唤醒 | `f3fffa7f0` |
| Ready Horizon | `3ddc39c22`、`c4ad01ef5`、`7831d45dd`、`d49214e59`；预算唤醒前序 `7caa438f6`、`4f41f5bab` |
| per-request 遥测 | `3b7320a55`、`a207b567f` |
| 轨迹可追溯（无 L3 的 radix 身份） | `e80f144dd` |

### 6.2 兄弟仓库自引测试口径

- READY/MAMBA/遥测/身份/预算选择：47 passed（2026-10-17）
- FULL+MAMBA 取消/共享前缀/回填/驱逐：11 passed
- write-back/drop 参数化锁存续：17 passed, 2 skipped
- 预算/遥测后续：41 passed；`getattr` 缺陷修复后 116 passed
- extent 后端单测文件 811 行（`test_extent_file_backend_unit.py`）

### 6.3 我方对照依据（本仓库）

`python/sglang/srt/mem_cache/{storage_prefetch.py,buffer_mode/pipeline.py,buffer_mode/storage_existence_cache.py,unified_radix_cache.py,hybrid_cache/hybrid_cache_controller.py}`、
`python/sglang/srt/managers/{cache_controller.py,scheduler.py}`；
E050 数据：`experiments/E050-lfm3-hicache-three-tier/results.md`。
