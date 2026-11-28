# 兄弟仓库 HiCache 成果审计（对 v100-harness@5a9061f0c3）— v2

审计基座：**`origin/3090-24g-hicache @ c1feed4d2`（2026-11-28）**，分支 `3090-24g-hicache`。
本仓库 `v100-harness@5a9061f0c3`（官方 v0.5.23 前向合并，E049）。

> **基线更正**：v1 报告（提交 `85d866b13b`）基于当时的本地检出 `592a9ef99`（2026-10-17）。
> 复核发现远端已推进到 `c1feed4d2` 且**历史被重写**（非 fast-forward，同一逻辑提交换了
> hash），v1 引用的提交号在当前分支上已失效，并漏掉 2026-09-19…2026-11-28 约 30 个
> HiCache 相关提交与两轮决定性证据。v2 重做：所有 hash 取自当前分支，映射见 §6.4。
> 本轮仅 `git fetch` 更新远端引用；源仓库工作树、当前分支与其既有 dirty 文件均未变动。
>
> 本轮只读审计，不改生产代码，不跑新基准，不决定移植范围。

## 0. 结论摘要（v2）

1. **兄弟仓库 = 旧基线（v0.5.6.post2）+ 自有 HiCache 二期到现在仍在推进的中期形态**。
   自有 HiCache 路径提交 **70 个**（2026-08-29…2026-11-28，含测试与工具），其中
   2026-11-22…11-26 的 r60/r61 系列是"write_through 下 Host 重复副本回收"的根因修复，
   属于本次 v2 新增的关键内容。`SOURCE FACT`
2. **多数机制 v0.5.23 已有不同实现**（`StoragePrefetchRetries`、buffer staging 2025 行 vs 961、
   `storage_existence_cache` 199 vs 89、`restorable_prefix_pages`、`declined_too_short`）；
   真正缺口集中在：L3 **恢复契约**（anchor 起的最长可持久跨度）、暂存裁剪策略与
   TRAILING_PAGES 保护、容量/miss/IO 三分类有界等待、host 可分配代唤醒、extent 后端、
   per-request 遥测、host 重复副本回收。`SOURCE FACT` + `SYNTHESIS`
3. **v1 关于裁剪保护的结论需更正**：`_storage_hit_crop_is_safe` 在当前分支**不是死代码**，
   它被 `_staging_crop_length()` 调用（`unified_radix_cache.py:274`，调用点 `:7452`）：
   容量不足时按"当前空闲量、页对齐、且裁剪量 ≥ max(page, alloc_len/8)"裁剪，
   **但 TRAILING_PAGES（Mamba/SWA sidecar）前沿一律不裁剪**；不满足则整笔撤销重算。
   这正是 E050 观察到的 `host_capacity → 重算` 的成熟形态。`SOURCE FACT`
4. **性能：现在有可复现的三级收益，但归因必须精确**。同一负载、36.5K 上下文、
   3.5M 工作集 vs 2.32M 两级窗口下：L1+L2(+L3 经 Host 暂存) 相对 L1-only
   **+43.0% TPS / −30.1% wall**；`source="l3"` 计数为 0 是**匹配期权威口径的伪影**，
   L3 实际通过 Host 暂存参与了恢复（723 次 extent read）。T012 的 +12.71%/+8.90%
   来自 weight loader（Humming），不是 HiCache 收益。`SOURCE FACT`

## 1. 审计方法与可追溯性

- 提交集：`git log --author={charles,charleslai,opencode} -- <HiCache 路径族>` 在当前分支
  → **70 个**；热点：`unified_radix_cache.py` 28、`scheduler_components/hicache_restore.py` 12、
  `storage/extent/extent_file_backend.py` 10、`hybrid_cache_controller.py` 8、
  `unified_cache/unified_tree_core.py` 7、`extent_allocator.py` 5、`ready_horizon.py` 5。
- 增量（`592a9ef99..c1feed4d2`，作者自有）中触及 HiCache 路径者 ~30 个，按主题分：
  r61 锚点/重复副本（10）、ack-drain 与延迟写（4）、extent/持久化（6）、
  Gemma4 owner KV（5）、SWA 所有权（1）、遥测/工具（4）。
- 文档：`control/frontier.md`（当前权威状态）、`control/queued-prefetch-host-ready/`、
  `control/falcon-h1-3090-hicache-perf/runs/R16C-*`、`control/gemma4-*/runs/T0*`、
  `.memory/hicache-l3-extent-storage/ledger.md`、`.memory/phase2-hicache-design/`。
- 语义对照三问（机制 / 不变量 / 可观测结果），我方结论逐条给依据；不以函数名或 patch-id 等价。

## 2. 功能差异矩阵（主表）

判定：**等价** / **部分** / **缺失** / **冲突或硬件相关**。
"我方"指 `v100-harness@5a9061f0c3`（v0.5.23 语义）。

### A. FULL/SWA/MAMBA 混合缓存

| # | 能力 | 兄弟仓库（机制/提交/位置） | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| A1 | 联合恢复同前沿 + 尾部键重对齐 | `PoolHitPolicy.TRAILING_PAGES`、`_sync_trailing_keys`、`batch_exists_v2` 各池 min | 代码 + T004 逐笔 query/get/l3_host | 生产 | 同机制（`_sync_trailing_keys`、`restorable_prefix_pages`、`all_or_nothing`） | 等价 |
| A2 | 全部或全无语义 | `all_succeeded` + 结果必须发布 | `8e1edde1d`、`f7d22a0a0` | 生产 | `_finish_storage_prefetch(fulfilled=0)` 释放请求 | 部分 |
| A3 | **暂存裁剪策略 + TRAILING_PAGES 保护** | `_storage_hit_crop_is_safe`（`c399971cc` 引入）现由 `_staging_crop_length()` 使用：页对齐、≥max(page, alloc_len/8)、TRAILING_PAGES 前沿否决；否则 `revoke_pending_prefetch(alloc_failed=True)` | `c399971cc`、`c58cf9d5c`；`test_staging_crop.py`（`test_trailing_pages_frontier_is_never_shortened` 等） | 生产 | `allocate_storage_hit(allow_partial=True, min_tokens=threshold)` 无 TRAILING 保护、无"裁剪需自偿"门槛 | **缺失** |
| A4 | 丢弃就绪不得当 READY | `record/pop_storage_prefetch_completion` + `discarded/rejected → DENIED_SHORT + short_tail_recompute` | `8e1edde1d`、`f7d22a0a0` | 生产 | 丢弃即按 reason 放行重算，无逐请求完成记录 | 部分 |
| A5 | 存储空洞 D→H 备份守卫 | `f0006fea6`：按 device value 存在性决定是否备份 | 提交 + 测试 | 生产 | 无 | 缺失 |
| A6 | **write_through 下 Host 重复副本回收**（r61 根因） | `13d1b3263`（`unified_tree_core.py` +200 行、`full_component.py`、`test_host_duplicate_reclaim.py` 223 行）；`cfcbbcb23` 允许 SWA 独立重复副本生命周期；`2be547bd4` 按需求驱动回收 | 提交 + 测试 + T010 证据 | 生产 | 无（E050 观察到 flush 后 host 仍留最近请求占用） | **缺失** |

### B. 缓存池、容量分配与驱逐

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| B1 | `--hicache-size` 在 KV/MAMBA 间按比例切分 | `_split_hicache_size()`；实测 4 GiB 总额只给 FULL KV 0.22 GiB（MAMBA 3.90 GiB）；ratio=10 超发 62.85 GiB 启动失败 | T004 FINDINGS | 生产（限制已知） | 同款 `_split_hicache_size`（`hybrid_pool_assembler.py:239`） | 等价（同一限制） |
| B2 | host 可分配代 + 唤醒仲裁 + `[HAE]` 遥测 | `_bump_host_allocability`、`_host_allocability_generations`、wake arbiter；`9f5db4d81`（NO_HEADROOM 按释放 epoch 唤醒）、`0099f3863`（DEVICE_QUOTA 等待前向退化）、`452d583f1`（粘性 miss 重臂门）；`a54a6d695` **revert** 了 Fix B 的 DEVICE_QUOTA 超时计数门 | `71374ac3e`、`e00515aef`、`0fa522302`、`2f07f6b6d`、`d442a0886`；`test_hicache_host_allocability.py` | 生产 | 无 | **缺失** |
| B3 | Host 锁与 LRU 驱逐正确性 | lock-window 参数化测试 17 passed/2 skipped | 测试 + plan | 生产 | `inc_lock_ref/dec_lock_ref` 存在，语义未逐条比对 | 部分 |
| B4 | 投机 staging 不驱逐 | RH M2：speculative 只吃 headroom，失败静默拒绝 | `ready_horizon.py` | 生产（flag 关闭时无此路径） | `prefetch_capacity_limit`/`rate_limited` 限流 | 部分 |
| B5 | 备份前收入检查遵循 write-back floor | `_backup_shortfall(host_avail, floor, kv_tokens)`：`alloc_writeback` 上限是 `pool - floor`，用裸 available 会静默丢备份（r61 floor arm 实测） | `c58cf9d5c` | 生产 | 无 | 缺失 |

### C. L2/L3 持久化与恢复

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| C1 | **Extent L3 后端**（单稀疏文件 + RAM 索引 + pwritev/preadv 合并 + 代验证 + WAL + native 写） | `storage/extent/`；`--hicache-storage-backend extent`；env `SGLANG_HICACHE_EXTENT_BACKEND_{STORAGE_DIR,MAX_SIZE,MIN_FREE_SPACE,EXTENT_SIZE,FILE_SIZE,WAL_CHECKPOINT_BYTES,DIRECT_IO}` | `75f109fe9`、`36cc7b00d`、`d5322bec6`、`1ab91c507`、`58be936d4`、`68c03ea07`、`b909dbaab` | 生产 | 无（有 `file/mmap/umbp/...`） | **缺失** |
| C2 | write_back 链补写（anchor→leaf 连续覆盖） | `9108566fc`；`2d5587ef3` 在祖先备份失败后停止 L3 空洞写 | 提交 + 测试 | 生产（write_back 下） | `write_through` 状态机 + `publish_node_ids`，无补链 | 部分 |
| C3 | 持久化不阻塞 ACK drain | `7b0179e04` 延迟 L3 持久化、`83dd83666` demote 后 flush、`55ff0b2f0` 延迟写排空不重入 | 提交 + `test_drain_reentry.py`、`test_hicache_frontier_waiter_lifecycle.py` | 生产 | 未见 | 缺失 |
| C4 | 删除节点跳过 stale persist | `376688b86` | 提交 | 生产 | 未见 | 缺失 |
| C5 | per-anchor 持久性版本 + no-material 快进 | `7d4186945`、`2c422b56d` | 提交 + 回归锁 | 生产 | `storage_existence_cache` beliefs/heal（199 行），非 anchor+version | 部分 |
| C6 | WAL checkpoint 水位/持久性检查点 | `1ab91c507` | 提交 | 生产 | 不适用 | 缺失（随 C1） |
| C7 | 合并 staged write-back 拷贝、owner buffer 一次解析 | `6ff1d81e2`（`pool_host/mha.py`、extent、mamba_component） | 提交 | 生产 | 未见 | 缺失 |
| C8 | **write_back 下 flush→restore 的正确性边界** | `87d453ea2` 记录：`cold→/flush_cache→restore` 在 write_back 下**不恢复**（flush 丢弃未驱逐的 host-dirty 页）→ `write_through` 保持为正确性默认 | `control/frontier.md` | 生产约束 | 我方 E050 全程用 `write_through` | 与本案一致（记录性） |

### D. 异步 IO、预取与传输

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| D1 | GIL-free native 写 + 槽回收迁移活跃引用 + 单键 set 计量 | `58be936d4`、`d5322bec6`、`36cc7b00d` | 提交 | 生产（extent 内） | 不适用 | 缺失（随 C1） |
| D2 | 预取合约关联 + 辅助完成 join | `37b90a97c`、`f7d22a0a0`、`4334a1981` | 提交 + 测试 | 生产 | 有 outcome stats/operation，无 attempt 身份 | 部分 |
| D3 | 延迟 ACK 对账（frontier HOL） | `22eddb20f` | 提交 | 生产 | `_storage_prefetch_wait` + `ongoing_backup` 部分覆盖 | 部分 |
| D4 | buffer-only staging（锚点/splice/覆盖检测） | 基线 961 行 | 源码 | 生产 | 我方 2025 行，含 `try_lock_anchor`/`staged_span_covered`/`anchor_lost`/`device_covered` | 我方更全 |
| D5 | **异步预取锚点 pin + 身份校验 + 容忍失效 victim** | `b795ee421`（pin anchors，容忍 stale victims）、`97839a025`（in-flight 锚点不入设备驱逐）、`663244943`（commit 时校验锚点身份）、`c6907ccf6`（身份 sanity 调用点）、`0b307529e`（忽略失效延迟 victim）、`8d35da1f8`（快照驱逐遍历容忍已删节点）、`7c540c965`（reclaim pin 穿过 tuple 消费者） | `test_prefetch_anchor_identity.py` 309 行 | 生产 | 无 pin/身份校验；我方 `anchor_lost` 只覆盖 buffer 模式 | **缺失** |
| D6 | ACK drain 与有界备份恢复 | `c58cf9d5c`（一次 drain + 重试、`_backup_shortfall` 地板）、`55ff0b2f0`（延迟写排空）、`d442a0886`（削减 per-pass reconcile、丢弃陈旧 L3-only 等待）、`dbf2a6a24`（宿主暂存回收结果关联） | 提交 + 测试 | 生产 | 无 | 缺失 |

### E. Scheduler admission、容量等待与重试

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| E1 | 恢复状态机 T1–T15 | `hicache_restore.py`（12 次提交），纯函数 `next_restore()` | 源码 + 测试 | 生产 | 无状态机；`StoragePrefetchRetries`+reason 分类粗粒度等价 | 缺失（能力部分覆盖） |
| E2 | 容量不足**有界等待** | WAIT_CAPACITY + T13 `hicache_restore_wait_timeout_s` → 重算；T11/T12 唤醒 | 源码 | 生产 | 容量不足**立即**重算 + arm 有界重试 | 语义不同（策略差异） |
| E3 | 真实 miss / 容量 / IO 失败三分类 | `genuine_miss` 粘性、`DENIED_ALLOC` vs `DENIED_SHORT`、`REVOKED_*` | 源码 | 生产 | `host_capacity/below_threshold/deadline/storage_transfer/unservable_prefix/device_covered` | 部分 |
| E4 | too_short 抖动闸门 | `4531f61f4` | 提交 | 生产 | `declined_too_short`（不 arm 重试） | 等价 |
| E5 | 正命中跨容量等待保留 | `87ff0f046` | 提交 + 3 测试文件 | 生产 | 容量失败丢弃本次命中，靠重试重查 | 部分 |
| E6 | Ready Horizon 预准入预取 | `ready_horizon.py`；`--enable-ready-horizon` **默认 False**；spec fraction 0.6 | 源码 + T004 | 可用但默认关闭 | 无 | 缺失 |
| E7 | 暂存裁剪 + 一次 drain 重试的组合策略 | `c58cf9d5c` + `_staging_crop_length` | `test_staging_crop.py` | 生产 | 无 | 缺失（同 A3/B5/D6） |

### F. 并发、资源所有权与生命周期

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| F1 | 取消/异常/迟到 ACK 所有权 | 取消释放 staging；迟到 ACK 不二次释放；`_storage_prefetch_terminal/completion` 一次性记录；`d442a0886` 丢弃陈旧 L3-only 等待 | 提交 + `test_hicache_restore*.py`、`test_hicache_frontier_waiter_lifecycle.py` | 生产 | `_retire_ongoing_prefetch` 等；E050 实测 `wait_complete` 挂起 | 部分（我方缺陷待修） |
| F2 | 逐请求 attempt/代身份 | `hicache_prefetch_active_attempt_id`、generation | 源码 | 生产 | 无 | 缺失（影响遥测） |
| F3 | 请求行 SWA 所有权释放 | `622beddeb`：587.94 TPS 验证；`308e7d6b9` 去掉 SWA 窗口元数据设备标量读 | 提交 + C-8 | 生产 | 未见 | 缺失（与 HiCache 弱相关，属 SWA 所有权） |

### G. 性能、遥测与实验工具

| # | 能力 | 兄弟仓库 | 证据 | 生产状态 | 我方现状 | 判定 |
|---|---|---|---|---|---|---|
| G1 | per-request 有界遥测（`hicache_request.v1`） | `observability/hicache_telemetry.py`；stages: storage_query / prefetch_to_ready / ready_to_admission / admission_to_prefill_finish；outcomes complete/partial/invalid | `af7809a46`、`bddad6135`；873 行测试 | 生产 | 仅服务级 metrics + `cached_tokens_details` | 缺失 |
| G2 | host 分配遥测（`[HAE]`） | 代/变异序列、reclaim 深度 | `2f07f6b6d` | 生产 | 无 | 缺失（随 B2） |
| G3 | 实验车/驱动 | `3090-dev/*`、`control/*/runs/`（T004 driver、T010 比较器、T014 hwprof） | 目录 | 实验资产 | 我方 `benchmark/perf_telemetry` + E0xx | 不迁移 |
| G4 | 三级缓存端到端证据（T010） | 36.5K 上下文、3.5M 工作集：L1-only 78.66 TPS → L1+L2+L3 **112.51 TPS（+43.0%）**，wall 1874.5→1310.6 s（−30.1%）；L2 命中 687,524 tokens；L3 经 Host 暂存参与（723 extent read）；L3 计数为 0 属**匹配期权威口径伪影**；把 L3 提到 64 GiB 仍为 0 且再掉 ~4 TPS → **容量假设被证伪** | `bcfde4898`、`46f7c2038`、`a76206a5f`、`c102ccbc5` | 证据 | 与 E050 的 page 尺寸结论互补 | 参考 |
| G5 | 吞吐线结论 | write_back+wait_complete 全三级验收 800/800、零 sanity 失败，但 499.64–521.97 TPS / 71.4–72.7% duty（<540 目标）；per-pass reconcile 仅占窗口 0.55%，瓶颈在 admission/result-path host 工作；prefill CUDA graph 已关闭（594.23 TPS 未达 600）；FA2 Forward 计划复用为 **opt-in**；T012 的 +12.71%/+8.90% 来自 Humming weight loader，非 HiCache | `87d453ea2`、`e3beda608`、`8b607f675`、`ed6cec929`、`7ed79dab1` | 证据 | — | 参考 |

### H. 已知缺陷与失败尝试

| # | 事项 | 结论 | 出处 |
|---|---|---|---|
| H1 | `--hicache-size 4` 只给 FULL KV 0.22 GiB；`--hicache-ratio 10` 超发 62.85 GiB MAMBA 启动失败 | 容量模型对"FULL 小 + MAMBA 大"不友好 | T004 FINDINGS |
| H2 | extent 首版 64 MiB 几何被拒（FULL 页写失败） | 改为 150,994,944 B/extent + 32 GiB 文件 | `.memory/.../ledger.md` |
| H3 | `_simulate_backup_tree()` 合成 host 张量不归分配器所有 | 测试无效已删除；拒绝放松 TreeCore 不变量 | T004 FINDINGS |
| H4 | Fix B（DEVICE_QUOTA 超时计数门） | 已 **revert**（`a54a6d695`）；改为前向退化界定（`0099f3863`） | 提交 |
| H5 | 容量释放/回填链 | 未观察到（`capacity_release_observed=false`）；H2D→GPU 逐请求 join、物理盘 vs page-cache 归因缺失 | T004 FINDINGS / plan |
| H6 | L3 "零命中" 误判 | 早期以 `source="l3"` 计数下的结论被证实为口径伪影并更正；L3 容量假设亦被证伪 | `a76206a5f`、`46f7c2038` |
| H7 | `flush_cache` 在 write_back 下无法恢复 | `write_through` 保持为正确性默认 | `87d453ea2` |

## 3. 与 E050 实测的交叉验证（关键）

| E050 现象（我方 V100，page=256/write_through） | 兄弟仓库对应 | 关系 |
|---|---|---|
| 容量不足时直接 `host_capacity → 丢弃本次命中并重算`（1K 时首轮 0 命中） | r60 C-8 实测同类损失：525 次 demand staging 请求 3.8–13.8K tokens、池内仅 0.2–0.9K 空闲 → 全部撤销重算；其对策 = 一次 drain 重试 + **自偿型暂存裁剪**（TRAILING_PAGES 除外） | `SOURCE FACT` 对应；我方缺口 A3/B5/D6/E7 |
| `/flush_cache` 后 `hicache_host_used_tokens` 残留最近请求占用（512/1024，10 轮不增长） | r61 根因：write_through 下 Host **重复副本**未被回收（`13d1b1263` 200 行 + 223 行测试）；`2be547bd4` 按需求驱动回收 | `INFERENCE`：我方残留可能就是同型重复副本，需实测确认 |
| `wait_complete` 缺页挂起（4 分钟无响应，随后 watchdog） | `d442a0886` 专门"丢弃陈旧 L3-only 等待"；`c58cf9d5c` 修 ack drain 与有界备份恢复；`8e1edde1d` 把丢弃结果映射为重算 | `INFERENCE`：我方 `_can_terminate_prefetch` 对 `wait_complete` 恒 False 且不 arm deadline → 同类死等；需 E051 复现确认 |
| `page_size=1` 小文件瓶颈：502,006 文件/5.75 GB，8K 恢复 ≈25 MB/s | NVMe 16 KiB 小文件：读 41,323 files/s（677 MB/s）、写 630 files/s（10 MB/s）；其 L3 目录 484,747 文件/20G；extent 后端设计目标即"20K 页 ≈63K syscall" | `SOURCE FACT`；同型瓶颈 |
| `page_size=256`：47 文件/123 MB，8K TTFT 474 ms（重算 8026 ms） | 无对应配置数据 | 我方独有结论，与 extent 互补 |
| 有效前缀 = 严格小于尾部的 256 整数倍 | TRAILING_PAGES 尾部键对齐；T010 记录 `granted = requested − 一个输出 turn` 与 `+1` 的 off-by-one | `INFERENCE`：同一约束的两种表述 |
| L3 恢复收益有限（8K 仅 0.97× 重算） | T010：三级栈 +43% TPS，主要由 L2 DRAM + L3 经 Host 暂存贡献；但**恢复契约要求 root 连续持久链**，被 L2 驱逐且无完整 L3 链的会话完全得不到恢复（6.32M tokens 重算）→ 其下一步即"从 anchor 起的最长可持久跨度恢复" | `SOURCE FACT`；直接指向我方长上下文缺口 |

## 4. 对 LFM3 的候选清单（不构成移植决定）

**P0 — 与 E051 验收项直接对应，且我方已确认缺口**
1. **自偿型暂存裁剪 + TRAILING_PAGES 前沿否决**（兄弟最终态：页对齐、≥max(page, alloc_len/8)、
   sidecar 前沿不裁剪、否则一次性 drain 重试后撤销）。对照我方 `allocate_storage_hit`。
2. **丢弃的混合就绪必须转重算**（逐请求完成记录 + `discarded → DENIED_SHORT`）。
3. **`wait_complete` 的终止/超时兜底与迟到 ACK 所有权**（含"丢弃陈旧 L3-only 等待"）。
4. **容量/miss/IO 三分类的有界重试**（我方已有 reason 框架，补语义与"保留正命中"）。

**P1 — 长上下文与高频短增量 Extend 直接受益**
5. **L3 恢复契约改为从 anchor 起的最长可持久跨度**（T010 的头号损失项；预计把 6.32M 重算
   转回 H2D，约 +13 TPS/1M token；我方 E050 的 8K 场景同构）。
6. **write_through 下 Host 重复副本回收**（对应我方 flush 后残留；r61 已有根因修复与测试）。
7. **per-request 恢复遥测**（恢复量/重算量/等待归因；E050 已证明现有 metrics 不足以拆阶段）。
8. **Extent L3 后端**（消除小文件 syscall 瓶颈，与我方 page=256 结论互补，需本机实测）。

**P2 — 多会话/排队场景，需策略验证**
9. Ready Horizon 预准入预取（兄弟默认关闭；无独立吞吐证据，仅 T004 机制证据）。
10. host 可分配代唤醒 + `[HAE]` 遥测。
11. 恢复状态机（若已做 P0 第 4 项，可不引入全状态机）。

**P3 — 不建议**
12. write_back 补链 / per-anchor 持久性版本 / WAL checkpoint：依赖 extent 与自有持久性模型，
    与 v0.5.23 `write_through` + `storage_existence_cache` beliefs 架构冲突。

**明确不迁移**：实验车/驱动/分析器、Gemma4 owner KV 压缩（模型与 SM86 专属）、
qwen3_coder 检测器、prefill graph（已关闭方向）、FA2 计划复用（opt-in 且属 attention 侧）。

## 5. 未决与限制

- 兄弟仓库"生产"判定来自其自身 control 记录；本轮未复跑其基准，且其 `frontier.md` 仍有多项
  `ACTIVE`（T007/T008 吞吐目标、T010 L3 恢复契约、T003 数值基线）。
- 我方"缺失/部分"判定基于符号与代码路径核对，未做行为级 A/B（本轮不允许改生产代码）。
- 兄弟仓库历史被重写，旧 hash 不再可达；本文所有引用以 `c1feed4d2` 为准（映射见 §6.4）。
- 源仓库 1 个既有 dirty 文件未纳入审计；本轮仅 `git fetch`，未改动其工作树/分支。
- 绝对性能只在各自硬件成立（兄弟 RTX 3090 / NVMe / Gemma4 / Falcon-H1；我方 V100-16GB / SATA / LFM3），
  跨硬件数字不可直接比较。

## 6. 溯源附录

### 6.1 关键提交（当前分支 hash）

| 功能 | 提交 |
|---|---|
| 暂存裁剪保护（引入 → 组合策略） | `c399971cc` → `c58cf9d5c`（`_staging_crop_length`） |
| 丢弃就绪拒绝 / 辅助完成 join | `8e1edde1d`、`f7d22a0a0` |
| 正命中跨容量等待 | `87ff0f046` |
| 延迟 ACK 对账 | `22eddb20f` |
| 持久化脱离 ACK drain / demote 后 flush / stale persist / 延迟写排空 | `7b0179e04`、`83dd83666`、`376688b86`、`55ff0b2f0` |
| write_back 链补写 / 祖先失败停写 | `9108566fc`、`2d5587ef3` |
| 空洞备份守卫 | `f0006fea6` |
| per-anchor 持久性版本 / no-material | `7d4186945`、`2c422b56d` |
| too_short 抖动闸门 / 粘性 miss 重臂 | `4531f61f4`、`452d583f1` |
| host 可分配代 / 唤醒 / 遥测 | `71374ac3e`、`e00515aef`、`0fa522302`、`2f07f6b6d`、`9f5db4d81`、`0099f3863`、`d442a0886`；revert `a54a6d695` |
| Extent L3 后端 | `75f109fe9`、`36cc7b00d`、`d5322bec6`、`1ab91c507`、`58be936d4`、`68c03ea07`、`b909dbaab` |
| ACK drain / 备份恢复 / staged 合并 | `c58cf9d5c`、`dbf2a6a24`、`6ff1d81e2` |
| r61 锚点 pin / 身份 / 重复副本 | `13d1b3263`、`b795ee421`、`97839a025`、`663244943`、`c6907ccf6`、`0b307529e`、`8d35da1f8`、`7c540c965`、`2be547bd4`、`cfcbbcb23` |
| Gemma4 owner KV（模型/SM86 专属） | `a503cf631`、`0a0077904`、`3846c03a2`、`482106797`、`af25ac162`、`1694ac2b5` |
| SWA 所有权 / prefill graph / FA2 | `622beddeb`、`308e7d6b9`、`8b607f675`、`e3beda608`、`ed6cec929` |
| 恢复重试 + frontier 唤醒 | `29d0e1a4e` |
| Ready Horizon | `16172513f`、`4334a1981`、`37b90a97c`、`897e3b5e0` |
| per-request 遥测 | `af7809a46`、`bddad6135` |
| T010/T012/T014 证据 | `bcfde4898`、`46f7c2038`、`a76206a5f`、`c102ccbc5`、`7ed79dab1`、`872bfe1d5`、`c1feed4d2` |

### 6.2 兄弟仓库自引测试口径（截至 `c1feed4d2`）

- READY/MAMBA/遥测/身份/预算：47 passed；FULL+MAMBA 取消/共享前缀/回填/驱逐：11 passed；
  write-back/drop 参数化锁存续：17 passed, 2 skipped；预算/遥测后续：41 passed。
- 新增：`test_staging_crop.py`（**17 例**，含 `test_trailing_pages_frontier_is_never_shortened`、
  `test_crops_to_the_free_prefix`、`test_no_crop_when_the_crop_is_too_small_to_pay_for_itself`、
  `test_drains_once_then_evicts_the_shortfall`、`test_ack_reentry_defers_new_generation_without_double_consumption`）、
  `test_prefetch_anchor_identity.py`（309 行）、`test_host_duplicate_reclaim.py`（223 行）、
  `test_hicache_frontier_waiter_lifecycle.py`、`test_extent_file_backend_unit.py`（**3,609 行**）。
- T010 三级：800/576 请求零失败；T012：800/800 零失败。

### 6.3 我方对照依据（本仓库）

`python/sglang/srt/mem_cache/{storage_prefetch.py,buffer_mode/pipeline.py,buffer_mode/storage_existence_cache.py,unified_radix_cache.py,hybrid_cache/hybrid_cache_controller.py}`、
`python/sglang/srt/managers/{cache_controller.py,scheduler.py}`；
E050 数据 `experiments/E050-lfm3-hicache-three-tier/results.md`。
符号矩阵（我方/兄弟命中文件数）：`storage_prefetch`(3/0)、`staged_span_covered`(2/0)、
`anchor_lost`(3/0)、`restorable_prefix_pages`(3/0)、`unservable_prefix`(1/0)、
`hicache_telemetry`(0/11)、`host_allocability`(0/5)、`enable_ready_horizon`(0/3)。

### 6.4 历史重写映射（v1 引用 → 当前分支）

| v1 hash | 当前 hash | 主题 |
|---|---|---|
| `5b61e6399` | `c399971cc` | 混合预取前沿裁剪保护 |
| `67b39b5ac` | `8e1edde1d` | 拒绝丢弃的混合就绪 |
| `6881529ac` | `f7d22a0a0` | 就绪 Host 辅助完成 join |
| `93e6f5487` | `87ff0f046` | 正命中跨容量等待 |
| `30a5a3698` | `22eddb20f` | 延迟 ACK 对账 |
| `2bdfcb532` | `7b0179e04` | 延迟 L3 持久化出 ACK drain |
| `cd6c7d4c2` | `83dd83666` | demote 后 flush |
| `caeadcf49` | `376688b86` | 删除节点跳过 stale persist |
| `4f0632446` | `9108566fc` | write_back 链补写 |
| `cc718c3c2` | `f0006fea6` | 空洞备份守卫 |
| `183001f86` | `7d4186945` | per-anchor 持久性版本 |
| `0f8542c1a` | `2c422b56d` | durable-ancestor no-material |
| `7155cd897` | `4531f61f4` | too_short 抖动闸门 |
| `37cdf219e` | `71374ac3e` | host 分配代唤醒 |
| `ebaf9e5c4` | `e00515aef` | host alloc epoch 跟进 |
| `a3d1dd631` | `0fa522302` | host alloc epoch 评审 |
| `82ab23560` | `2f07f6b6d` | host alloc 遥测收口 |
| `e281df363` | `75f109fe9` | extent L3 后端 |
| `036fb8945` | `36cc7b00d` | 单键 set 计量 |
| `63f7eff61` | `d5322bec6` | extent 槽回收迁移活跃引用 |
| `036ce197a` | `1ab91c507` | WAL checkpoint |
| `b26bd39ec` | `58be936d4` | GIL-free native 写路径 |
| `f3fffa7f0` | `29d0e1a4e` | 恢复重试 + frontier 唤醒 |
| `c4ad01ef5` | `4334a1981` | RH 当前匹配门 |
| `7831d45dd` | `37b90a97c` | RH 预取合约关联 |
| `3ddc39c22` | `16172513f` | RH 预准入所有权 |
| `3b7320a55` | `af7809a46` | per-request 遥测 |
| `a207b567f` | `bddad6135` | 遥测 WIP 快照 |
| `e80f144dd` | `1d04132da` | 无 L3 时的 radix 身份追踪 |
