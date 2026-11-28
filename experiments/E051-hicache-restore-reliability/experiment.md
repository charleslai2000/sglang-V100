# E051 — HiCache 恢复正确性与生命周期（P2.2-A）

## 目标

以 R001 v2 审计为依据，把兄弟仓库 `../sglang-hicache@c1feed4d2` 的**恢复契约语义**
（而非其状态机框架）移植到 `v100-harness`：

1. **安全暂存分配**：容量不足时按策略裁剪，禁止缩短已校验的 `TRAILING_PAGES` 前沿；
2. **异步恢复所有权**：保存 anchor 身份（链式页哈希），提交前校验，失效安全撤销；
3. **恢复进展保证**：区分真实 miss / 容量不足 / IO 失败；`wait_complete` 不再无超时挂起；
   ACK 消费与资源释放恰好一次。

不引入完整 Restore 状态机、Ready Horizon、Extent 后端或早期 GPU anchor pin。
Host 重复副本回收留 P2.2-B。

## 冻结配置

与 E050 同基线，唯一差别是预取策略改为 **`wait_complete`**（E050 因挂起而未采用）：

| 项 | 值 |
|---|---|
| 模型 | `/var/tmp/lfm3`（LFM3-1.7B，13 conv / 6 full_attn / 5 sliding_attn，kv_heads=4，head_dim=128） |
| 精度/后端 | FP16，`--attention-backend triton`，`--sampling-backend pytorch` |
| 长度/并发 | `--context-length 8448`，`--max-running-requests 32`，`--cuda-graph-max-bs-decode 32`，`--mem-fraction-static 0.60` |
| Radix | `--mamba-radix-cache-strategy extra_buffer` |
| HiCache | `cache` 模式，`--hicache-io-backend direct`，`--hicache-mem-layout page_first_direct`，`--hicache-storage-backend file`，`write_through` |
| 页大小 | `--page-size 256` |
| 预取 | **`--hicache-storage-prefetch-policy wait_complete`**，`prefetch_timeout_base=20.0`，`per_ki_token=5.0` |
| 容量压力 | 追加 `--hicache-host-memory-fraction 0.05`（0.8 GiB）/ `0.02`（0.3 GiB） |
| 观测 | `--enable-metrics` |

## 移植范围（只这三组，其余不引入）

| # | 兄弟语义 | 本方落地 | 有意偏离 |
|---|---|---|---|
| A | `_staging_crop_length`：页对齐、`≥max(page, alloc_len/8)`、TRAILING 前沿否决 | `hybrid_cache_controller.staging_crop_length` + `storage_hit_crop_is_safe` | 额外把 `min_tokens`（`prefetch_threshold`）作为下限；额外否决 `sidecar_hash_values` 已裁剪头与 KV 派生池（其键即 KV 页链） |
| B | 容量不足 → 一次 ACK drain 后重算 | **有界重排队**（`_defer_staging_attempt`，预算 2 次） | 不做嵌套 drain：外层 drain 的队列计数已固定，嵌套消费备写 ACK 会让外层 `q.get()` 永阻塞 |
| C | anchor last-page hash 在提交前校验，失效即终止 | `_OngoingPrefetch.anchor_last_hash` + `_prefetch_anchor_is_valid` + `_discard_prefetch_stale_anchor`；`node_exists` 进 tree-core | 未哈希 anchor（发起点无链）降为"存在性 + id 不复用"判定，而非无条件拒绝；本方树允许无哈希节点，硬拒绝会丢弃此类全部预取 |
| D | `wait_complete` 由 ACK drain 终结 | 追加**停顿检测**（进度水位 + 预算），终态与策略超时分开 | 兄弟无此检测；本方 E050 观察到该策略可无限期阻塞 admission |

## 实际缺陷根因（三条，均有对照证据）

1. **暂存分配无裁剪**：`allocate_storage_hit` 只能整笔成功或整笔失败，`available` 不足即整段撤销。
   E050 观测的 `host_capacity → 重算` 即此路径；兄弟在 r60 C-8 上量到同一损失（525 次 demand
   staging，请求 3.8–13.8K vs 空闲 0.2–0.9K）。
2. **异步预取持有裸 NodeId**：anchor 被驱逐后 `dec_host_lock_ref` 会对死 id 抛 `KeyError`；
   更坏的情况是 id 被复用后把 L3 读回的数据嫁接到无关前缀。原实现只靠 host lock 保护，
   而 device-only anchor 在 `write_through` 下没有 host slot 可锁。
3. **`wait_complete` 没有活性条件**：`_can_terminate_prefetch` 对 `wait_complete` 恒返回
   `False`，且 `_start_storage_prefetch_wait` 只为 `timeout` 策略计时。任何"完成 ACK 未到达"
   的情形（ACK 丢弃、IO 线程阻塞在缺页读、后台异常被 worker 吞掉）都会永久卡住 admission——
   这正是 E050 用 `timeout` 规避的挂起。

## 验证方法

- 单元：新增 `test/registered/unit/mem_cache/test_hicache_restore_reliability.py`（23 例：
  裁剪策略 7、重排队 2、anchor 身份 6、失效撤销 2、`wait_complete` 活性 3，余为边界）；
  逐文件回归见 `results.md`。
- GPU：`e051_verify.py`（matrix / missing / repeat / cancel / pressure / stress / capacity）
  + `e051_timing.py`；数据在同目录 `*.json`。

## 复现

```bash
# 服务（wait_complete）
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=python .venv/bin/python -m sglang.launch_server \
  --model-path /var/tmp/lfm3 --trust-remote-code --dtype float16 --attention-backend triton \
  --sampling-backend pytorch --context-length 8448 --mem-fraction-static 0.60 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --mamba-radix-cache-strategy extra_buffer --enable-hierarchical-cache \
  --hicache-host-memory-mode cache --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"file_storage_path":"<dir>","prefetch_timeout_base":20.0,"prefetch_timeout_per_ki_token":5.0}' \
  --page-size 256 --enable-metrics --port 30183 --host 127.0.0.1
# 场景
python3 e051_verify.py matrix  --port 30183 --out e051_matrix.json
python3 e051_verify.py missing --port 30183
python3 e051_verify.py repeat  --port 30183
python3 e051_verify.py cancel  --port 30183
python3 e051_verify.py capacity --port 30183 --clients 8   # 需小 host 池
python3 e051_timing.py
```
