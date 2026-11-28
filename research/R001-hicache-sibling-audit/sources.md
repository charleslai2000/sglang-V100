# 源（仅列实际使用者）

## 源仓库（只读）

- `../sglang-hicache`，分支 `3090-24g-hicache`，审计基座 **`origin/3090-24g-hicache @ c1feed4d2`（2026-11-28）**，官方基线 tag `v0.5.6.post2`（`5c8bd8b51`）。
  - 注：v1 审计用的是当时的本地检出 `592a9ef99`（2026-10-17）；远端已推进且**历史被重写**（非 fast-forward），旧 hash 在当前分支不可达，映射见 `findings.md` §6.4。本轮仅 `git fetch origin 3090-24g-hicache` 更新引用，工作树/分支/既有 dirty 文件未动。
- 源码：`python/sglang/srt/mem_cache/{unified_radix_cache.py,unified_cache/*,hybrid_cache/*,buffer_mode/*,pool_host/*,storage/extent/*}`、
  `python/sglang/srt/managers/{scheduler.py,cache_controller.py,scheduler_components/{hicache_restore.py,ready_horizon.py}}`、
  `python/sglang/srt/observability/{hicache_telemetry.py,metrics_collector.py}`、
  `python/sglang/srt/{environ.py,server_args.py,io_struct.py}`。
- 记录：`control/frontier.md`、`control/queued-prefetch-host-ready/{goal,plan}.md` 与
  `runs/T004-FALCON-20261016/FINDINGS.md`、`control/falcon-h1-3090-hicache-perf/{goal-r16c.md,plan-r16c.md,runs/R16C-20261012T0000Z/*}`、
  `control/falcon-h1-3090-online-hicache/report.md`、
  `.memory/hicache-l3-extent-storage/ledger.md`、`.memory/phase2-hicache-design/`。
- 记录（v2 新增）：`control/gemma4-humming-w4a16/runs/{T012-C8-AB-AND-TIERS.md,T010-LONG-CONTEXT-TIERS-20261126.md,T010-GPU-HARDWARE-PROFILING-20261127.md}`、`control/frontier.md`（write_back/flush 边界、L3 恢复契约、prefill graph 关闭、FA2 opt-in）。
- 提交：见 `findings.md` §6.1（均在 `c1feed4d2` 可 `git log -1` 复核）；v1→v2 hash 映射见 §6.4。

## 目标仓库

- `v100-harness@5a9061f0c3`；源码同上路径族（v0.5.23 语义）；
  证据 `experiments/E050-lfm3-hicache-three-tier/results.md`。

## 限制

- 源仓库 1 个既有 dirty 文件未纳入；本轮未切换分支、未运行其基准。
