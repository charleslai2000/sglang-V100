# 源（仅列实际使用者）

## 源仓库（只读）

- `../sglang-hicache`，分支 `3090-24g-hicache`，HEAD `592a9ef99`，官方基线 tag `v0.5.6.post2`（`5c8bd8b51`）。
- 源码：`python/sglang/srt/mem_cache/{unified_radix_cache.py,unified_cache/*,hybrid_cache/*,buffer_mode/*,pool_host/*,storage/extent/*}`、
  `python/sglang/srt/managers/{scheduler.py,cache_controller.py,scheduler_components/{hicache_restore.py,ready_horizon.py}}`、
  `python/sglang/srt/observability/{hicache_telemetry.py,metrics_collector.py}`、
  `python/sglang/srt/{environ.py,server_args.py,io_struct.py}`。
- 记录：`control/frontier.md`、`control/queued-prefetch-host-ready/{goal,plan}.md` 与
  `runs/T004-FALCON-20261016/FINDINGS.md`、`control/falcon-h1-3090-hicache-perf/{goal-r16c.md,plan-r16c.md,runs/R16C-20261012T0000Z/*}`、
  `control/falcon-h1-3090-online-hicache/report.md`、
  `.memory/hicache-l3-extent-storage/ledger.md`、`.memory/phase2-hicache-design/`。
- 提交：见 `findings.md` §6.1（均在上述分支可 `git log -1` 复核）。

## 目标仓库

- `v100-harness@5a9061f0c3`；源码同上路径族（v0.5.23 语义）；
  证据 `experiments/E050-lfm3-hicache-three-tier/results.md`。

## 限制

- 源仓库 1 个既有 dirty 文件未纳入；本轮未切换分支、未运行其基准。
