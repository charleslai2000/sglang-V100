# E052 — Extent L3 backend for LFM3 FULL/SWA/Conv joint persistence

## Question

`E050` showed the file L3 backend degrades badly at small page sizes (`page=1`:
8K restore 8037 ms vs 474 ms recompute) because it issues one `open`+`read` per
page. The sibling HiCache tree solved this with an extent-allocated, range-merged
store (`storage/extent/` + `rust/l3wal`). Does porting it give a production path
on V100/SM70 for LFM3's three-pool backup (FULL KV / SWA KV / Conv checkpoints)
without regressing E050/E051 semantics?

## Baseline

- Tree: `v100-harness @ c3212da455` (E051 done).
- Source: `sglang-hicache @ origin/3090-24g-hicache = c1feed4d2` (read-only).
- Model: `LiquidAI/LFM3-1.7B` at `/var/tmp/lfm3`, FP16, Triton attention,
  page 256 (and page 1 for the syscall-bound case).
- Reference: E050 three-tier run (same launch flags, file backend) and the
  no-HiCache baseline matrix (`/var/tmp/e050_matrix_baseline.json`).

## Ported scope

1. `python/sglang/srt/mem_cache/storage/extent/` — `extent_allocator.py` (589),
   `extent_metadata.py` (1775), `extent_file_backend.py` (2986).
2. `rust/l3wal/` (1499 lines) as a workspace member with
   `[package.metadata.sglang] python-module`, loaded by the existing
   `rust_extensions/loader.py` (no change to the loader).
3. Wiring: `StorageBackendFactory` (`"extent"`), `--hicache-storage-backend extent`,
   6 `SGLANG_HICACHE_EXTENT_BACKEND_*` env vars, and `"extent"` added to the
   `cache_controller` v1 zero-copy whitelist so reads land directly in the host
   pool pages (`batch_get_v1`) instead of dummy-page + scatter.
4. Reclaim → existence beliefs: `StorageExistenceCache.discard()` +
   `UnifiedRadixCache._drain_reclaimed_storage_keys()` (buffer_only mode).
5. Cost-of-port adaptation: `extent_trace.py` inert shim (the sibling's `t38c`
   probe is not ported), O_DIRECT removed (buffered IO only).

Not ported (deliberate): sibling Scheduler changes, Restore state machine,
Ready Horizon, temporary probes, O_DIRECT tier, `mem_cache/trace_context.py`.

## Method

- Unit/parity: sibling tests reused and fixture-adapted (`available_host_memory_bytes`
  mock, `_host_pool_kwargs`), plus 3 new reclaim-invalidation tests.
- End-to-end (`/var/tmp/e052_ab.py`): warm 1K/4K/8K prefixes → `/flush_cache` →
  restore ×3, TTFT/TPOT from the SSE stream, per-phase `/proc/<pid>/{io,stat}`
  deltas over all `sglang::*` children, `/proc/diskstats` for the data disk.
- A/B arms: `file` / `extent` (Python reference) / `extent` +
  `SGLANG_L3_NATIVE_WRITE_PATH=1` (Rust native writer), each at page 256 and 1.
- Extra: cold restart against a persisted extent dir (manifest + WAL replay),
  manifest-absent recovery, ring reclamation under `buffer_only`, c8/c32 load.

## Success criteria

- Sibling extent/WAL/native tests pass unchanged in intent (abnormal exit, WAL
  tail corruption, space reclaim, generation invalidation, incomplete group).
- LFM3 restore correctness identical to the file backend (ids + logprob).
- At `page=1`, extent restore must beat both the file backend and recompute.
- Reclaim must invalidate existence beliefs (no stale-positive write suppression).
