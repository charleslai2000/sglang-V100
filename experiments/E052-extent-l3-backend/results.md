# E052 results — Extent L3 backend

Branch `v100-harness`, base `c3212da455`. GPU: Tesla V100-SXM2-16GB (SM70),
single card, LFM3-1.7B FP16, Triton attention, `wait_complete`, `write_through`,
`page_first_direct`, `--hicache-host-memory-mode cache` unless stated.

## 1. Port + integration status

| Item | Result |
|---|---|
| `storage/extent/` (allocator, metadata, backend) | ported, 5,350 lines |
| `rust/l3wal` workspace member + native writer | ported, 1,499 lines, builds on pinned 1.92 |
| Factory / CLI / env wiring | `--hicache-storage-backend extent`, 6 env vars |
| v1 zero-copy read path (`batch_get_v1`) | enabled for `extent` (one merged read per page, no dummy pages) |
| Group commit (FULL+SWA+Conv per backup) | ported as-is; `FRAME_GROUP_COMMIT` / `GroupCommitPayload` |
| Reclaim → `StorageExistenceCache` | new `discard()` + `_drain_reclaimed_storage_keys()` |
| O_DIRECT | removed; warns and falls back to buffered |
| `t38c` durable-probe | not ported (`extent_trace.py` inert shim, call sites kept) |

## 2. Tests

| Suite | Result |
|---|---|
| `test_extent_g1_allocator_unit.py` | 17 passed |
| `test_extent_g1_metadata_unit.py` | 42 passed (WAL tail, generation invalidation, crash recovery) |
| `test_extent_file_backend_unit.py` | 124 passed, 1 skipped |
| `native/test_l3_native_wal_equivalence.py` + `…drop_reuse_durability.py` | 20 passed (byte-identical WAL, lease/shutdown barrier, chunked arm parity after restart) |
| `test_hicache_restore_reliability.py` (E051 + 3 new reclaim cases) | 26 passed |
| `test_storage_prefetch_lifecycle.py` / `test_buffer_mode_sidecar.py` | 16 / 6 passed |
| `unified_radix_cache/test_unified_radix_cache_storage.py` | 210 passed, 53 skipped |

Regression note: running those suites with `CUDA_VISIBLE_DEVICES=999` produces
233 spurious `No accelerator` failures; they need the real GPU (and no live
server, otherwise 2 OOM failures).

## 3. Correctness on the real serving path

- **Matrix** (`e052_matrix_extent.json`): 45/45 cases byte-identical token ids vs
  the no-HiCache E050 baseline, max logprob deviation **7.55e-3**, 17 cases served
  from storage after `/flush_cache`.
- **Repeat** (`e052_repeat_extent.json`): 5/5 rounds `storage=1024`, ids stable.
- **Missing pages** (`e052_missing_extent.json`): 0.27 s, full recompute, no hang.
- **Cold restart** (`e052_restart_cold.json`): after killing the server and
  restarting on the same extent dir (manifest + WAL replay):

  | prefix | device | storage | ids match |
  |---|---|---|---|
  | 1024 | 0 | 1024 | yes |
  | 4096 | 1024 | 3072 | yes |
  | 8192 | 4096 | 4096 | yes |

  TTFT 0.23 / 0.29 / 0.36 s. Conv (Mamba) checkpoints are part of the restored
  group — ids match only if the SSM state was recovered too.
- **Manifest-absent** (`e052_recovery_manifest_absent.log`): deleting the manifest
  while WAL/data remain fails closed —
  `RecoveryError: manifest is absent while prior metadata/WAL exists
  (class=manifest-absent)` — instead of silently starting cold on stale extents.
- **Reclaim under `buffer_only`, `file_size=200M` / 3×64M ring**: 5 reclaim
  events, each `HiCache extent reclaim: 22 keys, 22 existence beliefs invalidated`;
  all 8 distinct 4K prefixes still restored (`storage=4096`) with identical ids,
  i.e. the dropped beliefs cause an idempotent rewrite rather than a hole.
  (In `cache` mode no existence beliefs exist, so the drain is not wired there.)

## 4. Performance A/B — restore after `/flush_cache`

Serving config identical across arms; only the backend differs. `syscr`/`rchar`/
`cpu_ms` are per-phase deltas of all `sglang::*` processes.

### page = 256 (production config)

| prefix | backend | TTFT ms | syscr | rchar | cpu ms |
|---|---|---|---|---|---|
| 1024 | file | 102.8 | — | 15.5 MB | — |
| 1024 | extent py | 96.6 | 136 | 15.5 MB | 360 |
| 1024 | extent native | 99.7 | 137 | 15.5 MB | 360 |
| 4096 | file | 238.3 | — | 53.3 MB | — |
| 4096 | extent py | 205.6 | 140 | 53.3 MB | 510 |
| 4096 | extent native | 207.5 | 141 | 53.3 MB | 500 |
| 8192 | file | 418.0 | 166 | 103.7 MB | 830 |
| 8192 | extent py | **348.6** | 145 | 103.7 MB | **690** |
| 8192 | extent native | 353.8 | 145 | 103.7 MB | 710 |

→ page 256: extent is **1.20×** faster TTFT and **1.17×** cheaper in CPU than
file; native ≈ Python (reads identical, as expected).

### page = 1 (the case E050 flagged)

| prefix | backend | TTFT ms | syscr | rchar | cpu ms |
|---|---|---|---|---|---|
| 1024 | file | 1141.8 | 1413 | 15.5 MB | 1600 |
| 1024 | extent | **361.6** | 143 | 15.5 MB | 650 |
| 4096 | file | 4015.2 | 4489 | 53.3 MB | 4730 |
| 4096 | extent | **964.7** | 171 | 53.3 MB | 1360 |
| 8192 | file | 7625.7 | 8593 | 103.6 MB | 9340 |
| 8192 | extent | **1810.9** | **207** | 103.6 MB | **2190** |

→ page 1: **4.21×** faster 8K restore, **41.5×** fewer read syscalls, **4.26×**
less host CPU, same bytes read (`rchar` identical ⇒ read amplification ≡ 1.00
for both; the difference is syscall/open overhead, not extra data).

### Serving throughput

| load | cold (write-through) | restore (prefix already in tier) |
|---|---|---|
| c8 | 297–310 tok/s, TTFT p50 0.79–1.03 s | 327 tok/s, cached p50 2304 |
| c32 | 190–337 tok/s, TTFT p50 2.48–3.13 s | 423 tok/s, cached p50 3072 |

No stalls, no failed requests, no `prefetch_stalled`, no anchor loss in either
load level.

### Write path (Rust native vs Python)

Cold 8K prefill (write-through to L3): native 58.6 MiB written / 5970 ms CPU /
5823 ms wall; Python 29.3 MiB / 5970 ms / 5816 ms. **No measurable end-to-end
gain** — the write volume is small relative to prefill compute (≈5.8 s) and the
writes are buffered. The native writer's value here is verified by the
byte-equivalence and lease-barrier tests, not by TTFT.

## 5. Honest gaps

- `read_bytes` and `disk_sectors_read` are 0 in every arm: the harness runs as
  uid 1000, `/proc/sys/vm/drop_caches` is not writable, so all reads are
  page-cache hits. Physical-disk throughput could not be separated from page
  cache; `rchar`/`syscr` are the authoritative numbers and they are identical
  across arms for bytes, differing only in syscall count.
- No measured native-writer win at this model size (see §4).
- The `page=1` file arm re-confirms E050 (8037 ms) → extent is the only viable
  small-page L3 configuration here.
- Anchor-invalidation window (E051) still not externally reproducible; unaffected
  by this port (all lifecycle tests pass).

## 6. Recommended production config (LFM3, V100-16GB)

```
--enable-hierarchical-cache --hicache-host-memory-mode cache
--hicache-io-backend direct --hicache-mem-layout page_first_direct
--hicache-write-policy write_through
--hicache-storage-backend extent
--hicache-storage-prefetch-policy wait_complete
--hicache-storage-backend-extra-config '{"storage_dir": "<dir>",
   "file_size": "40G", "extent_size": "64M",
   "prefetch_timeout_base": 20.0, "prefetch_timeout_per_ki_token": 5.0}'
--page-size 256
```

Keep `write_through` (E051: `write_back` loses flushed host-dirty pages).
`buffer_only` is only needed if the host pool must stay small; that is also the
mode where reclaim invalidates existence beliefs.
