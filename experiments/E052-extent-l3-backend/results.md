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

- **Abnormal exit (SIGKILL mid-group-commit)**: four 4K prefixes were in flight
  when the scheduler was killed (`evidence/e052_recovery_after_sigkill.log`).
  Restart on the same dir recovered cleanly, then:

  | prefix | outcome | matches no-HiCache ground truth |
  |---|---|---|
  | 0 (in flight at kill) | `cached=0`, recomputed | yes |
  | 1–3 (committed) | `storage=4096` | yes |

  The un-ACKed group is simply absent — no incomplete restore frontier, and no
  wrong tokens served from a half-written group.
- **Undersized tier** (`file_size=64M`, 1 slot, 8K prefix whose joint backup
  ≈100 MB): writes cannot fit, restores report `storage=0` and recompute,
  outputs identical, no hang, no refusal spam (16.5 s wall for two 8K requests
  ≈ 2× prefill). Degradation is "recompute", never "partial restore".

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
| 8192 | extent native | 1837.5 | 211 | 103.6 MB | 2230 |

(native page-1 arm: 372.1 / 988.7 / 1837.5 ms at 1K/4K/8K, i.e. within noise of
the Python reference — the read path and its syscall count are shared.)

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

## 4b. Round 2 — group commit on the backup chain, restorable-prefix intersection

Two production-path gaps were left by the port and are now closed
(`fe82c69d99`, `a1f4f7c913`).

### 4b.1 One backup = one group

`HybridCacheController._page_backup` now opens one backend group per backup and
writes FULL KV, SWA KV and Conv state under it. KV batches reach the group
through the new `BaseHiCacheController._page_set_for_operation` (zero-copy v1
when the backend stages v1, else the v2 transfer route); sidecars keep
`layout_lease` and `_trailing_chain_groups`. `commit_group` runs only after
every member succeeded; any failure — refused sidecar, failed KV batch, a
commit that returns False, or a raised exception — aborts the group in
`finally`, so an uncommitted backup can never become a restore frontier.

Live-path evidence (`evidence/e052_group_commit_wal_scan.txt`), real LFM3
server, real controller chain, page 256, 45-case run:

```
frame histogram: {'FRAME_SET': 36, 'BATCH_COMMIT': 36, 'GROUP_COMMIT': 17}
group commits: 17, members per group: [2, 3, 5, 6]
```

Multi-pool backups (KV + SWA + Conv, 2–6 members) commit as one group. Each
staged batch still logs its `FRAME_SET`/`BATCH_COMMIT` pair (36 each) but no
longer fsyncs it, so the members' durability comes from the single
`GROUP_COMMIT` sync — the frames describe the group, they do not publish it.

Failure injection through the real controller + backend
(`test_extent_group_commit.py`): refused Conv member (geometry), failed KV
batch, `commit_group` returning False, and an exception inside `batch_set_v2`.
Each case asserts: no `GROUP_COMMIT` in the WAL, `_staged_groups` and
`_staged_reserved` empty (resources released), `exists()` false, `batch_exists_v2`
reporting 0 restorable pages, and — after reopening the backend — still absent.

Both writer arms are pinned by `native/test_l3_native_wal_equivalence.py`
`GroupCommitArmTests`: identical WAL frame sequences for a grouped write, abort
writing no commit frame, and an uncommitted group not recovered after restart.

### 4b.2 Restorable prefixes are intersected, not min-folded

`ExtentFileBackend.batch_exists_v2` used to fold each pool to its own maximum
and take the minimum. SWA and Conv hold *trailing windows*, so their restorable
lengths are sparse sets that need not coincide:

| construction | SWA max | Conv max | old answer (min) | new answer |
|---|---|---|---|---|
| SWA window 2 ending at page 4; Conv window 1 at page 6 | 4 | 6 | **4** (illegal: Conv has no state at 4) | **0** |
| SWA window 1 at pages 4,6; Conv window 1 at pages 4,5 | 6 | 5 | 5 (illegal for SWA) | **4** |
| KV pages 5..7 absent; both windows cover 0..4 | 5 | 5 | 5 | 5 |

The function now uses `HiCacheFile.batch_exists_v2`'s `restorable_prefix_pages`
set intersection with `ExtentAllocator.lookup` as the existence query, keeping
the extent-specific trailing-window sizing and the placeholder-key handling for
prefetch transfers; `extra_pool_hit_pages` still reports each pool's own
boundary so the per-pool existence beliefs heal independently.

### 4b.3 Writer choice and cost

Routing KV through `batch_set_v2` inside a group cost ~10% at page=1 (per-page
Python loop over 8192 pages), so `batch_set_v1` gained a `group_id` and stages
its zero-copy pages like v2 does; the controller picks the writer once per
process. Staged members also skip their own `fdatasync`, since nothing is
visible before `GROUP_COMMIT`. In-process, one 8192-page backup:

```
ungrouped v2      153.0 ms
grouped   v2+v1   141.7 ms
```

### 4b.4 Performance re-test

| arm | 1K | 4K | 8K |
|---|---|---|---|
| extent, before round 2 (page 256) | 96.6 | 205.6 | 348.6 |
| extent, after round 2 (page 256) | 97.4 | 207.6 | 354.3 |
| extent, before round 2 (page 1) | 361.6 | 964.7 | 1810.9 |
| extent, after round 2 (page 1) | 354.3 | 1045.5 | 2021.5 |
| **unmodified build re-measured the same day (page 1)** | — | — | **1967.2** |

The page-1 arm was re-measured against the pre-change code on the same machine
(`git stash` of the round-2 sources, same launch line, same tier directory):
1967.2 ms vs 2021.5 ms, i.e. within the 1811–2055 ms spread this box shows
run-to-run, so group commit costs no measurable restore time. Read syscalls and
bytes are unchanged (8K: 207 vs 209 `syscr`, 103.6 MB `rchar`).

Correctness is unchanged on the new path, in every arm:

| arm (page 256) | matrix | storage-served cases | repeat |
|---|---|---|---|
| extent, Python writer | 45/45 identical ids | 18 | 5/5 `storage=1024` |
| extent, native writer (`SGLANG_L3_NATIVE_WRITE_PATH=1`) | 45/45 | 17 | — |
| extent, `buffer_only` (existence beliefs active) | 45/45 | 17 | 5/5 |

The native arm's live WAL shows the same grouped commits as the Python arm
(`evidence/e052_group_commit_wal_scan_native.txt`: 17 `GROUP_COMMIT`, members
2/3/5/6), so the Rust writer participates in group commit rather than bypassing
it. Native restore timings with group commit: 96.7 / 207.9 / 351.0 ms at
1K/4K/8K, i.e. equal to the Python arm.

### 4b.5 Test totals after round 2

extent backend 127 (incl. 3 v1-group cases), extent group commit 10, allocator
17, metadata 42, native 23 (incl. 3 arm-parity group cases), reliability 26,
prefetch lifecycle 16, sidecar 6.

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
