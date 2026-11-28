"""
Unit tests for the ExtentFileBackend (circular extent allocator, aggregated
page IO, generation invalidation, watermark GC, crash-reopen semantics) and
its pure-logic allocator.

The bulk of the suite is driven with bytearray pages end-to-end (no torch
import): ``_page_byte_view`` treats byte-like buffers as raw bytes.  The
``TestExtentTensorPages`` cases additionally drive the real torch.Tensor
path (the engine's host-pool flat pages are torch tensors, bf16/fp16/fp32
CPU) and skip themselves when torch is not installed.
``TestExtentV1ZeroCopy`` covers the batch_get_v1/batch_set_v1 zero-copy
surface against a real page_first MHA host pool (k/v dual-block preadv
split, config-suffix isolation, return-contract negatives); it skips itself
when torch is not installed.  ``TestExtentIovMaxBoundary`` pins the
IOV_MAX(1024)-iovec sharding boundary on the raw IO helpers (torch-free).

Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_extent_file_backend_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None  # tensor cases skip; the bytearray suite stays torch-free

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.extent.extent_allocator import (
    ExtentAllocator,
    ExtentRef,
)
from sglang.srt.mem_cache.storage.extent.extent_file_backend import (
    ExtentFileBackend,
    _page_byte_view,
)
from sglang.srt.mem_cache.storage.extent.extent_metadata import (
    FRAME_BATCH_COMMIT,
    FRAME_CLEAR,
    FRAME_DROP_SLOT,
    FRAME_GROUP_COMMIT,
    GroupCommitPayload,
    RecoveryError,
    component_from_pool_name,
    decode_group_commit_payload,
    encode_group_commit_payload,
    iter_segment_frames,
    read_manifest,
    read_current,
    segment_path,
    write_manifest_atomic,
)
from sglang.test.test_utils import CustomTestCase

_EXTENT = 4096


def _write(alloc: ExtentAllocator, key: str, length: int) -> ExtentRef:
    refs, passed = alloc.plan([(key, length)])
    assert refs[0] is not None, f"key {key} refused by allocator"
    alloc.commit([key], refs, passed)
    return refs[0]



def _host_pool_kwargs(device_pool):
    """Host-pool construction kwargs for the current host-pool API.

    The v0.5.23 host pools take a ``MHABufferInfo`` describing the device
    pool's physical buffers instead of holding the device pool object.
    """
    return {
        "buffer_info": device_pool.get_device_pool_infos()[0].buffer_info,
        "device_capacity_tokens": (
            device_pool.size
            if device_pool.host_capacity_tokens is None
            else device_pool.host_capacity_tokens
        ),
        "start_layer": device_pool.start_layer,
        "end_layer": device_pool.end_layer,
        "write_back_staging": device_pool.hicache_write_back_staging,
    }

class TestExtentAllocator(CustomTestCase):
    def test_reclaim_refusal_when_wrapped_batch_exhausts_all_tail_space(self):
        """A whole-file batch may leave no legal relocation destination.

        This is a deliberately constructed fail-closed geometry; r3's
        operation-time allocator state has not been reconstructed. The test
        uses 512 slots,
        64-MiB extents, a 770-page/14,336-byte batch wrapping through slot 0,
        and four small live refs in slot 0 after its cursor was previously
        advanced to the tail. Cursors in all non-victim slots are full, so the
        770-page plan is valid but the live refs cannot be relocated.
        """
        extent_size = 64 * 1024 * 1024
        alloc = ExtentAllocator(extent_size=extent_size, num_slots=512)
        alloc._next_slot = 511
        alloc._cursors = [extent_size] * alloc.num_slots
        # The slot cursor remains at the tail although only four live 25-KiB
        # entries survive later overwrites in that slot.
        live_refs = [
            ExtentRef(0, extent_size - 100 * 1024 + i * 25 * 1024, 25 * 1024, 0)
            for i in range(4)
        ]
        for i, ref in enumerate(live_refs):
            key = f"victim-{i}"
            alloc._index[key] = ref
            alloc._slot_keys[0].add(key)
            alloc._slot_bytes[0] += ref.length
            alloc._recency[key] = None
        keys = [(f"batch-{i}", 14_336) for i in range(770)]
        refs, passed, drop_events = alloc.plan_split(keys)
        self.assertTrue(all(ref is not None for ref in refs))
        self.assertEqual(passed, [0])
        excluded = {slot for slot, _ in drop_events}
        excluded.update(ref.extent_id for ref in refs if ref is not None)
        relocated = alloc.plan_relocation(
            [0],
            [(f"victim-{i}", ref.length) for i, ref in enumerate(live_refs)],
            excluded_slots=sorted(excluded),
            start_slot=max(ref.extent_id for ref in refs if ref is not None),
        )
        self.assertIsNone(relocated)
        self.assertEqual(alloc.lookup("victim-0"), live_refs[0])

    def _alloc(self, num_slots=4, extent_size=_EXTENT) -> ExtentAllocator:
        return ExtentAllocator(extent_size=extent_size, num_slots=num_slots)

    def test_extent_size_must_be_4096_aligned(self):
        for bad in (0, -1, 4095, 8191):
            with self.assertRaises(ValueError):
                ExtentAllocator(extent_size=bad, num_slots=2)
        ExtentAllocator(extent_size=4096, num_slots=2)  # must not raise

    def test_plan_is_pure_until_commit(self):
        # A failed pwrite must leave no state behind: plan() is read-only.
        alloc = self._alloc()
        refs, passed = alloc.plan([("k1", 100), ("k2", 100)])
        self.assertEqual(passed, [])
        self.assertIsNone(alloc.lookup("k1"))
        self.assertEqual(alloc.used_bytes, 0)
        # Ref gens are valid for a later commit.
        alloc.commit(["k1", "k2"], refs, passed)
        self.assertIsNotNone(alloc.lookup("k1"))
        self.assertEqual(alloc.used_bytes, 200)

    def test_pack_consecutive_keys_into_one_extent(self):
        alloc = self._alloc()
        refs, passed = alloc.plan([(f"k{i}", 1000) for i in range(4)])
        self.assertEqual(passed, [])
        self.assertEqual({r.extent_id for r in refs}, {0})
        self.assertEqual([r.offset for r in refs], [0, 1000, 2000, 3000])
        alloc.commit([f"k{i}" for i in range(4)], refs, passed)
        self.assertEqual(alloc.used_bytes, 4000)

    def test_circular_wrap_drops_oldest_slot_and_bumps_generation(self):
        # 4 keys per 4096 slot with 1000-byte pages: key 17 wraps into slot 0.
        alloc = self._alloc(num_slots=4)
        gen0 = []
        for i in range(17):
            ref = _write(alloc, f"k{i}", 1000)
            gen0.append(ref)
        wrapped = alloc.lookup("k16")
        self.assertEqual(wrapped.extent_id, 0)
        self.assertEqual(wrapped.offset, 0)
        self.assertEqual(wrapped.generation, 1)  # slot 0 reused -> gen++
        # Keys of the dropped slot are gone (miss), including via snapshot.
        self.assertIsNone(alloc.lookup("k0"))
        self.assertEqual(alloc.snapshot(["k0", "k1", "k16"]), [None, None, wrapped])
        # A ref captured before the wrap fails re-verification.
        self.assertEqual(alloc.verify(["k0"], [gen0[0]]), [False])
        self.assertEqual(alloc.verify(["k16"], [wrapped]), [True])

    def test_generation_invalidation_after_slot_reuse(self):
        alloc = self._alloc(num_slots=1, extent_size=_EXTENT)
        first = _write(alloc, "k", 1000)
        second = _write(alloc, "k", 1000)  # same key: cursor advances
        self.assertNotEqual(first.offset, second.offset)
        self.assertEqual(first.generation, second.generation)
        # Fill the single slot, then plan a page that no longer fits: the
        # entered slot is reused, so generations must bump and stale refs
        # must miss after commit.
        for i in range(14):
            _write(alloc, f"f{i}", 1000)
        refs, passed = alloc.plan([("other", 1000)])
        self.assertIn(0, passed)
        alloc.commit(["other"], refs, passed)
        self.assertEqual(alloc.verify(["k"], [second]), [False])
        self.assertEqual(alloc.lookup("k"), None)
        self.assertEqual(alloc.verify(["other"], [refs[0]]), [True])

    def test_same_key_rewrite_unlinks_old_slot_accounting(self):
        alloc = self._alloc(num_slots=2)
        old = _write(alloc, "k", 1000)  # slot 0
        _write(alloc, "j", 1000)  # slot 0, offset 1000
        new = _write(alloc, "k", 500)  # still fits slot 0 -> offset 2000
        self.assertEqual(old.extent_id, new.extent_id)
        self.assertEqual(alloc.used_bytes, 1000 + 500)  # old length unlinked
        self.assertEqual(alloc.verify(["k"], [old]), [False])
        self.assertEqual(alloc.lookup("k"), new)
        self.assertIsNotNone(alloc.lookup("j"))

    def test_page_larger_than_extent_refused(self):
        alloc = self._alloc()
        refs, _ = alloc.plan([("big", 5000)])
        self.assertIsNone(refs[0])
        self.assertEqual(alloc.used_bytes, 0)

    def test_batch_larger_than_whole_file_refused(self):
        alloc = self._alloc(num_slots=2, extent_size=_EXTENT)
        refs, _ = alloc.plan([(f"k{i}", 3000) for i in range(3)])
        # 9000 bytes > 8192 total: cannot place without self-overwrite.
        self.assertTrue(all(r is None for r in refs))
        self.assertEqual(alloc.used_bytes, 0)

    def test_evict_slot_lru_frees_oldest_slot(self):
        alloc = self._alloc(num_slots=4)
        for i in range(8):
            _write(alloc, f"k{i}", 1000)
        # k0..k3 in slot 0, k4..k7 in slot 1; k0 is the LRU key.
        freed = alloc.evict_slot_lru()
        self.assertEqual(freed, 4000)  # slot 0's live bytes
        self.assertEqual(alloc.used_bytes, 4000)
        self.assertIsNone(alloc.lookup("k0"))
        self.assertIsNotNone(alloc.lookup("k4"))
        # Slots holding newer keys survive; second evict takes slot 1.
        self.assertEqual(alloc.evict_slot_lru(), 4000)
        self.assertEqual(alloc.used_bytes, 0)
        self.assertEqual(alloc.evict_slot_lru(), 0)  # empty -> no-op

    def test_touch_reorders_recency(self):
        # 2500-byte pages force one key per 4096 slot; touching "old" makes
        # "new" the LRU victim, so the evicted slot must be the newer key's.
        alloc = self._alloc(num_slots=4, extent_size=_EXTENT)
        _write(alloc, "old", 2500)
        _write(alloc, "new", 2500)
        alloc.touch("old")
        self.assertEqual(alloc.evict_slot_lru(), 2500)
        self.assertIsNotNone(alloc.lookup("old"))
        self.assertIsNone(alloc.lookup("new"))

    def test_clear_resets_index_and_bytes(self):
        alloc = self._alloc()
        _write(alloc, "k1", 1000)
        _write(alloc, "k2", 1000)
        alloc.clear()
        self.assertEqual(alloc.used_bytes, 0)
        self.assertIsNone(alloc.lookup("k1"))
        # Post-clear writes start from slot 0 again.
        ref = _write(alloc, "k3", 1000)
        self.assertEqual((ref.extent_id, ref.offset), (0, 0))


class _PagePool:
    """Minimal torch-free host-pool stand-in (page_size + flat page APIs)."""

    def __init__(self, page_bytes: int, slots: int = 256):
        self.page_size = 1
        self._page_bytes = page_bytes
        self._pages = [bytearray(page_bytes) for _ in range(slots)]

    def get_data_page(self, index, flat=True):
        return self._pages[index]

    def get_dummy_flat_data_page(self):
        return bytearray(self._page_bytes)

    def set_from_flat_data_page(self, index, data_page):
        self._pages[index][:] = data_page


def _make_config(extra_config=None, **overrides) -> HiCacheStorageConfig:
    cfg = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
    )
    cfg.update(overrides)
    return HiCacheStorageConfig(extra_config=extra_config, **cfg)


class ExtentBackendTestBase(CustomTestCase):
    """Drive the real backend against bytearray pages in a temp dir."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="extent_unit_")
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )

    def tearDown(self):
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _page(data: bytes) -> bytearray:
        return bytearray(data)

    def _page_bytes(self, i: int, page_bytes: int = 1000) -> bytearray:
        return self._page(bytes([i % 251]) * page_bytes)

    def _make_keys(self, count: int, start: int = 0) -> list:
        return [f"k{i:04d}" for i in range(start, start + count)]

    def _write_keys(self, count: int, page_bytes: int = 1000) -> list:
        keys = self._make_keys(count)
        values = [self._page_bytes(i, page_bytes) for i in range(count)]
        self.assertTrue(self.backend.batch_set(keys, values))
        return keys

    def _assert_pages_equal(self, got, expected_values):
        self.assertEqual(len(got), len(expected_values))
        for g, e in zip(got, expected_values):
            self.assertIsNotNone(g)
            self.assertEqual(bytes(g), bytes(e))


class TestExtentV1(ExtentBackendTestBase):
    def _wal_config(self, threshold):
        return _make_config(
            extra_config={
                "storage_dir": self.tmpdir,
                "extent_size": "16Ki",
                "file_size": "64Ki",
                "wal_checkpoint_bytes": threshold,
            }
        )

    def test_wal_checkpoint_threshold_disabled_is_inert(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(0))
        self.assertIsNone(self.backend._checkpoint_thread)
        self._write_keys(3)
        self.assertEqual(self.backend.stats_snapshot()["wal_checkpoint_successes"], 0)

    def test_wal_checkpoint_threshold_coalesces_and_close_joins_worker(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(1))
        self._write_keys(4)
        deadline = time.monotonic() + 2
        while self.backend.stats_snapshot()["wal_checkpoint_successes"] < 1:
            if time.monotonic() >= deadline:
                self.fail("coalesced WAL checkpoint did not complete")
            time.sleep(0.01)
        worker = self.backend._checkpoint_thread
        self.backend.close()
        self.assertFalse(worker.is_alive())
        self.assertIsNone(self.backend._checkpoint_thread)

    def test_wal_checkpoint_threshold_rotates_and_reopens_three_times(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(1))
        keys = []
        for round_id in range(3):
            key = f"automatic-{round_id}"
            self.assertTrue(self.backend.set(key, self._page_bytes(round_id)))
            keys.append(key)
            deadline = time.monotonic() + 2
            while self.backend.stats_snapshot()["wal_checkpoint_successes"] < round_id + 1:
                if time.monotonic() >= deadline:
                    self.fail("automatic WAL checkpoint did not complete")
                time.sleep(0.01)
        self.assertGreaterEqual(self.backend.stats_snapshot()["wal_checkpoint_successes"], 3)
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(0))
        self.assertTrue(all(self.backend.exists(key) for key in keys))

    def test_wal_checkpoint_config_precedence_and_validation(self):
        self.backend.close()
        with mock.patch.dict(
            os.environ,
            {"SGLANG_HICACHE_EXTENT_BACKEND_WAL_CHECKPOINT_BYTES": "2Ki"},
        ):
            env_backend = ExtentFileBackend(
                _make_config(
                    extra_config={
                        "storage_dir": self.tmpdir,
                        "extent_size": "16Ki",
                        "file_size": "64Ki",
                    }
                )
            )
            try:
                self.assertEqual(env_backend._wal_checkpoint_bytes, 2 * 1024)
                self.assertEqual(env_backend._wal_checkpoint_source, "env")
            finally:
                env_backend.close()
        for invalid in (-1, True, "not-a-size"):
            with self.assertRaises(ValueError):
                ExtentFileBackend(self._wal_config(invalid))

    def test_wal_checkpoint_failure_preserves_wal_and_retries(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(1))
        original = self.backend.checkpoint
        attempts = []

        def fail_once(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("injected checkpoint failure")
            return original(*args, **kwargs)

        with mock.patch.object(self.backend, "checkpoint", side_effect=fail_once):
            self.assertTrue(self.backend.set("retry", self._page_bytes(1)))
            deadline = time.monotonic() + 2
            while self.backend.stats_snapshot()["wal_checkpoint_successes"] < 1:
                if time.monotonic() >= deadline:
                    self.fail("checkpoint retry did not complete")
                time.sleep(0.01)
        stats = self.backend.stats_snapshot()
        self.assertGreaterEqual(stats["wal_checkpoint_failures"], 1)
        self.assertGreaterEqual(stats["wal_checkpoint_retries"], 1)
        self.assertGreaterEqual(len(attempts), 2)
        self.assertTrue(self.backend.exists("retry"))

    def test_wal_checkpoint_worker_preserves_newer_declined_request(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(1))
        attempts = []

        def decline_once(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                self.backend._checkpoint_event.set()
                return False
            return True

        with mock.patch.object(self.backend, "checkpoint", side_effect=decline_once):
            self.backend._checkpoint_event.set()
            deadline = time.monotonic() + 2
            while len(attempts) < 2:
                if time.monotonic() >= deadline:
                    self.fail("newer checkpoint request was cleared after decline")
                time.sleep(0.01)

    def test_wal_checkpoint_final_group_abort_retriggers_after_decline(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(1))
        group_id = self.backend.open_group()
        self.assertTrue(
            self.backend._write_pages(
                [self.backend._storage_key("aborted-retry")],
                [self._page_bytes(1)],
                group_id=group_id,
            )
        )
        deadline = time.monotonic() + 2
        while self.backend.stats_snapshot()["wal_checkpoint_requested"] < 1:
            if time.monotonic() >= deadline:
                self.fail("automatic checkpoint did not observe incomplete group")
            time.sleep(0.01)
        self.assertEqual(self.backend.stats_snapshot()["wal_checkpoint_successes"], 0)
        self.assertTrue(self.backend.abort_group(group_id))
        deadline = time.monotonic() + 2
        while self.backend.stats_snapshot()["wal_checkpoint_successes"] < 1:
            if time.monotonic() >= deadline:
                self.fail("final group abort did not retrigger checkpoint")
            time.sleep(0.01)

    def test_close_serializes_checkpoint_and_release(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(0))
        checkpoint_calls = []
        original = self.backend.checkpoint

        def checkpoint(*args, **kwargs):
            checkpoint_calls.append(1)
            time.sleep(0.05)
            return original(*args, **kwargs)

        with mock.patch.object(self.backend, "checkpoint", side_effect=checkpoint):
            closers = [threading.Thread(target=self.backend.close) for _ in range(2)]
            for closer in closers:
                closer.start()
            for closer in closers:
                closer.join(2)
                self.assertFalse(closer.is_alive())
        self.assertEqual(checkpoint_calls, [1])
        self.assertTrue(self.backend._closed)

    def test_checkpoint_worker_start_failure_cleans_up_for_reopen(self):
        self.backend.close()
        config = self._wal_config(1)
        with mock.patch(
            "sglang.srt.mem_cache.storage.extent.extent_file_backend.threading.Thread.start",
            side_effect=RuntimeError("injected thread start failure"),
        ), self.assertRaises(RuntimeError):
            ExtentFileBackend(config)

        reopened = ExtentFileBackend(self._wal_config(0))
        reopened.close()

    def test_wal_checkpoint_group_commit_signals_after_complete_group(self):
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(1))
        group_id = self.backend.open_group()
        keys = self._make_keys(2)
        storage_keys = [self.backend._storage_key(key) for key in keys]
        self.assertTrue(
            self.backend._write_pages(
                [storage_keys[0]], [self._page_bytes(1)], group_id=group_id
            )
        )
        self.assertTrue(
            self.backend._write_pages(
                [storage_keys[1]], [self._page_bytes(2)], group_id=group_id
            )
        )
        self.assertEqual(self.backend.stats_snapshot()["wal_checkpoint_successes"], 0)
        self.assertTrue(self.backend.commit_group(group_id))
        deadline = time.monotonic() + 2
        while self.backend.stats_snapshot()["wal_checkpoint_successes"] < 1:
            if time.monotonic() >= deadline:
                self.fail("group commit did not trigger automatic checkpoint")
            time.sleep(0.01)
        self.assertTrue(all(self.backend.exists(key) for key in keys))

    def test_group_commit_blocks_reader_until_all_members_install(self):
        group_id = self.backend.open_group()
        keys = self._make_keys(2)
        storage_keys = [self.backend._storage_key(key) for key in keys]
        self.assertTrue(
            self.backend._write_pages(
                [storage_keys[0]], [self._page_bytes(1)], group_id=group_id
            )
        )
        self.assertTrue(
            self.backend._write_pages(
                [storage_keys[1]], [self._page_bytes(2)], group_id=group_id
            )
        )

        install_started = threading.Event()
        allow_install = threading.Event()
        reader_started = threading.Event()
        reader_done = threading.Event()
        reader_result = []
        original_install = self.backend._allocator.commit_install

        def paused_install(install_keys, refs):
            install_started.set()
            self.assertTrue(allow_install.wait(2))
            return original_install(install_keys, refs)

        def reader():
            reader_started.set()
            reader_result.append(self.backend.batch_exists(keys))
            reader_done.set()

        with mock.patch.object(
            self.backend._allocator, "commit_install", side_effect=paused_install
        ):
            commit_thread = threading.Thread(
                target=self.backend.commit_group, args=(group_id,)
            )
            commit_thread.start()
            self.assertTrue(install_started.wait(2))

            reader_thread = threading.Thread(target=reader)
            reader_thread.start()
            self.assertTrue(reader_started.wait(2))
            self.assertFalse(reader_done.wait(0.1))

            allow_install.set()
            commit_thread.join(2)
            reader_thread.join(2)

        self.assertFalse(commit_thread.is_alive())
        self.assertFalse(reader_thread.is_alive())
        self.assertEqual(reader_result, [len(keys)])

    def test_same_namespace_backend_open_is_fenced_until_close(self):
        with self.assertRaises(RecoveryError) as raised:
            ExtentFileBackend(
                _make_config(
                    extra_config={
                        "storage_dir": self.tmpdir,
                        "extent_size": "16Ki",
                        "file_size": "64Ki",
                    }
                )
            )
        self.assertEqual(raised.exception.failure_class, "lock-busy")
        self.assertTrue(self.backend.set("first", self._page_bytes(1)))

        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        self.assertTrue(self.backend.exists("first"))

    def test_conflicting_duplicate_does_not_truncate_live_data_file(self):
        values = [self._page_bytes(i) for i in range(17)]
        keys = self._make_keys(len(values))
        self.assertTrue(self.backend.batch_set(keys, values))
        target_key = keys[-1]
        target_value = values[-1]
        data_path = self.backend._data_path
        original_size = os.stat(data_path).st_size
        self.assertEqual(original_size, 64 * 1024)
        self.assertEqual(
            bytes(self.backend.get(target_key, bytearray(len(target_value)))),
            bytes(target_value),
        )

        with self.assertRaises(RecoveryError) as raised:
            ExtentFileBackend(
                _make_config(
                    extra_config={
                        "storage_dir": self.tmpdir,
                        "extent_size": "16Ki",
                        "file_size": "16Ki",
                    }
                )
            )
        self.assertEqual(raised.exception.failure_class, "lock-busy")
        self.assertEqual(os.stat(data_path).st_size, original_size)
        self.assertEqual(
            bytes(self.backend.get(target_key, bytearray(len(target_value)))),
            bytes(target_value),
        )

        later_key = "after-conflicting-duplicate"
        later_value = self._page_bytes(99)
        self.assertTrue(self.backend.set(later_key, later_value))
        self.assertEqual(
            bytes(self.backend.get(later_key, bytearray(len(later_value)))),
            bytes(later_value),
        )

    def test_owner_only_namespace_isolated_from_legacy_cache_data(self):
        legacy = self.backend
        legacy_key = "gemma4-page-0001"
        legacy_value = self._page_bytes(41)
        self.assertTrue(legacy.set(legacy_key, legacy_value))

        owner = ExtentFileBackend(
            _make_config(
                model_name="testmodel-kv-owner-v1",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        try:
            self.assertNotEqual(legacy.config_suffix, owner.config_suffix)
            self.assertTrue(legacy.exists(legacy_key))
            self.assertFalse(owner.exists(legacy_key))
            owner_key = "gemma4-page-0001"
            owner_value = self._page_bytes(73)
            self.assertTrue(owner.set(owner_key, owner_value))
            self.assertTrue(owner.exists(owner_key))
            self.assertTrue(legacy.exists(legacy_key))
            self.assertEqual(
                bytes(legacy.get(legacy_key, bytearray(len(legacy_value)))),
                bytes(legacy_value),
            )
            self.assertEqual(
                bytes(owner.get(owner_key, bytearray(len(owner_value)))),
                bytes(owner_value),
            )
        finally:
            owner.close()

    def test_separate_config_suffix_can_open_while_namespace_is_held(self):
        other = ExtentFileBackend(
            _make_config(
                model_name="othermodel",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        try:
            self.assertNotEqual(self.backend.config_suffix, other.config_suffix)
            self.assertTrue(other.set("other", self._page_bytes(2)))
        finally:
            other.close()

    def test_group_open_uses_backend_epoch_and_monotonic_seq(self):
        first = self.backend.open_group()
        second = self.backend.open_group()
        self.assertEqual(first[0], self.backend._namespace_epoch)
        self.assertTrue(any(first[0]))
        self.assertEqual(first[1], 1)
        self.assertEqual(second, (first[0], 2))
        self.assertNotEqual(first, (b"\x00" * 16, 0))

    def test_grouped_write_is_hidden_until_commit(self):
        group_id = self.backend.open_group()
        self.assertTrue(
            self.backend._write_pages(
                [self.backend._storage_key("staged")],
                [self._page_bytes(3)],
                group_id=group_id,
            )
        )
        self.assertFalse(self.backend.exists("staged"))
        self.assertTrue(self.backend.commit_group(group_id))
        self.assertTrue(self.backend.exists("staged"))

    def test_abort_group_leaves_no_visible_entries(self):
        group_id = self.backend.open_group()
        self.assertTrue(
            self.backend._write_pages(
                [self.backend._storage_key("aborted")],
                [self._page_bytes(4)],
                group_id=group_id,
            )
        )
        self.assertTrue(self.backend.abort_group(group_id))
        self.assertFalse(self.backend.exists("aborted"))
        self.assertFalse(self.backend.commit_group(group_id))

    def test_group_commit_wal_and_restart_restore(self):
        group_id = self.backend.open_group()
        self.assertTrue(
            self.backend._write_pages(
                [self.backend._storage_key("grouped")],
                [self._page_bytes(5)],
                group_id=group_id,
            )
        )
        self.assertTrue(self.backend.commit_group(group_id))
        frames = iter_segment_frames(
            segment_path(self.tmpdir, self.backend.config_suffix, self.backend._wal_generation),
            active=True,
        )
        self.assertIn(FRAME_GROUP_COMMIT, [frame.frame_type for frame in frames])
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(extra_config={"storage_dir": self.tmpdir, "extent_size": "16Ki", "file_size": "64Ki"})
        )
        self.assertTrue(self.backend.exists("grouped"))

    def test_incomplete_group_wal_is_dropped_on_restart(self):
        group_id = self.backend.open_group()
        self.assertTrue(
            self.backend._write_pages(
                [self.backend._storage_key("incomplete")],
                [self._page_bytes(6)],
                group_id=group_id,
            )
        )
        self.assertFalse(self.backend.exists("incomplete"))
        # Simulate a crash close: preserve the durable SET/BATCH_COMMIT prefix
        # and skip the normal close-time checkpoint/abort path.
        self.backend._constructor_complete = False
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(extra_config={"storage_dir": self.tmpdir, "extent_size": "16Ki", "file_size": "64Ki"})
        )
        self.assertFalse(self.backend.exists("incomplete"))

    def test_group_commit_member_count_mismatch_stays_invisible_after_restart(self):
        group_id = self.backend.open_group()
        keys = self._make_keys(2)
        storage_keys = [self.backend._storage_key(key) for key in keys]
        self.assertTrue(
            self.backend._write_pages(
                storage_keys,
                [self._page_bytes(8), self._page_bytes(9)],
                group_id=group_id,
            )
        )
        # The staged batch has two entries, but this durable commit advertises
        # one.  Replay must reject the whole group rather than installing a
        # partial group.
        self.backend._append_wal_frame(
            FRAME_GROUP_COMMIT,
            encode_group_commit_payload(
                GroupCommitPayload(group_id[0], group_id[1], member_count=1)
            ),
        )
        self.backend._sync_wal()
        frames = iter_segment_frames(
            segment_path(self.tmpdir, self.backend.config_suffix, self.backend._wal_generation),
            active=True,
        )
        group_commits = [
            decode_group_commit_payload(frame.payload)
            for frame in frames
            if frame.frame_type == FRAME_GROUP_COMMIT
        ]
        self.assertEqual(group_commits[-1].member_count, 1)

        self.backend._constructor_complete = False
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        self.assertFalse(any(self.backend.exists(key) for key in keys))
        self.assertTrue(
            all(self.backend._allocator.lookup(key) is None for key in storage_keys)
        )

    def test_invalid_group_ids_fail_closed(self):
        with self.assertRaises(ValueError):
            self.backend.commit_group((b"\x00" * 16, 1))
        with self.assertRaises(ValueError):
            self.backend.abort_group((self.backend._namespace_epoch, 0))
        with self.assertRaises(ValueError):
            self.backend.commit_group((self.backend._namespace_epoch, 0))
        self.assertFalse(
            self.backend._write_pages(
                [self.backend._storage_key("invalid")],
                [self._page_bytes(7)],
                group_id=(b"\x00" * 16, 1),
            )
        )

    def test_single_key_round_trip(self):
        self.assertTrue(self.backend.set("k0", self._page_bytes(1)))
        self.assertTrue(self.backend.exists("k0"))
        target = bytearray(1000)
        got = self.backend.get("k0", target)
        self.assertIsNotNone(got)
        self.assertEqual(bytes(got), bytes(self._page_bytes(1)))
        self.assertFalse(self.backend.exists("missing"))

    def test_batch_round_trip(self):
        keys = self._make_keys(10)
        values = [self._page_bytes(i) for i in range(10)]
        self.assertTrue(self.backend.batch_set(keys, values))
        targets = [bytearray(1000) for _ in keys]
        got = self.backend.batch_get(keys, targets)
        self._assert_pages_equal(got, values)
        # Misses come back as None, in order.
        got = self.backend.batch_get(keys + ["nope"], targets + [bytearray(1000)])
        self.assertIsNone(got[-1])
        self.assertEqual(self.backend.batch_get([], []), [])

    def test_same_key_rewrite_returns_new_value(self):
        self.assertTrue(self.backend.set("k", bytearray(b"A" * 1000)))
        self.assertTrue(self.backend.set("k", bytearray(b"B" * 1000)))
        got = self.backend.get("k", bytearray(1000))
        self.assertEqual(bytes(got), b"B" * 1000)

    def test_get_size_mismatch_returns_none(self):
        self.assertTrue(self.backend.set("k", self._page_bytes(1)))
        self.assertIsNone(self.backend.get("k", bytearray(500)))

    def test_batch_exists_consecutive_count(self):
        self._write_keys(5)
        self.assertEqual(self.backend.batch_exists(self._make_keys(5) + ["x"]), 5)
        self.assertEqual(self.backend.batch_exists(["x", "k0000"]), 0)
        self.assertEqual(self.backend.batch_exists([]), 0)

    def test_aggregate_read_one_pread_per_extent(self):
        # One extent (128 KiB) holds the whole 100-key batch: batch_get must
        # issue exactly one pread (double-checked via stats and a syscall spy).
        backend = ExtentFileBackend(
            _make_config(
                model_name="aggregate-read",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "128Ki",
                    "file_size": "1Mi",
                }
            )
        )
        try:
            keys = self._make_keys(100)
            values = [self._page_bytes(i) for i in range(100)]
            self.assertTrue(backend.batch_set(keys, values))
            preads_before = backend.stats_snapshot()["preads"]
            with mock.patch.object(os, "preadv", wraps=os.preadv) as spy:
                targets = [bytearray(1000) for _ in keys]
                got = backend.batch_get(keys, targets)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(backend.stats_snapshot()["preads"] - preads_before, 1)
            self._assert_pages_equal(got, values)
        finally:
            backend.close()

    def test_aggregate_write_one_pwrite_per_extent(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="aggregate-write",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "128Ki",
                    "file_size": "1Mi",
                }
            )
        )
        try:
            keys = self._make_keys(100)
            values = [self._page_bytes(i) for i in range(100)]
            pwrites_before = backend.stats_snapshot()["pwrites"]
            with mock.patch.object(os, "pwritev", wraps=os.pwritev) as spy:
                self.assertTrue(backend.batch_set(keys, values))
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(backend.stats_snapshot()["pwrites"] - pwrites_before, 1)
        finally:
            backend.close()

    def test_multi_extent_round_trip_over_128_keys(self):
        # 200 x 1000 B = 13 extents in a 256 KiB file: multiple pwritev
        # regions, but every page must come back byte-identical.
        backend = ExtentFileBackend(
            _make_config(
                model_name="multi-extent",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "256Ki",
                }
            )
        )
        try:
            keys = self._make_keys(200)
            values = [self._page_bytes(i) for i in range(200)]
            self.assertTrue(backend.batch_set(keys, values))
            targets = [bytearray(1000) for _ in keys]
            got = backend.batch_get(keys, targets)
            self._assert_pages_equal(got, values)
            self.assertGreater(backend.stats_snapshot()["pwrites"], 1)
        finally:
            backend.close()

    def test_circular_wrap_relocates_live_refs_or_evicts_cache_safely(self):
        """F1 reclaim preserves data when it fits and evicts only on pressure.

        When relocation capacity exists the live head is moved. When all
        remaining tail space is exhausted, this cache backend may invalidate
        the LRU victim with a durable DROP_SLOT and accept the new write.
        """
        # (a) The 4x16 KiB file is full. Relocation has no destination, so
        # circular reuse durably evicts the victim slot and accepts new pages.
        for start in range(0, 60, 20):
            self._write_keys_into(self.backend, 20, start)
        retained_keys = [f"k{i:04d}" for i in (0, 19, 20, 39, 40, 59)]
        retained_values = [self._page_bytes(i) for i in (0, 19, 20, 39, 40, 59)]
        wal_offset_before = self.backend._wal_offset
        wrap_keys = self._make_keys(20, start=60)
        wrap_values = [self._page_bytes(i) for i in range(60, 80)]
        self.assertTrue(self.backend.batch_set(wrap_keys, wrap_values))
        self.assertFalse(self.backend.exists("k0000"))
        self.assertTrue(self.backend.exists("k0060"))
        evicted_hashes = self.backend.drain_evicted_storage_keys()
        self.assertIn("k0000", evicted_hashes)
        self.assertNotIn("k0060", evicted_hashes)
        self.assertLess(self.backend._wal_offset, wal_offset_before + 4096)
        retained_targets = [bytearray(1000) for _ in retained_keys]
        retained = self.backend.batch_get(retained_keys, retained_targets)
        self.assertEqual(retained[0], None)
        self._assert_pages_equal(retained[1:], retained_values[1:])
        # Confirm the eviction is durable across a restart.
        self.backend.close()
        self.backend = ExtentFileBackend(self._wal_config(0))
        self.assertFalse(self.backend.exists("k0000"))
        self.assertTrue(self.backend.exists("k0060"))
        new_targets = [bytearray(1000) for _ in wrap_keys]
        self._assert_pages_equal(
            self.backend.batch_get(wrap_keys, new_targets), wrap_values
        )
        # (b) Destination capacity present: slot 0 is full but holds only a
        # small live head (a large recorded filler was overwritten out of it)
        # and the middle slots keep tail room, so the same circular reclaim
        # relocates the head into non-victim space instead of dropping it.
        relocate = ExtentFileBackend(
            _make_config(
                model_name="circular-relocate",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        try:
            head = {
                "h0": self._page_bytes(10, 500),
                "h1": self._page_bytes(11, 500),
                "h2": self._page_bytes(12, 500),
            }
            self.assertTrue(
                relocate.batch_set(
                    ["F1", "h0", "h1", "h2"],
                    [
                        self._page_bytes(1, 14000),
                        head["h0"],
                        head["h1"],
                        head["h2"],
                    ],
                )
            )
            # F1 no longer fits slot 0: it moves to slot 1 and dies in place.
            self.assertTrue(
                relocate.batch_set(["F1"], [self._page_bytes(99, 14000)])
            )
            # g6 wraps into the full slot 0 -> reclaim with the live head.
            self.assertTrue(
                relocate.batch_set(
                    [f"g{i}" for i in range(9)],
                    [self._page_bytes(100 + i, 5000) for i in range(9)],
                )
            )
            self.assertEqual(relocate.batch_exists(["h0", "h1", "h2"], None), 3)
            self.assertEqual(relocate.drain_evicted_storage_keys(), [])
            for key, value in head.items():
                got = relocate.get(key, bytearray(len(value)))
                self.assertEqual(bytes(got), bytes(value))
                ref = relocate._allocator.lookup(relocate._storage_key(key))
                self.assertIsNotNone(ref)
                self.assertNotEqual(ref.extent_id, 0)
        finally:
            relocate.close()

    def test_eviction_during_inflight_read_returns_miss_not_reused_bytes(self):
        for start in range(0, 60, 20):
            self._write_keys_into(self.backend, 20, start)
        read_started = threading.Event()
        allow_read = threading.Event()
        original_pread = self.backend._pread_full

        def delayed_pread(buf, offset):
            read_started.set()
            if not allow_read.wait(timeout=5):
                raise TimeoutError("test did not release the in-flight read")
            return original_pread(buf, offset)

        result = {}
        with mock.patch.object(self.backend, "_pread_full", side_effect=delayed_pread):
            thread = threading.Thread(
                target=lambda: result.setdefault(
                    "pages", self.backend.batch_get(["k0000"], [bytearray(1000)])
                )
            )
            thread.start()
            self.assertTrue(read_started.wait(timeout=5))
            wrap_keys = self._make_keys(20, start=60)
            wrap_values = [self._page_bytes(i) for i in range(60, 80)]
            self.assertTrue(self.backend.batch_set(wrap_keys, wrap_values))
            allow_read.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(result["pages"], [None])
        self.assertTrue(self.backend.exists("k0060"))
        self.assertFalse(self.backend.exists("k0000"))

    def test_lru_reclaim_survives_multiple_full_file_cycles_and_restart(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="multi-cycle-cache-reclaim",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        try:
            live = {}
            evicted = set()
            for cycle in range(5):
                for batch in range(8):
                    base = cycle * 80 + batch * 10
                    keys = [f"cycle-{base+i}" for i in range(10)]
                    values = [self._page_bytes(base+i, 1000) for i in range(10)]
                    before = set(live)
                    self.assertTrue(backend.batch_set(keys, values))
                    live.update(zip(keys, values))
                    for key in before:
                        if not backend.exists(key):
                            evicted.add(key)
                            live.pop(key, None)
                    targets = [bytearray(1000) for _ in keys]
                    self._assert_pages_equal(backend.batch_get(keys, targets), values)
                self.assertGreater(backend._allocator._next_slot, -1)
                self.assertEqual(
                    backend._allocator.used_bytes,
                    sum(ref.length for ref in backend._allocator._index.values()),
                )
                self.assertEqual(
                    set(backend._allocator._index),
                    {key + backend.config_suffix for key in live},
                )
            self.assertTrue(evicted, "cyclic workload did not exercise LRU eviction")
            expected = dict(live)
            backend.close()
            backend = ExtentFileBackend(
                _make_config(
                    model_name="multi-cycle-cache-reclaim",
                    extra_config={
                        "storage_dir": self.tmpdir,
                        "extent_size": "16Ki",
                        "file_size": "64Ki",
                    },
                )
            )
            for key in evicted:
                self.assertFalse(backend.exists(key))
            readback = backend.batch_get(
                list(expected), [bytearray(1000) for _ in expected]
            )
            self._assert_pages_equal(readback, list(expected.values()))
        finally:
            backend.close()

    @unittest.skip("exploratory: group reservation relocation model not established")
    def test_group_reclaim_can_relocate_into_own_reserved_slot_tail(self):
        """A group may use its own staged extent's still-free tail for relocation.

        Other groups' staged reservations remain excluded. This models the
        write-through L3 case where one atomic group accumulates multiple
        storage writes before the final group commit.
        """
        backend = ExtentFileBackend(
            _make_config(
                model_name="group-relocate-own-tail",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        try:
            # Slot 0: live refs to relocate; slot 1: staged by this group;
            # slot 2: staged by another group and must stay unavailable.
            kv = _PagePool(1000)
            mamba = _PagePool(1000)
            backend.register_mem_host_pool_v2(kv, PoolName.KV)
            backend.register_mem_host_pool_v2(mamba, PoolName.MAMBA)
            for idx, value in enumerate((1, 2, 3, 4, 5, 6, 7)):
                kv._pages[idx][:] = self._page_bytes(value, 1000)
            group = backend.open_group()
            other = backend.open_group()
            self.assertTrue(
                backend.batch_set_v2(
                    [PoolTransfer(PoolName.KV, host_indices=[0], keys=["own-reservation"])],
                    group_id=group,
                )[PoolName.KV][0]
            )
            self.assertTrue(
                backend.batch_set_v2(
                    [PoolTransfer(PoolName.KV, host_indices=[1], keys=["other-reservation"])],
                    group_id=other,
                )[PoolName.KV][0]
            )
            # Advance to slot 3, then wrap to victim slot 0. The new payload
            # occupies slot 0 after its old refs have been safely relocated.
            self.assertTrue(
                backend._write_pages(
                    ["advance2"], [self._page_bytes(5, 16000)], group_id=group
                )
            )
            self.assertTrue(
                backend._write_pages(
                    ["advance3"], [self._page_bytes(6, 16000)], group_id=group
                )
            )
            self.assertTrue(
                backend._write_pages(
                    ["wrap"], [self._page_bytes(7, 8000)], group_id=group
                )
            )
            self.assertTrue(backend.commit_group(group))
            self.assertTrue(backend.abort_group(other))
            self.assertEqual(backend.batch_exists(["live0", "live1"], None), 2)
            self._assert_pages_equal(
                backend.batch_get(
                    ["live0", "live1"], [bytearray(6000), bytearray(6000)]
                ),
                [self._page_bytes(1, 6000), self._page_bytes(2, 6000)],
            )
            self._assert_pages_equal(
                backend.batch_get(
                    ["own-reservation"], [bytearray(1000)]
                ),
                [self._page_bytes(3, 1000)],
            )
            self.assertEqual(backend.batch_exists(["other-reservation"], None), 0)
        finally:
            backend.close()

    def test_watermark_gc_keeps_used_bytes_under_cap(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="watermark-gc",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "1Mi",
                    "max_size": "48Ki",
                    "eviction_ratio": 0.9,
                }
            )
        )
        try:
            # 40 KiB is below the 43.2 KiB watermark; 46 KiB crosses it, and
            # the next write evicts the oldest 16 KiB slot before planning.
            self._write_keys_into(backend, 40, start=0)
            self._write_keys_into(backend, 6, start=40)
            self._write_keys_into(backend, 2, start=46)
            self.assertFalse(backend.exists("k0000"))
            self.assertTrue(backend.exists("k0016"))
            self.assertTrue(backend.exists("k0047"))
            self.assertLessEqual(
                backend._allocator.used_bytes, int(48 * 1024 * 0.9)
            )
        finally:
            backend.close()

    def _write_keys_into(self, backend, count, start=0):
        keys = self._make_keys(count, start=start)
        values = [self._page_bytes(i) for i in range(start, start + count)]
        self.assertTrue(backend.batch_set(keys, values))

    def test_min_free_space_pressure_evicts_oldest(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="min-free-space",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "1Mi",
                    "min_free_space": "1Ti",
                }
            )
        )
        try:
            # Unmeetable free-space watermark: every write evicts the oldest slot.
            self._write_keys_into(backend, 1, start=0)
            self._write_keys_into(backend, 1, start=1)
            self.assertFalse(backend.exists("k0000"))
            self.assertTrue(backend.exists("k0001"))
        finally:
            backend.close()

    def test_gc_victim_selection_samples_statvfs_once_and_walks_lru(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="gc-linear-lru",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "1Mi",
                    "min_free_space": "1Ti",
                },
            )
        )
        try:
            self._write_keys_into(backend, 8)
            recency = backend._allocator._recency
            expected = []
            selected = set()
            used = backend._allocator.used_bytes
            for key in recency:
                ref = backend._allocator._index[key]
                if ref.extent_id in selected:
                    continue
                selected.add(ref.extent_id)
                expected.append(ref.extent_id)
                used -= backend._allocator._slot_bytes[ref.extent_id]
                if used <= int(backend._max_size_bytes * backend._eviction_ratio):
                    break
            with mock.patch(
                "sglang.srt.mem_cache.storage.extent.extent_file_backend.os.statvfs",
                return_value=type("Stat", (), {"f_bavail": 0, "f_frsize": 4096})(),
            ) as statvfs:
                self.assertEqual(backend._select_gc_victims(), expected)
            statvfs.assert_called_once_with(backend._storage_dir)
        finally:
            backend.close()

    def test_gc_evict_stops_once_free_space_recovers(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="gc-no-overevict",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    # Large size ceiling: only the free-space watermark may
                    # drive this sweep.
                    "file_size": "64Mi",
                    "min_free_space": "1Ti",
                },
            )
        )
        try:
            self._write_keys_into(backend, 6)
            before = len(backend._allocator._index)
            # Free space is sampled per reclaimed slot, so the sweep stops as
            # soon as the filesystem reports enough room again instead of
            # draining the whole LRU on one stale pressure reading.
            falling = type("Stat", (), {"f_bavail": 0, "f_frsize": 4096})()
            recovered = type("Stat", (), {"f_bavail": 1 << 40, "f_frsize": 4096})()
            with mock.patch(
                "sglang.srt.mem_cache.storage.extent.extent_file_backend.os.statvfs",
                side_effect=[falling, recovered],
            ) as statvfs:
                backend._gc_evict()
            # Two samples: the pressure check, then the post-eviction re-check
            # that ends the sweep. A single stale reading would have kept
            # evicting every remaining slot.
            self.assertEqual(statvfs.call_count, 2)
            self.assertLess(len(backend._allocator._index), before)
        finally:
            backend.close()

    def test_batch_set_refused_when_page_exceeds_extent(self):
        big = bytearray(20 * 1024)  # > 16 KiB extent
        self.assertFalse(self.backend.set("big", big))
        self.assertFalse(self.backend.exists("big"))
        # All-or-nothing: the batch must not register the good keys either.
        self.assertFalse(
            self.backend.batch_set(["a", "b"], [bytearray(1000), big])
        )
        self.assertFalse(self.backend.exists("a"))

    def test_clear_empties_then_reusable(self):
        keys = self._write_keys(5)
        self.assertTrue(self.backend.clear())
        for k in keys:
            self.assertFalse(self.backend.exists(k))
        # Storage is reusable after clear.
        self._write_keys(3)
        self.assertTrue(self.backend.exists("k0000"))

    def test_checkpoint_rotates_and_reopens_manifest_snapshot(self):
        keys = self._write_keys(5)
        old_generation = self.backend._wal_generation
        self.assertTrue(self.backend.checkpoint())
        state = read_manifest(self.backend._manifest_path)
        self.assertEqual(state.wal_floor, old_generation + 1)
        self.assertEqual(read_current(self.backend._wal_current_path), old_generation + 1)
        self.assertTrue(os.path.exists(segment_path(self.tmpdir, self.backend.config_suffix, old_generation + 1)))
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(extra_config={"storage_dir": self.tmpdir, "extent_size": "16Ki", "file_size": "64Ki"})
        )
        self.assertTrue(all(self.backend.exists(key) for key in keys))

    def test_mid_life_checkpoint_crash_reopen_replays_post_checkpoint_appends(self):
        """A mid-life checkpoint must not renumber later WAL frames.

        ``checkpoint()`` publishes ``checkpoint_lsn = N`` and rotates to a
        fresh segment, so recovery seeds ``expected_seq = N + 1``.  If
        rotation lets ``_open_active_wal()`` drag ``_wal_next_seq`` back to
        the stale startup recovery watermark, every frame appended after the
        checkpoint is numbered below ``N + 1`` and the next non-clean restart
        raises ``WAL sequence discontinuity``.  A clean ``close()`` masks the
        defect because its shutdown checkpoint folds the post-checkpoint
        appends into the manifest.
        """
        pre_keys = self._write_keys(4)
        self.assertTrue(self.backend.checkpoint())
        checkpoint_lsn = read_manifest(self.backend._manifest_path).checkpoint_lsn
        self.assertEqual(self.backend._wal_next_seq, checkpoint_lsn + 1)
        post_keys = self._make_keys(4, start=100)
        post_values = [self._page_bytes(50 + i) for i in range(4)]
        self.assertTrue(self.backend.batch_set(post_keys, post_values))
        frames = iter_segment_frames(
            segment_path(
                self.tmpdir,
                self.backend.config_suffix,
                self.backend._wal_generation,
            ),
            active=True,
        )
        frame_seqs = [frame.seq for frame in frames]
        self.assertGreaterEqual(len(frame_seqs), 1)
        self.assertEqual(frame_seqs[0], checkpoint_lsn + 1)
        self.assertEqual(
            frame_seqs,
            list(range(frame_seqs[0], frame_seqs[0] + len(frame_seqs))),
        )
        # Non-clean exit: close the fds but skip the shutdown checkpoint
        # (same crash simulation as the incomplete-group restart test).
        self.backend._constructor_complete = False
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        for key in pre_keys + post_keys:
            self.assertTrue(self.backend.exists(key), key)
        targets = [bytearray(len(value)) for value in post_values]
        self._assert_pages_equal(self.backend.batch_get(post_keys, targets), post_values)

    def test_close_checkpoint_reopens_entries(self):
        keys = self._write_keys(3)
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(extra_config={"storage_dir": self.tmpdir, "extent_size": "16Ki", "file_size": "64Ki"})
        )
        self.assertTrue(all(self.backend.exists(key) for key in keys))

    def test_clear_is_durable_and_keeps_generations_monotonic(self):
        self._write_keys(3)
        before = list(self.backend._allocator._generations)
        self.assertTrue(self.backend.clear())
        self.assertEqual(read_manifest(self.backend._manifest_path).entries, [])
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(extra_config={"storage_dir": self.tmpdir, "extent_size": "16Ki", "file_size": "64Ki"})
        )
        self.assertEqual(self.backend._allocator.used_bytes, 0)
        self.assertGreaterEqual(self.backend._allocator._generations, before)

    def test_pending_rotation_manifest_is_promoted_on_reopen(self):
        keys = self._write_keys(2)
        active = self.backend._wal_generation
        with self.backend._lock:
            pending = self.backend._checkpoint_state(active + 1, active)
        write_manifest_atomic(self.backend._manifest_path, pending)
        create_path = segment_path(self.tmpdir, self.backend.config_suffix, active + 1)
        self.backend._constructor_complete = False
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(extra_config={"storage_dir": self.tmpdir, "extent_size": "16Ki", "file_size": "64Ki"})
        )
        self.assertEqual(self.backend._wal_generation, active + 1)
        self.assertTrue(os.path.exists(create_path))
        self.assertTrue(all(self.backend.exists(key) for key in keys))

    def test_checkpoint_unlinks_only_named_generations_below_floor(self):
        self._write_keys(2)
        first = self.backend._wal_generation
        self.assertTrue(self.backend.checkpoint())
        second = self.backend._wal_generation
        self.assertTrue(self.backend.checkpoint())
        state = read_manifest(self.backend._manifest_path)
        self.assertEqual(state.wal_floor, self.backend._wal_generation)
        self.assertFalse(os.path.exists(segment_path(self.tmpdir, self.backend.config_suffix, first)))
        self.assertFalse(os.path.exists(segment_path(self.tmpdir, self.backend.config_suffix, second)))
        with mock.patch.object(os, "listdir", side_effect=AssertionError("recovery must not listdir")):
            self.backend.close()
            self.backend = ExtentFileBackend(
                _make_config(extra_config={"storage_dir": self.tmpdir, "extent_size": "16Ki", "file_size": "64Ki"})
            )

    def test_checkpoint_holds_gate_through_unlink_and_releases_writer(self):
        self._write_keys(1)
        old_generation = self.backend._wal_generation
        unlink_started = threading.Event()
        allow_unlink = threading.Event()
        writer_done = threading.Event()
        original_unlink = os.unlink

        def unlink(path):
            if path == segment_path(self.tmpdir, self.backend.config_suffix, old_generation):
                self.assertTrue(self.backend._write_gate._is_owned())
                unlink_started.set()
                self.assertTrue(allow_unlink.wait(2))
            return original_unlink(path)

        def writer():
            try:
                self.backend.set("after-checkpoint", self._page_bytes(8))
            finally:
                writer_done.set()

        with mock.patch.object(os, "unlink", side_effect=unlink):
            checkpoint_thread = threading.Thread(target=self.backend.checkpoint)
            checkpoint_thread.start()
            self.assertTrue(unlink_started.wait(2))
            writer_thread = threading.Thread(target=writer)
            writer_thread.start()
            self.assertFalse(writer_done.wait(0.1))
            allow_unlink.set()
            checkpoint_thread.join(2)
            writer_thread.join(2)
        self.assertFalse(checkpoint_thread.is_alive())
        self.assertFalse(writer_thread.is_alive())
        self.assertTrue(self.backend.exists("after-checkpoint"))

    def test_crash_reopen_restores_durable_entries(self):
        keys = self._write_keys(5)
        for k in keys:
            self.assertTrue(self.backend.exists(k))
        self.backend.close()
        # G1 oracle change: the manifest/WAL rebuild restores entries after a
        # process restart. This is process-crash semantics only; power-loss
        # data durability is a separate, stronger tier (D1(b)).
        reopened = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        try:
            for k in keys:
                self.assertTrue(reopened.exists(k))
                self.assertIsNotNone(reopened.get(k, bytearray(1000)))
            stats = reopened.stats_snapshot()
            self.assertGreater(stats["preads"], 0)
            self.assertEqual(stats["pwrites"], 0)
            self.assertEqual(stats["opens"], 1)
            # New writes land in the same file after rebuild.
            self.assertTrue(reopened.set("fresh", self._page_bytes(1)))
            self.assertTrue(reopened.exists("fresh"))
        finally:
            reopened.close()

    def test_manifest_absent_with_persistent_lock_fails_closed(self):
        manifest_path = self.backend._manifest_path
        current_path = self.backend._wal_current_path
        lock_path = self.backend._meta_lock_path
        config_suffix = self.backend.config_suffix
        self.backend.close()
        self.backend = None

        active_generation = read_current(current_path)
        wal_active_path = segment_path(
            self.tmpdir, config_suffix, active_generation
        )
        wal_one_path = segment_path(
            self.tmpdir, config_suffix, active_generation + 1
        )
        os.unlink(manifest_path)
        os.unlink(current_path)
        os.replace(wal_active_path, wal_one_path)

        with self.assertRaises(RecoveryError) as raised:
            ExtentFileBackend(
                _make_config(
                    extra_config={
                        "storage_dir": self.tmpdir,
                        "extent_size": "16Ki",
                        "file_size": "64Ki",
                    }
                )
            )
        self.assertEqual(raised.exception.failure_class, "manifest-absent")
        self.assertTrue(os.path.exists(lock_path))

    def test_drop_slot_is_durable_before_reuse_pwrite(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="drop-slot",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                    "max_size": "1Ki",
                    "eviction_ratio": 0.5,
                }
            )
        )
        try:
            self.assertTrue(backend.set("old", self._page_bytes(1)))
            original = backend._pwritev
            observed = []

            def pwrite(offset, buffers):
                frames = iter_segment_frames(
                    segment_path(self.tmpdir, backend.config_suffix, 0), active=True
                )
                observed.append([frame.frame_type for frame in frames])
                return original(offset, buffers)

            with mock.patch.object(backend, "_pwritev", side_effect=pwrite):
                self.assertTrue(backend.set("new", self._page_bytes(2)))
            self.assertTrue(observed)
            self.assertIn(FRAME_DROP_SLOT, observed[-1])
        finally:
            backend.close()

    def test_batch_commit_fsync_follows_successful_data_pwrite(self):
        events = []
        original = self.backend._pwritev

        def pwrite(offset, buffers):
            events.append("pwrite")
            return original(offset, buffers)

        with mock.patch.object(self.backend, "_pwritev", side_effect=pwrite):
            with mock.patch.object(
                os, "fdatasync", side_effect=lambda fd: events.append("fdatasync")
            ):
                self.assertTrue(self.backend.set("ordered", self._page_bytes(1)))
        self.assertEqual(events, ["pwrite", "fdatasync"])

    def test_pwrite_failure_has_no_batch_commit_or_ack(self):
        with mock.patch.object(self.backend, "_pwritev", return_value=False):
            self.assertFalse(self.backend.set("failed", self._page_bytes(1)))
        frames = iter_segment_frames(
            segment_path(self.tmpdir, self.backend.config_suffix, 0), active=True
        )
        self.assertNotIn(FRAME_BATCH_COMMIT, [frame.frame_type for frame in frames])
        self.assertFalse(self.backend.exists("failed"))

    def test_successful_ungrouped_write_rebuilds_after_restart(self):
        value = self._page_bytes(7)
        self.assertTrue(self.backend.set("restart", value))
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        self.assertTrue(self.backend.exists("restart"))
        self.assertEqual(bytes(self.backend.get("restart", bytearray(1000))), bytes(value))

    def test_stats_counters_accumulate(self):
        keys = self._write_keys(10)
        targets = [bytearray(1000) for _ in keys]
        self.backend.batch_get(keys, targets)
        stats = self.backend.stats_snapshot()
        self.assertEqual(stats["opens"], 1)
        self.assertGreaterEqual(stats["pwrites"], 1)
        self.assertGreaterEqual(stats["preads"], 1)
        self.assertEqual(stats["bytes_written"], 10 * 1000)
        self.assertEqual(stats["bytes_read"], 10 * 1000)

    def test_get_stats_matches_engine_contract(self):
        # Bug regression (3090 extent engine crash): the scheduler's
        # check_hicache_events assigns onto get_stats()'s return value, so it
        # must be None (or StorageMetrics) -- a bare dict raises
        # AttributeError.  The raw io counters live under stats_snapshot().
        self.assertIsNone(self.backend.get_stats())
        stats = self.backend.stats_snapshot()
        for key in ("opens", "preads", "pwrites", "bytes_read", "bytes_written"):
            self.assertIn(key, stats)


class TestExtentV2(ExtentBackendTestBase):
    def _register_pools(self, page_bytes=1000):
        kv = _PagePool(page_bytes)
        mamba = _PagePool(page_bytes)
        self.backend.register_mem_host_pool_v2(kv, PoolName.KV)
        self.backend.register_mem_host_pool_v2(mamba, PoolName.MAMBA)
        return kv, mamba

    def _transfer(self, name, keys, indices=None, hit_policy=None):
        return PoolTransfer(
            name=name,
            host_indices=indices if indices is not None else list(range(len(keys))),
            keys=list(keys),
            hit_policy=hit_policy,
        )

    def test_unified_kv_mamba_extent_group_atomicity_and_restore(self):
        """Real Extent storage: KV+MAMBA commit together and restore bytes.

        This is the CPU storage half of Unified MAMBA integration. It uses
        genuine bytearray-backed registered pools and the production extent
        index/WAL/group/read paths; CUDA D<->H transfer remains a separate gate.
        """
        kv, mamba = self._register_pools(page_bytes=1024)
        kv_key, mamba_key, missing_key = self._make_keys(3)
        kv_bytes = self._page_bytes(41, page_bytes=1024)
        mamba_bytes = self._page_bytes(99, page_bytes=1024)
        kv._pages[0][:] = kv_bytes
        mamba._pages[0][:] = mamba_bytes

        group_id = self.backend.open_group()
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.KV, [kv_key])], group_id=group_id
            ),
            {PoolName.KV: [True]},
        )
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.MAMBA, [mamba_key])], group_id=group_id
            ),
            {PoolName.MAMBA: [True]},
        )
        # Neither key is visible before the combined KV+sidecar group commit.
        self.assertFalse(self.backend.exists(kv_key))
        self.assertFalse(self.backend.exists(mamba_key))
        self.assertTrue(self.backend.commit_group(group_id))
        self.assertTrue(self.backend.exists(kv_key))
        self.assertIsNotNone(
            self.backend._allocator.lookup(
                self.backend._component_key(PoolName.MAMBA, mamba_key)
            )
        )

        # A KV prefix without the required MAMBA checkpoint is not a usable
        # restore prefix; logical KV presence alone must fail closed.
        missing_result = self.backend.batch_exists_v2(
            [missing_key], [self._transfer(PoolName.MAMBA, [missing_key])]
        )
        self.assertEqual(missing_result.kv_hit_pages, 0)

        # Read into fresh CPU staging slots through the real backend pread path.
        restore_kv = _PagePool(page_bytes=1024)
        restore_mamba = _PagePool(page_bytes=1024)
        self.backend.register_mem_host_pool_v2(restore_kv, PoolName.KV)
        self.backend.register_mem_host_pool_v2(restore_mamba, PoolName.MAMBA)
        restored = self.backend.batch_get_v2(
            [
                self._transfer(PoolName.KV, [kv_key]),
                self._transfer(PoolName.MAMBA, [mamba_key]),
            ]
        )
        self.assertEqual(restored[PoolName.KV], [True])
        self.assertEqual(restored[PoolName.MAMBA], [True])
        self.assertEqual(bytes(restore_kv._pages[0]), bytes(kv_bytes))
        self.assertEqual(bytes(restore_mamba._pages[0]), bytes(mamba_bytes))
        stats = self.backend.stats_snapshot()
        self.assertGreaterEqual(stats["preads"], 1)
        self.assertGreaterEqual(stats["bytes_read"], 2 * len(kv_bytes))

    def test_unified_kv_mamba_extent_group_commit_failure_is_atomic(self):
        kv, mamba = self._register_pools(page_bytes=1024)
        kv_key, mamba_key = self._make_keys(2)
        kv._pages[0][:] = self._page_bytes(17, page_bytes=1024)
        mamba._pages[0][:] = self._page_bytes(23, page_bytes=1024)
        group_id = self.backend.open_group()
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.KV, [kv_key])], group_id=group_id
            ),
            {PoolName.KV: [True]},
        )
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.MAMBA, [mamba_key])], group_id=group_id
            ),
            {PoolName.MAMBA: [True]},
        )

        with mock.patch(
            "os.pwrite", side_effect=OSError("injected extent commit failure")
        ):
            with self.assertRaisesRegex(OSError, "injected extent commit failure"):
                self.backend.commit_group(group_id)
        self.assertFalse(self.backend.exists(kv_key))
        self.assertFalse(self.backend.exists(mamba_key))

    def test_v2_round_trip_kv_pool(self):
        kv, _ = self._register_pools()
        keys = self._make_keys(4)
        values = [self._page_bytes(i) for i in range(4)]
        for i, v in enumerate(values):
            kv._pages[i][:] = v
        result = self.backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
        self.assertEqual(result[PoolName.KV], [True] * 4)
        result = self.backend.batch_get_v2([self._transfer(PoolName.KV, keys)])
        self.assertEqual(result[PoolName.KV], [True] * 4)
        for i, v in enumerate(values):
            self.assertEqual(bytes(kv._pages[i]), bytes(v))

    def test_v2_grouped_write_stages_until_backend_commit(self):
        kv, _ = self._register_pools()
        key = self._make_keys(1)[0]
        value = self._page_bytes(8)
        kv._pages[0][:] = value
        group_id = self.backend.open_group()
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.KV, [key])], group_id=group_id
            ),
            {PoolName.KV: [True]},
        )
        self.assertFalse(self.backend.exists(key))
        self.assertTrue(self.backend.commit_group(group_id))
        self.assertTrue(self.backend.exists(key))
        self.assertEqual(
            self.backend.batch_get_v2([self._transfer(PoolName.KV, [key])]),
            {PoolName.KV: [True]},
        )
        self.assertEqual(bytes(kv._pages[0]), bytes(value))

    def test_group_members_restore_after_crash_reopen_with_component_metadata(self):
        kv, mamba = self._register_pools()
        kv_key, mamba_key = self._make_keys(2)
        kv_value = self._page_bytes(10)
        mamba_value = self._page_bytes(11)
        kv._pages[0][:] = kv_value
        mamba._pages[0][:] = mamba_value
        group_id = self.backend.open_group()

        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.KV, [kv_key])], group_id=group_id
            ),
            {PoolName.KV: [True]},
        )
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.MAMBA, [mamba_key])], group_id=group_id
            ),
            {PoolName.MAMBA: [True]},
        )
        self.assertFalse(self.backend.exists(kv_key))
        self.assertFalse(self.backend.exists(mamba_key))
        self.assertTrue(self.backend.commit_group(group_id))

        frames = iter_segment_frames(
            segment_path(self.tmpdir, self.backend.config_suffix, self.backend._wal_generation),
            active=True,
        )
        group_commits = [
            decode_group_commit_payload(frame.payload)
            for frame in frames
            if frame.frame_type == FRAME_GROUP_COMMIT
        ]
        self.assertEqual(group_commits[-1].member_count, 2)

        self.backend._constructor_complete = False
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        restored_kv, restored_mamba = self._register_pools()
        self.assertTrue(self.backend.exists(kv_key))
        self.assertIsNotNone(
            self.backend._allocator.lookup(
                self.backend._component_key(PoolName.MAMBA, mamba_key)
            )
        )
        self.assertEqual(
            self.backend._components[self.backend._component_key(PoolName.KV, kv_key)],
            component_from_pool_name(PoolName.KV),
        )
        self.assertEqual(
            self.backend._components[
                self.backend._component_key(PoolName.MAMBA, mamba_key)
            ],
            component_from_pool_name(PoolName.MAMBA),
        )
        self.assertEqual(
            self.backend.batch_get_v2(
                [self._transfer(PoolName.KV, [kv_key])]
            )[PoolName.KV],
            [True],
        )
        self.assertEqual(
            self.backend.batch_get_v2(
                [self._transfer(PoolName.MAMBA, [mamba_key])]
            )[PoolName.MAMBA],
            [True],
        )
        self.assertEqual(bytes(restored_kv._pages[0]), bytes(kv_value))
        self.assertEqual(bytes(restored_mamba._pages[0]), bytes(mamba_value))

    def test_v2_sidecar_component_keys_do_not_share_index(self):
        _, mamba = self._register_pools()
        keys = self._make_keys(3)
        for i in range(3):
            mamba._pages[i][:] = self._page_bytes(i)
        result = self.backend.batch_set_v2(
            [self._transfer(PoolName.MAMBA, keys)]
        )
        self.assertEqual(result[PoolName.MAMBA], [True] * 3)
        # Component writes must not create KV index entries (and vice versa).
        for k in keys:
            self.assertFalse(self.backend.exists(k))
        # The MAMBA component keys carry the ".mamba" suffix in the index.
        suffix = self.backend.config_suffix
        for k in keys:
            self.assertIsNotNone(self.backend._allocator.lookup(k + ".mamba" + suffix))
        self.assertIsNone(self.backend._allocator.lookup(k + suffix))
        # v2 exists reports the KV prefix as absent while the mamba pool
        # transfer sees its component pages.
        result = self.backend.batch_exists_v2(
            keys, [self._transfer(PoolName.MAMBA, keys)]
        )
        self.assertEqual(result.kv_hit_pages, 0)

    def test_v2_mamba_component_survives_restart_checkpoint(self):
        _, mamba = self._register_pools()
        key = self._make_keys(1)[0]
        mamba._pages[0][:] = self._page_bytes(1)
        self.assertEqual(
            self.backend.batch_set_v2([self._transfer(PoolName.MAMBA, [key])]),
            {PoolName.MAMBA: [True]},
        )
        storage_key = self.backend._component_key(PoolName.MAMBA, key)

        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        self.assertTrue(self.backend.checkpoint())
        manifest_entry = next(
            entry
            for entry in read_manifest(self.backend._manifest_path).entries
            if entry.storage_key.decode("utf-8") == storage_key
        )
        self.assertEqual(
            manifest_entry.component,
            component_from_pool_name(PoolName.MAMBA),
        )

    def test_v2_prefix_semantics(self):
        kv, mamba = self._register_pools()
        keys = self._make_keys(5)
        for i in range(3):
            kv._pages[i][:] = self._page_bytes(i)
        for i in (1, 2):
            mamba._pages[i][:] = self._page_bytes(i)
        # KV transfer writes pages 0..2; MAMBA transfer writes components for
        # keys 1..2 only (page 0's component intentionally left absent).
        self.backend.batch_set_v2(
            [
                self._transfer(PoolName.KV, keys[:3]),
                self._transfer(PoolName.MAMBA, keys[1:3]),
            ]
        )
        # kv prefix is 3; mamba components exist for pages 1..2 only.
        all_pages = self.backend.batch_exists_v2(
            keys,
            [self._transfer(PoolName.MAMBA, keys[1:3], hit_policy=PoolHitPolicy.ALL_PAGES)],
        )
        # Missing page-0 component shrinks the usable prefix to 0 (fold
        # semantics: final_pages = min across pools).
        self.assertEqual(all_pages.kv_hit_pages, 0)
        self.assertEqual(all_pages.extra_pool_hit_pages[PoolName.KV], 3)
        # Boundary 0 is not recorded (HiCacheFile fold semantics).
        self.assertEqual(all_pages.extra_pool_hit_pages.get(PoolName.MAMBA, 0), 0)
        trailing = self.backend.batch_exists_v2(
            keys,
            [
                self._transfer(
                    PoolName.MAMBA, keys[1:3], hit_policy=PoolHitPolicy.TRAILING_PAGES
                )
            ],
        )
        # trailing = 2 (transfer.keys len): pages [1..2] all present -> 3.
        self.assertEqual(trailing.kv_hit_pages, 3)
        self.assertEqual(trailing.extra_pool_hit_pages[PoolName.MAMBA], 3)

    def test_v2_indices_length_mismatch(self):
        self._register_pools()
        keys = self._make_keys(3)
        result = self.backend.batch_get_v2(
            [self._transfer(PoolName.KV, keys, indices=[0])]
        )
        self.assertEqual(result[PoolName.KV], [False] * 3)

    def test_v2_trailing_pages_reverse_scan_narrow_cases(self):
        """TRAILING_PAGES reverse-scans from kv_pages for the nearest mamba
        checkpoint; a checkpoint farther back (page 15) still bounds the hit
        even when the query prefix extends past it.

        User-specified regression: a backend that only checked the KV last
        present page would clamp query 0..20 to 0 (no checkpoint at page 20),
        losing the valid page-15 checkpoint that must yield boundary=16.
        """
        kv, mamba = self._register_pools()
        keys = self._make_keys(32)
        for i in range(32):
            kv._pages[i][:] = self._page_bytes(i)
        r = self.backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
        self.assertEqual(r[PoolName.KV], [True] * 32)
        for i in (15, 31):
            mamba._pages[i][:] = self._page_bytes(i)
        r = self.backend.batch_set_v2(
            [self._transfer(PoolName.MAMBA, [keys[15], keys[31]])]
        )
        self.assertEqual(r[PoolName.MAMBA], [True] * 2)

        def query(qlen):
            return self.backend.batch_exists_v2(
                keys[:qlen],
                [
                    self._transfer(
                        PoolName.MAMBA,
                        ["__placeholder__"],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ],
            )

        q20 = query(21)
        self.assertEqual(q20.kv_hit_pages, 16)
        self.assertEqual(q20.extra_pool_hit_pages[PoolName.MAMBA], 16)
        q31 = query(32)
        self.assertEqual(q31.kv_hit_pages, 32)
        self.assertEqual(q31.extra_pool_hit_pages[PoolName.MAMBA], 32)
        q10 = query(11)
        self.assertEqual(q10.kv_hit_pages, 0)
        self.assertNotIn(PoolName.MAMBA, q10.extra_pool_hit_pages)

    def test_v2_trailing_pages_live_shape_storage_hit_positive(self):
        """Live shape (3090 M4 node65): a node's KV fully present with a mamba
        checkpoint at its write-time last page must yield storage_hit > 0 when
        queried by a longer chain sharing the same prefix -- not clamp to 0."""
        # Larger extent to hold the 2048-page live chain (test base is 16Ki).
        backend = ExtentFileBackend(
            _make_config(
                model_name="trailing-live-shape",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "64Mi",
                    "file_size": "512Mi",
                }
            )
        )
        old = self.backend
        self.backend = backend
        try:
            kv = _PagePool(1000, slots=4096)
            mamba = _PagePool(1000, slots=4096)
            backend.register_mem_host_pool_v2(kv, PoolName.KV)
            backend.register_mem_host_pool_v2(mamba, PoolName.MAMBA)
            n = 2048
            keys = self._make_keys(n)
            for i in range(n):
                kv._pages[i][:] = self._page_bytes(i)
            r = backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
            self.assertEqual(r[PoolName.KV], [True] * n)
            mamba._pages[n - 1][:] = self._page_bytes(n - 1)
            r = backend.batch_set_v2(
                [self._transfer(PoolName.MAMBA, [keys[n - 1]])]
            )
            self.assertEqual(r[PoolName.MAMBA], [True])
            # Longer query chain sharing the same 2048-page prefix.
            qkeys = keys + [f"q{i:04d}" for i in range(11427 - n)]
            res = backend.batch_exists_v2(
                qkeys,
                [
                    self._transfer(
                        PoolName.MAMBA,
                        ["__placeholder__"],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ],
            )
            self.assertEqual(res.extra_pool_hit_pages.get(PoolName.KV, 0), n)
            self.assertGreater(res.kv_hit_pages, 0)
            self.assertEqual(res.extra_pool_hit_pages[PoolName.MAMBA], n)
        finally:
            self.backend = old
            backend.close()

    def test_v2_placeholder_keys_never_interpreted_as_all_pages(self):
        """Bug regression (live 3090 M4, storage_hit=0 for every hybrid
        prefetch): a MAMBA PREFETCH transfer carries placeholder keys
        (["__placeholder__"], built by mamba_component) whose presence is
        the *only* signal that the pool's write pattern is TRAILING (node-last
        checkpoints).  If any transfer reconstruction drops `hit_policy`, the
        PoolTransfer default (ALL_PAGES, hicache_storage.py) takes over and
        the boundary scan checks every page of the prefix from 0 -- but mamba
        only stores node-last pages, so the scan fails at the first KV-only
        page and clamps kv_hit to 0.  A placeholder-key transfer must be
        treated as TRAILING regardless of the (possibly defaulted) policy."""
        kv, mamba = self._register_pools()
        keys = self._make_keys(9)
        for i in range(9):
            kv._pages[i][:] = self._page_bytes(i)
        r = self.backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
        self.assertEqual(r[PoolName.KV], [True] * 9)
        # Live shape: two nodes (3 + 6 pages), a mamba checkpoint at each
        # node's last page (chain[2] and chain[8]).
        for i in (2, 8):
            mamba._pages[i][:] = self._page_bytes(i)
        r = self.backend.batch_set_v2(
            [self._transfer(PoolName.MAMBA, [keys[2], keys[8]])]
        )
        self.assertEqual(r[PoolName.MAMBA], [True] * 2)

        # A transfer whose hit_policy was dropped defaults to ALL_PAGES.  The
        # placeholder keys must still route it to the TRAILING scan; reading
        # it as ALL_PAGES produces the live kv_hit=0 signature.
        res = self.backend.batch_exists_v2(
            keys,
            [
                self._transfer(
                    PoolName.MAMBA,
                    ["__placeholder__"],
                    hit_policy=PoolHitPolicy.ALL_PAGES,
                )
            ],
        )
        self.assertGreater(res.kv_hit_pages, 0)
        self.assertEqual(res.extra_pool_hit_pages.get(PoolName.KV, 0), 9)
        self.assertEqual(res.extra_pool_hit_pages.get(PoolName.MAMBA, 0), 9)

    def test_v2_placeholder_keys_all_pages_explicit_still_all_pages(self):
        """Placeholder detection must not swallow an intentionally-ALL_PAGES
        transfer: a non-placeholder (real-key) ALL_PAGES transfer keeps the
        strict every-page boundary (fold semantics).  This pins the negative
        branch so the placeholder special-case stays narrowly scoped."""
        kv, mamba = self._register_pools()
        keys = self._make_keys(9)
        for i in range(9):
            kv._pages[i][:] = self._page_bytes(i)
        r = self.backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
        self.assertEqual(r[PoolName.KV], [True] * 9)
        # Only a trailing checkpoint (node last page), like mamba really writes.
        mamba._pages[8][:] = self._page_bytes(8)
        r = self.backend.batch_set_v2([self._transfer(PoolName.MAMBA, [keys[8]])])
        self.assertEqual(r[PoolName.MAMBA], [True])
        # Real-key ALL_PAGES: page 0's component absent -> boundary 0.
        res = self.backend.batch_exists_v2(
            keys,
            [
                self._transfer(
                    PoolName.MAMBA,
                    list(keys),
                    hit_policy=PoolHitPolicy.ALL_PAGES,
                )
            ],
        )
        self.assertEqual(res.kv_hit_pages, 0)
        self.assertEqual(res.extra_pool_hit_pages.get(PoolName.KV, 0), 9)
        self.assertNotIn(PoolName.MAMBA, res.extra_pool_hit_pages)


class TestExtentV1V2KeyScope(ExtentBackendTestBase):
    """v1 write / v2 query (and reverse) must share the same storage key.

    Bug regression (3090 M3, storage_hit=0 for every prefetch): the engine
    writes pages through v1 batch_set (raw keys, backup_thread) but runs the
    hit query through v2 batch_exists_v2 (hybrid _storage_hit_query), which
    scopes keys with config_suffix (e.g. "_qwen3.5-9b-gptq_0_1").  The v1
    surface used to look up bare keys, so the v2 query never matched anything
    the engine had written.  HiCacheFile applies the suffix on both v1 and v2;
    ExtentFileBackend must too.
    """

    def _engine_style_backend(self) -> ExtentFileBackend:
        # Same HiCacheStorageConfig fields the engine passes, with the
        # Qwen3.5-9B model name of the 3090 repro: suffix "_qwen3.5-9b-gptq_0_1".
        backend = ExtentFileBackend(
            _make_config(
                model_name="qwen3.5-9b-gptq",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        self.assertEqual(backend.config_suffix, "_qwen3.5-9b-gptq_0_1")
        return backend

    def _transfer(self, name, keys, indices=None, hit_policy=None):
        return PoolTransfer(
            name=name,
            host_indices=indices if indices is not None else list(range(len(keys))),
            keys=list(keys),
            hit_policy=hit_policy,
        )

    def test_v1_write_then_v2_query_hits(self):
        # Engine flow: v1 batch_set (backup_thread) then v2 batch_exists_v2
        # (hybrid _storage_hit_query, sidecar pool_transfers present).  The
        # v2 query must find the v1-written pages.
        backend = self._engine_style_backend()
        try:
            keys = self._make_keys(4)
            values = [self._page_bytes(i) for i in range(4)]
            self.assertTrue(backend.batch_set(keys, values))
            self.assertEqual(backend.batch_exists(keys), 4)
            result = backend.batch_exists_v2(keys, pool_transfers=[])
            self.assertEqual(result.kv_hit_pages, 4)
            self.assertEqual(result.extra_pool_hit_pages.get(PoolName.KV), 4)
            # Single-key v1 exists must agree as well.
            self.assertTrue(backend.exists(keys[0]))
        finally:
            backend.close()

    def test_v1_write_then_v2_sidecar_query_hits(self):
        # The engine query additionally carries sidecar pool_transfers; the
        # v2 KV prefix must stay usable when component pages are also present.
        backend = self._engine_style_backend()
        try:
            mamba = _PagePool(1000)
            backend.register_mem_host_pool_v2(mamba, PoolName.MAMBA)
            keys = self._make_keys(3)
            for i in range(3):
                mamba._pages[i][:] = self._page_bytes(i)
            self.assertTrue(backend.batch_set(keys, [self._page_bytes(i) for i in range(3)]))
            self.assertTrue(
                backend.batch_set_v2([self._transfer(PoolName.MAMBA, keys)])[
                    PoolName.MAMBA
                ]
            )
            result = backend.batch_exists_v2(
                keys, [self._transfer(PoolName.MAMBA, keys)]
            )
            self.assertEqual(result.kv_hit_pages, 3)
            self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], 3)
        finally:
            backend.close()

    def test_v1_write_then_v2_get_reads_data(self):
        # Engine restore flow: after the v2 exists query hits, the pages are
        # read back through v2 batch_get_v2 (hybrid read path) into the KV
        # pool.  The v1-written bytes must come back intact.
        backend = self._engine_style_backend()
        try:
            kv = _PagePool(1000)
            backend.register_mem_host_pool_v2(kv, PoolName.KV)
            keys = self._make_keys(3)
            values = [self._page_bytes(i) for i in range(3)]
            self.assertTrue(backend.batch_set(keys, values))
            result = backend.batch_get_v2([self._transfer(PoolName.KV, keys)])
            self.assertEqual(result[PoolName.KV], [True] * 3)
            for i, v in enumerate(values):
                self.assertEqual(bytes(kv._pages[i]), bytes(v))
        finally:
            backend.close()

    def test_v2_write_then_v1_query_hits(self):
        # Reverse direction: v2 batch_set_v2 (sidecar path) writes pages that
        # the v1 surface (exists/batch_exists) must also see.
        backend = self._engine_style_backend()
        try:
            kv = _PagePool(1000)
            backend.register_mem_host_pool_v2(kv, PoolName.KV)
            keys = self._make_keys(3)
            for i in range(3):
                kv._pages[i][:] = self._page_bytes(i)
            result = backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
            self.assertEqual(result[PoolName.KV], [True] * 3)
            self.assertEqual(backend.batch_exists(keys), 3)
            for k in keys:
                self.assertTrue(backend.exists(k))
        finally:
            backend.close()

    def test_v2_write_then_v1_get_reads_data(self):
        # v2-written pages must be readable through the v1 read surface.
        backend = self._engine_style_backend()
        try:
            kv = _PagePool(1000)
            backend.register_mem_host_pool_v2(kv, PoolName.KV)
            keys = self._make_keys(3)
            for i in range(3):
                kv._pages[i][:] = self._page_bytes(i)
            backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
            targets = [bytearray(1000) for _ in keys]
            got = backend.batch_get(keys, targets)
            self._assert_pages_equal(got, [self._page_bytes(i) for i in range(3)])
        finally:
            backend.close()

    def test_cross_path_hits_with_minimal_suffix(self):
        # The minimal suffix ("_" with an empty model name) must behave the
        # same: key scoping must not depend on suffix length or content.
        backend = ExtentFileBackend(
            _make_config(
                model_name="",
                is_mla_model=True,
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        try:
            self.assertEqual(backend.config_suffix, "_")
            keys = self._make_keys(2)
            values = [self._page_bytes(i) for i in range(2)]
            self.assertTrue(backend.batch_set(keys, values))
            self.assertEqual(
                backend.batch_exists_v2(keys, pool_transfers=[]).kv_hit_pages, 2
            )
        finally:
            backend.close()


if torch is not None:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host import base
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost


@unittest.skipIf(torch is None, "torch not installed")
class TestExtentV1ZeroCopy(ExtentBackendTestBase):
    """v1 zero-copy contract against a real page_first MHA host pool.

    The pool is the real ``MHATokenToKVPoolHost``, so the k/v dual-block
    preadv split is exercised against mha.py's actual pointer math and
    buffer memory.  ``host_memory_budget_bytes`` is mocked: this box keeps
    a 10 GiB reserve against ~8 GiB available, so any real pool
    construction would fail the budget check (same mock as
    test_mem_pool_host.py).  The k and v planes carry distinct values, so
    a swapped or shifted scatter corrupts the buffer equality checks.
    """

    PAGE_SIZE = 2
    LAYER_NUM = 2
    HEAD_NUM = 2
    HEAD_DIM = 4
    # 2 planes * L * P * H * D * fp16 itemsize.
    PAGE_BYTES = 2 * LAYER_NUM * PAGE_SIZE * HEAD_NUM * HEAD_DIM * 2

    def setUp(self):
        super().setUp()
        self.assertNotEqual(self.backend.config_suffix, "")
        self._budget_patch = mock.patch.object(
            base,
            "available_host_memory_bytes",
            return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3),
        )
        self._budget_patch.start()
        self.addCleanup(self._budget_patch.stop)
        self.device_pool = MHATokenToKVPool(
            size=self.PAGE_SIZE * 2,
            page_size=self.PAGE_SIZE,
            dtype=torch.float16,
            head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            device="cpu",
            enable_memory_saver=False,
        )
        self.host_pool = MHATokenToKVPoolHost(
            **_host_pool_kwargs(self.device_pool),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.PAGE_SIZE,
            layout="page_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        self.backend.register_mem_pool_host(self.host_pool)

    @property
    def num_pages(self) -> int:
        return self.host_pool.page_num

    def _host_indices(self, num_pages) -> torch.Tensor:
        # Page p anchors at token index p * page_size (page_first layout).
        return torch.arange(num_pages * self.PAGE_SIZE, dtype=torch.int64)

    def _fill_pool(self) -> torch.Tensor:
        """Distinct k/v values per page; returns the expected buffer clone."""
        ps = self.PAGE_SIZE
        for p in range(self.num_pages):
            self.host_pool.kv_buffer[0, p * ps : (p + 1) * ps] = float(p + 1)
            self.host_pool.kv_buffer[1, p * ps : (p + 1) * ps] = float(p + 100)
        return self.host_pool.kv_buffer.clone()

    def _flat_bytes(self, page_idx: int) -> bytes:
        ps = self.PAGE_SIZE
        page = self.host_pool.kv_buffer[:, page_idx * ps : (page_idx + 1) * ps]
        return page.flatten().view(torch.uint8).numpy().tobytes()

    def test_v1_grouped_write_is_invisible_until_commit(self):
        """v1 batches staged into a group stay invisible until it commits."""
        self._fill_pool()
        keys = self._make_keys(self.num_pages)
        indices = self._host_indices(self.num_pages)
        group_id = self.backend.open_group()
        self.assertEqual(
            self.backend.batch_set_v1(keys, indices, group_id=group_id),
            [True] * self.num_pages,
        )
        # Staged, so neither existence nor the allocator index may see it.
        self.assertEqual(self.backend.batch_exists(keys), 0)
        self.assertFalse(self.backend.exists(keys[-1]))
        self.assertTrue(self.backend.commit_group(group_id))
        self.assertEqual(self.backend.batch_exists(keys), self.num_pages)

        # Bytes are the same flat pages the ungrouped v1 write produces.
        targets = [bytearray(self.PAGE_BYTES) for _ in keys]
        got = self.backend.batch_get(keys, targets)
        for i, g in enumerate(got):
            self.assertIsNotNone(g)
            self.assertEqual(bytes(g), self._flat_bytes(i))

    def test_v1_grouped_write_aborted_leaves_nothing(self):
        self._fill_pool()
        keys = self._make_keys(self.num_pages)
        indices = self._host_indices(self.num_pages)
        group_id = self.backend.open_group()
        self.assertEqual(
            self.backend.batch_set_v1(keys, indices, group_id=group_id),
            [True] * self.num_pages,
        )
        self.assertTrue(self.backend.abort_group(group_id))
        self.assertFalse(self.backend.exists(keys[-1]))
        self.assertEqual(self.backend.batch_exists(keys), 0)

    def test_v1_grouped_write_refused_for_foreign_group(self):
        self._fill_pool()
        keys = self._make_keys(self.num_pages)
        self.assertEqual(
            self.backend.batch_set_v1(
                keys, self._host_indices(self.num_pages), group_id=(b"\x01" * 16, 7)
            ),
            [False] * self.num_pages,
        )

    def test_v1_write_then_flat_read_round_trip(self):
        # v1 write gathers the pool's k/v blocks into stored flat pages, and
        # the flat-page read path must return them byte-identical (v1 write
        # -> v2-style read consistency).
        expected = self._fill_pool()
        keys = self._make_keys(self.num_pages)
        results = self.backend.batch_set_v1(keys, self._host_indices(self.num_pages))
        self.assertEqual(results, [True] * self.num_pages)
        self.assertGreater(self.backend.stats_snapshot()["bytes_written"], 0)
        targets = [bytearray(self.PAGE_BYTES) for _ in keys]
        got = self.backend.batch_get(keys, targets)
        for i, g in enumerate(got):
            self.assertIsNotNone(g)
            self.assertEqual(bytes(g), self._flat_bytes(i))
        # Sanity: the pool data actually differs between k and v planes.
        self.assertFalse(
            torch.equal(expected[0], expected[1]),
            "test requires distinct k/v plane data",
        )

    def test_v1_read_restores_pool_after_wipe(self):
        # v1 write -> wipe the pool -> v1 read must restore every page's k
        # and v segments exactly (the dual-block preadv split is exact).
        expected = self._fill_pool()
        keys = self._make_keys(self.num_pages)
        indices = self._host_indices(self.num_pages)
        self.assertTrue(all(self.backend.batch_set_v1(keys, indices)))
        self.host_pool.kv_buffer.zero_()
        results = self.backend.batch_get_v1(keys, indices)
        self.assertEqual(results, [True] * self.num_pages)
        self.assertTrue(
            torch.equal(self.host_pool.kv_buffer, expected),
            "v1 read must restore both k and v segments of every page",
        )

    def test_v1_write_then_v2_query_hits(self):
        # The engine restore hit query (v2 semantics on the KV pool) must see
        # pages written through the v1 zero-copy path.
        self._fill_pool()
        keys = self._make_keys(self.num_pages)
        indices = self._host_indices(self.num_pages)
        self.assertTrue(all(self.backend.batch_set_v1(keys, indices)))
        self.assertEqual(self.backend.batch_exists(keys), self.num_pages)
        result = self.backend.batch_exists_v2(keys, pool_transfers=None)
        self.assertEqual(result.kv_hit_pages, self.num_pages)
        for key in keys:
            self.assertTrue(self.backend.exists(key))

    def test_v2_write_then_v1_read(self):
        # v1 read must consume pages stored by the generic flat-page path
        # (mixed-generation restore: v1-loaded pages followed by v2 pages).
        expected = self._fill_pool()
        keys = self._make_keys(self.num_pages)
        values = [
            self.host_pool.get_data_page(i * self.PAGE_SIZE, flat=True)
            for i in range(self.num_pages)
        ]
        self.assertTrue(self.backend.batch_set(keys, values))
        self.host_pool.kv_buffer.zero_()
        results = self.backend.batch_get_v1(keys, self._host_indices(self.num_pages))
        self.assertEqual(results, [True] * self.num_pages)
        self.assertTrue(torch.equal(self.host_pool.kv_buffer, expected))

    def test_v1_config_suffix_isolation(self):
        # Keys land under key + config_suffix, so a backend with a different
        # suffix (a different model/rank) must not see them, and its own
        # writes must not clobber the first backend's copy.
        self._fill_pool()
        keys = self._make_keys(self.num_pages)
        indices = self._host_indices(self.num_pages)
        self.assertTrue(all(self.backend.batch_set_v1(keys, indices)))
        other = ExtentFileBackend(
            _make_config(
                model_name="othermodel",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        try:
            other.register_mem_pool_host(self.host_pool)
            self.assertEqual(other.batch_get_v1(keys, indices), [False] * self.num_pages)
            self.assertTrue(all(other.batch_set_v1(keys, indices)))
            self.assertEqual(
                self.backend.batch_get_v1(keys, indices), [True] * self.num_pages
            )
        finally:
            other.close()

    def test_v1_indices_length_mismatch_returns_false(self):
        self._fill_pool()
        keys = self._make_keys(3)
        bad = torch.arange(3, dtype=torch.int64)  # 3 != 3 * page_size
        self.assertEqual(self.backend.batch_set_v1(keys, bad), [False] * 3)
        self.assertEqual(self.backend.batch_get_v1(keys, bad), [False] * 3)

    def test_v1_unregistered_pool_returns_false_list(self):
        # No registered host pool: the ABC default would return None and the
        # controller's _page_get_zero_copy would TypeError on it; the extent
        # v1 contract must return a bool list instead.
        backend = ExtentFileBackend(
            _make_config(
                model_name="v1-unregistered",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )
        try:
            keys = self._make_keys(2)
            for call in (
                backend.batch_get_v1(keys, torch.arange(4, dtype=torch.int64)),
                backend.batch_set_v1(keys, torch.arange(4, dtype=torch.int64)),
            ):
                self.assertIsInstance(call, list)
                self.assertEqual(call, [False, False])
        finally:
            backend.close()

    def test_v1_layer_first_refused_without_corruption(self):
        # layer_first interleaves per-layer k/v blocks, which the flat-page
        # sequential scatter would corrupt; v1 must refuse and leave the pool
        # untouched.
        layer_pool = MHATokenToKVPoolHost(
            **_host_pool_kwargs(self.device_pool),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.PAGE_SIZE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        self.backend.register_mem_pool_host(layer_pool)
        layer_pool.kv_buffer.fill_(7)
        before = layer_pool.kv_buffer.clone()
        keys = self._make_keys(3)
        indices = torch.arange(3 * self.PAGE_SIZE, dtype=torch.int64)
        self.assertEqual(self.backend.batch_set_v1(keys, indices), [False] * 3)
        self.assertEqual(self.backend.batch_get_v1(keys, indices), [False] * 3)
        self.assertTrue(torch.equal(layer_pool.kv_buffer, before))

    def test_v1_partial_miss_keeps_other_keys(self):
        # Per-key success semantics: a missing key must not fail its
        # neighbors (the engine stops consuming after the first miss, but
        # earlier keys are already loaded).
        self._fill_pool()
        keys = self._make_keys(self.num_pages)
        self.assertTrue(
            all(self.backend.batch_set_v1(keys, self._host_indices(self.num_pages)))
        )
        results = self.backend.batch_get_v1(
            keys + ["never-written"], self._host_indices(self.num_pages + 1)
        )
        self.assertEqual(results, [True] * self.num_pages + [False])

    def test_v1_read_logs_request_correlated_extent_read_without_keys(self):
        self._fill_pool()
        keys = self._make_keys(self.num_pages)
        indices = self._host_indices(self.num_pages)
        self.assertTrue(all(self.backend.batch_set_v1(keys, indices)))
        with mock.patch(
            "sglang.srt.mem_cache.storage.extent.extent_file_backend.logger.info"
        ) as log_info:
            result = self.backend.batch_get_v1(
                keys,
                indices,
                extra_info=HiCacheStorageExtraInfo(
                    extra_info={"rid": "restore-rid", "attempt_id": "a1", "pool": "kv"}
                ),
            )
        self.assertEqual(result, [True] * self.num_pages)
        args = next(
            call.args
            for call in log_info.call_args_list
            if "evt=extent_read" in call.args[0]
        )
        self.assertIn("rid=%s", args[0])
        self.assertIn("requested_bytes=%d", args[0])
        self.assertIn("physical_bytes=%d", args[0])
        self.assertIn("restore-rid", args)
        self.assertIn("a1", args)
        self.assertNotIn(keys[0], " ".join(map(str, args)))

    def test_v1_read_one_pread_per_extent(self):
        backend = ExtentFileBackend(
            _make_config(
                model_name="full-file",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "128Ki",
                    "file_size": "1Mi",
                }
            )
        )
        try:
            backend.register_mem_pool_host(self.host_pool)
            self._fill_pool()
            keys = self._make_keys(self.num_pages)
            indices = self._host_indices(self.num_pages)
            self.assertTrue(all(backend.batch_set_v1(keys, indices)))
            preads_before = backend.stats_snapshot()["preads"]
            with mock.patch.object(os, "preadv", wraps=os.preadv) as spy:
                results = backend.batch_get_v1(keys, indices)
            self.assertEqual(results, [True] * self.num_pages)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(backend.stats_snapshot()["preads"] - preads_before, 1)
        finally:
            backend.close()

    def test_v1_gap_range_merges_with_scratch(self):
        # Force a <=4KiB gap between two pages in one extent (rewriting A
        # moves its ref past B's end), then read both: a single preadv
        # covers the merged range and the dead gap bytes must not shift the
        # scatter into the wrong host blocks.
        self._fill_pool()
        keys = self._make_keys(3)
        indices = self._host_indices(3)
        self.assertTrue(all(self.backend.batch_set_v1(keys, indices)))
        # Rewrite key 0: its new ref lands after key 2 (cursor advanced).
        self.assertTrue(all(self.backend.batch_set_v1([keys[0]], indices[:2])))
        fill = self.host_pool.kv_buffer.clone()
        self.host_pool.kv_buffer.zero_()
        with mock.patch.object(os, "preadv", wraps=os.preadv) as spy:
            results = self.backend.batch_get_v1(
                [keys[1], keys[0]], self._host_indices(2)
            )
        self.assertEqual(results, [True, True])
        self.assertEqual(spy.call_count, 1)
        # The read targets anchor 0 (dim1 [0,2)) with keys[1]'s data (stored
        # from dim1 [2,4)) and anchor 2 (dim1 [2,4)) with keys[0]'s data
        # (stored from dim1 [0,2)); everything else stays zero.
        ps = self.PAGE_SIZE
        expected = torch.zeros_like(fill)
        expected[:, 0:ps] = fill[:, ps : 2 * ps]
        expected[:, ps : 2 * ps] = fill[:, 0:ps]
        self.assertTrue(torch.equal(self.host_pool.kv_buffer, expected))

    def test_v1_read_write_stats_counters(self):
        self._fill_pool()
        keys = self._make_keys(2)
        indices = self._host_indices(2)
        s0 = self.backend.stats_snapshot()
        self.assertTrue(all(self.backend.batch_set_v1(keys, indices)))
        s1 = self.backend.stats_snapshot()
        self.assertEqual(s1["bytes_written"] - s0["bytes_written"], 2 * self.PAGE_BYTES)
        self.assertEqual(s1["pwrites"] - s0["pwrites"], 1)
        s2 = self.backend.stats_snapshot()
        self.assertTrue(all(self.backend.batch_get_v1(keys, indices)))
        s3 = self.backend.stats_snapshot()
        self.assertEqual(s3["bytes_read"] - s2["bytes_read"], 2 * self.PAGE_BYTES)
        self.assertEqual(s3["preads"] - s2["preads"], 1)
        self.assertEqual(
            s3["last_batch_requested_bytes"], 2 * self.PAGE_BYTES
        )
        self.assertEqual(s3["last_batch_physical_bytes"], 2 * self.PAGE_BYTES)

    def test_v1_stored_page_length_mismatch_refused(self):
        # A page stored under a different byte length (layout drift) must not
        # be scattered into the pool; it returns as a miss.
        self._fill_pool()
        key = self._make_keys(1)[0]
        self.assertTrue(self.backend.set(key, bytearray(64)))  # 64 != PAGE_BYTES
        self.assertEqual(
            self.backend.batch_get_v1(
                [key], torch.arange(self.PAGE_SIZE, dtype=torch.int64)
            ),
            [False],
        )

    def _big_pool_backend(self, num_pages):
        """Fresh backend + host pool with room for ``num_pages`` v1 pages.

        The setUp pool holds only ``page_num`` pages (a few), far below
        the >512 pages needed to cross the IOV_MAX boundary with 2 blocks
        per page; the budget mock from setUp is still active, so the
        second pool construction passes the host-memory check.
        """
        backend = ExtentFileBackend(
            _make_config(
                model_name="large-iov",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "256Ki",
                    "file_size": "1Mi",
                }
            )
        )
        device_pool = MHATokenToKVPool(
            size=num_pages * self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=torch.float16,
            head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            device="cpu",
            enable_memory_saver=False,
        )
        host_pool = MHATokenToKVPoolHost(
            **_host_pool_kwargs(device_pool),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.PAGE_SIZE,
            layout="page_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        self.assertGreaterEqual(host_pool.page_num, num_pages)
        backend.register_mem_pool_host(host_pool)
        return backend, host_pool

    def _fill_big_pool(self, host_pool, num_pages) -> torch.Tensor:
        ps = self.PAGE_SIZE
        for p in range(num_pages):
            host_pool.kv_buffer[0, p * ps : (p + 1) * ps] = float(p + 1)
            host_pool.kv_buffer[1, p * ps : (p + 1) * ps] = float(p + 100)
        return host_pool.kv_buffer.clone()

    def test_v1_large_batch_over_iov_max_shards_and_round_trips(self):
        # 600 pages x 2 blocks = 1,200 iovecs > IOV_MAX: batch_set_v1 must
        # split the pwritev (1024 + 176) and batch_get_v1 the preadv the
        # same way, and every page must survive byte-identical.  Bug
        # regression (3090 M4): a single 2,048-iovec pwritev failed with
        # EINVAL, so 1,495 backup batches never landed.
        num_pages = 600
        backend, host_pool = self._big_pool_backend(num_pages)
        try:
            expected = self._fill_big_pool(host_pool, num_pages)
            keys = self._make_keys(num_pages)
            indices = torch.arange(num_pages * self.PAGE_SIZE, dtype=torch.int64)
            with mock.patch.object(os, "pwritev", wraps=os.pwritev) as spy:
                results = backend.batch_set_v1(keys, indices)
            self.assertEqual(results, [True] * num_pages)
            self.assertEqual(spy.call_count, 2)  # 1024 + 176 iovecs
            host_pool.kv_buffer.zero_()
            with mock.patch.object(os, "preadv", wraps=os.preadv) as spy:
                results = backend.batch_get_v1(keys, indices)
            self.assertEqual(results, [True] * num_pages)
            self.assertEqual(spy.call_count, 2)  # read-side cap: 1024 + 176
            self.assertTrue(torch.equal(host_pool.kv_buffer, expected))
        finally:
            backend.close()

    def test_v1_write_exactly_iov_max_stays_single_syscall(self):
        # 512 pages x 2 blocks = exactly 1,024 iovecs: the shard cap must
        # not split below the Linux limit (a lowered cap would show up as
        # a spurious second pwritev here), and the batch must round-trip.
        num_pages = 512
        backend, host_pool = self._big_pool_backend(num_pages)
        try:
            expected = self._fill_big_pool(host_pool, num_pages)
            keys = self._make_keys(num_pages)
            indices = torch.arange(num_pages * self.PAGE_SIZE, dtype=torch.int64)
            with mock.patch.object(os, "pwritev", wraps=os.pwritev) as spy:
                results = backend.batch_set_v1(keys, indices)
            self.assertEqual(results, [True] * num_pages)
            self.assertEqual(spy.call_count, 1)
            host_pool.kv_buffer.zero_()
            self.assertTrue(all(backend.batch_get_v1(keys, indices)))
            self.assertTrue(torch.equal(host_pool.kv_buffer, expected))
        finally:
            backend.close()


@unittest.skipIf(torch is None, "torch not installed")
class TestExtentClosedLoop2048(ExtentBackendTestBase):
    """2048-page closed loop across the STORAGE_BATCH_SIZE(1024) boundary.

    Bug-regression pin (3090 M4 live: storage_hit=0 despite 5.7GB persisted).
    The live engine writes a 2048-page node as two batch_set_v1(1024) calls
    (_page_backup splits at STORAGE_BATCH_SIZE=1024) and reads back with
    batch_exists_v2.  This test drives the exact same split through BOTH the
    v1 and v2 surfaces and asserts: index registers all 2048, batch_exists
    returns 2048, and batch_get byte-compares identical for every page.

    A future diff that breaks second-batch commit, the prefix scan, or the
    read path across an extent/batch boundary must turn this test red.
    """

    PAGE_SIZE = 2
    LAYER_NUM = 2
    HEAD_NUM = 2
    HEAD_DIM = 4
    PAGE_BYTES = 2 * LAYER_NUM * PAGE_SIZE * HEAD_NUM * HEAD_DIM * 2
    N_PAGES = 2048
    BATCH = 1024

    def _large_pool_backend(self):
        """Backend with a host pool holding >= 2048 pages (device pool
        sized at N_PAGES*PAGE_SIZE tokens, ratio 2.0 -> > 2048 host pages)."""
        budget_patch = mock.patch.object(
            base,
            "available_host_memory_bytes",
            return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3),
        )
        budget_patch.start()
        self.addCleanup(budget_patch.stop)
        backend = ExtentFileBackend(
            _make_config(
                model_name="closed-loop-2048",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "64Mi",
                    "file_size": "512Mi",
                }
            )
        )
        device_pool = MHATokenToKVPool(
            size=self.N_PAGES * self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=torch.float16,
            head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            device="cpu",
            enable_memory_saver=False,
        )
        host_pool = MHATokenToKVPoolHost(
            **_host_pool_kwargs(device_pool),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.PAGE_SIZE,
            layout="page_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        backend.register_mem_pool_host(host_pool)
        backend.register_mem_host_pool_v2(host_pool, PoolName.KV)
        self.addCleanup(backend.close)
        return backend, host_pool

    def _fill_pool(self, host_pool, n_pages):
        ps = self.PAGE_SIZE
        for p in range(n_pages):
            host_pool.kv_buffer[0, p * ps : (p + 1) * ps] = float(p + 1)
            host_pool.kv_buffer[1, p * ps : (p + 1) * ps] = float(p + 100)
        return host_pool.kv_buffer.clone()

    def _flat_bytes(self, host_pool, page_idx):
        ps = self.PAGE_SIZE
        page = host_pool.kv_buffer[:, page_idx * ps : (page_idx + 1) * ps]
        return page.flatten().view(torch.uint8).numpy().tobytes()

    def _host_indices(self, n_pages, offset_pages=0):
        return torch.arange(
            offset_pages * self.PAGE_SIZE,
            (offset_pages + n_pages) * self.PAGE_SIZE,
            dtype=torch.int64,
        )

    def test_v1_2048_cross_batch_boundary_round_trip(self):
        # Live engine write pattern: 2048 pages in two batch_set_v1(1024).
        backend, host_pool = self._large_pool_backend()
        expected = self._fill_pool(host_pool, self.N_PAGES)
        keys_all = self._make_keys(self.N_PAGES)
        batch1, batch2 = keys_all[: self.BATCH], keys_all[self.BATCH :]
        self.assertEqual(
            backend.batch_set_v1(batch1, self._host_indices(self.BATCH, 0)),
            [True] * self.BATCH,
        )
        self.assertEqual(
            backend.batch_set_v1(
                batch2, self._host_indices(self.BATCH, self.BATCH)
            ),
            [True] * self.BATCH,
        )
        # Index must hold all 2048, batch_exists must return 2048.
        self.assertEqual(backend.batch_exists(keys_all), self.N_PAGES)
        with backend._lock:
            bitmap = [
                backend._allocator.lookup(backend._storage_key(k)) is not None
                for k in keys_all
            ]
        self.assertEqual(sum(bitmap), self.N_PAGES)
        # Read back full span and byte-compare every page.
        self.assertEqual(
            backend.batch_get_v1(keys_all, self._host_indices(self.N_PAGES, 0)),
            [True] * self.N_PAGES,
        )
        for i in range(self.N_PAGES):
            self.assertEqual(
                self._flat_bytes(host_pool, i),
                expected[:, i * self.PAGE_SIZE : (i + 1) * self.PAGE_SIZE]
                .flatten()
                .view(torch.uint8)
                .numpy()
                .tobytes(),
                f"page {i} byte mismatch",
            )

    def test_v2_2048_cross_batch_boundary_round_trip(self):
        # v2 surface (live read path) across the same 1024-batch split.
        backend, host_pool = self._large_pool_backend()
        expected = self._fill_pool(host_pool, self.N_PAGES)
        keys_all = self._make_keys(self.N_PAGES)
        batch1, batch2 = keys_all[: self.BATCH], keys_all[self.BATCH :]

        def transfer(keys, indices):
            return PoolTransfer(
                name=PoolName.KV,
                host_indices=indices,
                keys=list(keys),
                hit_policy=None,
            )

        self.assertEqual(
            backend.batch_set_v2([transfer(batch1, self._host_indices(self.BATCH, 0))])[
                PoolName.KV
            ],
            [True] * self.BATCH,
        )
        self.assertEqual(
            backend.batch_set_v2(
                [transfer(batch2, self._host_indices(self.BATCH, self.BATCH))]
            )[PoolName.KV],
            [True] * self.BATCH,
        )
        res = backend.batch_exists_v2(keys_all, pool_transfers=[])
        self.assertEqual(res.kv_hit_pages, self.N_PAGES)
        got = backend.batch_get_v2(
            [transfer(keys_all, self._host_indices(self.N_PAGES, 0))]
        )[PoolName.KV]
        self.assertEqual(got, [True] * self.N_PAGES)
        for i in range(self.N_PAGES):
            self.assertEqual(
                self._flat_bytes(host_pool, i),
                expected[:, i * self.PAGE_SIZE : (i + 1) * self.PAGE_SIZE]
                .flatten()
                .view(torch.uint8)
                .numpy()
                .tobytes(),
                f"v2 page {i} byte mismatch",
            )


class TestExtentIovMaxBoundary(ExtentBackendTestBase):
    """pwritev/preadv sharding at the Linux IOV_MAX(1024) iovec limit.

    Bug regression (3090 M4, 1,495 backup failures): batch_set_v1 packs 2
    blocks per page, so a 1024-page batch reaches 2,048 iovecs and one
    pwritev fails with EINVAL (measured on the 3090 box and locally: 1,024
    iovecs OK / 1,025 EINVAL / 2,048 EINVAL).  The IO helpers must split
    past 1,024 iovecs per syscall; these cases pin the raw-helper boundary
    at odd iovec counts that 2-block pool geometry cannot produce.
    """

    def test_pwritev_over_iov_max_shards_and_round_trips(self):
        n = 1025  # one past the measured 1,024-OK boundary
        chunk = 7
        buffers = [bytearray(bytes([i % 251]) * chunk) for i in range(n)]
        with mock.patch.object(os, "pwritev", wraps=os.pwritev) as spy:
            self.assertTrue(self.backend._pwritev(0, buffers))
        self.assertEqual(spy.call_count, 2)  # 1024 + 1
        out = [bytearray(chunk) for _ in range(n)]
        with mock.patch.object(os, "preadv", wraps=os.preadv) as spy:
            got = self.backend._preadv_full(0, out)
        self.assertEqual(got, n * chunk)
        self.assertEqual(spy.call_count, 2)  # read-side cap: 1024 + 1
        for i, b in enumerate(out):
            self.assertEqual(bytes(b), buffers[i])

    def test_pwritev_at_iov_max_stays_single_syscall(self):
        # Exactly 1,024 iovecs must remain one syscall: a shard cap lowered
        # below IOV_MAX would show up here as a spurious extra call.
        n = 1024
        buffers = [bytearray(bytes([i % 251]) * 7) for i in range(n)]
        with mock.patch.object(os, "pwritev", wraps=os.pwritev) as spy:
            self.assertTrue(self.backend._pwritev(0, buffers))
        self.assertEqual(spy.call_count, 1)
        out = [bytearray(7) for _ in range(n)]
        self.assertEqual(self.backend._preadv_full(0, out), n * 7)


class TestExtentReadAmplification(ExtentBackendTestBase):
    """Exact-range reads: physical bytes read must track requested bytes.

    Bug regression (3090 M3, 30x read amplification): the old read path
    issued one pread per extent covering [min_offset .. max_offset+length)
    over the batch's requested refs, so small keys sparsely scattered inside
    a shared 64 MiB extent pulled every interleaved page of other chains from
    disk -- 93 restores requested ~335 MB and physically read ~10 GB.  The
    read path must sort requested refs by offset, merge only adjacent ranges
    (gap <= 4 KiB), and pread each merged range exactly.
    """

    def _transfer(self, name, keys, indices=None, hit_policy=None):
        return PoolTransfer(
            name=name,
            host_indices=indices if indices is not None else list(range(len(keys))),
            keys=list(keys),
            hit_policy=hit_policy,
        )

    def test_sparse_keys_in_large_extent_read_exact_ranges(self):
        # 100 x 4 KiB pages inside one 64 MiB extent; request every 25th page
        # (100 KiB gaps).  The old span read would pull 400 KiB for a 16 KiB
        # request (25x); exact-range reads must pull exactly the 4 pages.
        backend = ExtentFileBackend(
            _make_config(
                model_name="sparse-read",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "64Mi",
                    "file_size": "64Mi",
                }
            )
        )
        try:
            page = 4096
            keys = self._make_keys(100)
            values = [self._page_bytes(i, page) for i in range(100)]
            self.assertTrue(backend.batch_set(keys, values))
            want = [0, 25, 50, 75]
            targets = [bytearray(page) for _ in want]
            preads_before = backend.stats_snapshot()["preads"]
            got = backend.batch_get([keys[i] for i in want], targets)
            self._assert_pages_equal(got, [values[i] for i in want])
            requested = len(want) * page
            stats = backend.stats_snapshot()
            self.assertEqual(stats["preads"] - preads_before, len(want))
            self.assertEqual(stats["last_batch_requested_bytes"], requested)
            self.assertEqual(stats["last_batch_physical_bytes"], requested)
            self.assertLessEqual(
                stats["physical_bytes_read"] / stats["requested_bytes"], 1.2
            )
        finally:
            backend.close()

    def test_contiguous_ranges_merge_into_one_pread(self):
        # Packed keys are contiguous, so the whole batch is one exact range:
        # one pread, physical == requested (no dead bytes either way).
        keys = self._write_keys(10)  # 1000 B each, packed back-to-back
        targets = [bytearray(1000) for _ in keys]
        preads_before = self.backend.stats_snapshot()["preads"]
        with mock.patch.object(os, "preadv", wraps=os.preadv) as spy:
            got = self.backend.batch_get(keys, targets)
        self.assertEqual(spy.call_count, 1)
        stats = self.backend.stats_snapshot()
        self.assertEqual(stats["preads"] - preads_before, 1)
        self.assertEqual(stats["last_batch_requested_bytes"], 10 * 1000)
        self.assertEqual(stats["last_batch_physical_bytes"], 10 * 1000)
        self._assert_pages_equal(got, [self._page_bytes(i) for i in range(10)])

    def test_disjoint_ranges_not_merged(self):
        # k0 [0,1000), filler [1000,6000), k1 [6000,7000): the 5 KiB dead gap
        # must not be absorbed -- two exact preads, 2000 physical bytes.
        self.assertTrue(
            self.backend.batch_set(
                ["k0", "filler", "k1"],
                [self._page_bytes(1), bytearray(b"F" * 5000), self._page_bytes(3)],
            )
        )
        targets = [bytearray(1000), bytearray(1000)]
        preads_before = self.backend.stats_snapshot()["preads"]
        got = self.backend.batch_get(["k0", "k1"], targets)
        self._assert_pages_equal(got, [self._page_bytes(1), self._page_bytes(3)])
        stats = self.backend.stats_snapshot()
        self.assertEqual(stats["preads"] - preads_before, 2)
        self.assertEqual(stats["last_batch_requested_bytes"], 2000)
        self.assertEqual(stats["last_batch_physical_bytes"], 2000)

    def test_merge_threshold_boundary_at_4kib(self):
        # The merge rule is gap <= 4 KiB: a filler of exactly 4096 B merges
        # (physical = requested + 4096), one of 4097 B does not (physical =
        # requested).  Pin the boundary so a threshold change is a conscious
        # diff, not an accidental one.
        for filler_len, merged in ((4096, True), (4097, False)):
            self.backend.clear()
            self.assertTrue(
                self.backend.batch_set(
                    ["k0", "filler", "k1"],
                    [
                        self._page_bytes(1),
                        bytearray(b"F" * filler_len),
                        self._page_bytes(3),
                    ],
                )
            )
            targets = [bytearray(1000), bytearray(1000)]
            got = self.backend.batch_get(["k0", "k1"], targets)
            self._assert_pages_equal(got, [self._page_bytes(1), self._page_bytes(3)])
            stats = self.backend.stats_snapshot()
            if merged:
                self.assertEqual(
                    stats["last_batch_physical_bytes"], 2000 + filler_len
                )
                self.assertEqual(stats["last_batch_preads"], 1)
            else:
                self.assertEqual(stats["last_batch_physical_bytes"], 2000)
                self.assertEqual(stats["last_batch_preads"], 2)

    def test_v2_read_path_counts_exact_ranges(self):
        # The v2 restore path (batch_get_v2) must apply the same exact-range
        # reads: request keys 0 and 7 of 10 written pages (6 KiB gap > 4 KiB),
        # physical == 2000 B == requested.
        kv = _PagePool(1000)
        self.backend.register_mem_host_pool_v2(kv, PoolName.KV)
        keys = self._make_keys(10)
        values = [self._page_bytes(i) for i in range(10)]
        for i, v in enumerate(values):
            kv._pages[i][:] = v
        self.assertTrue(
            self.backend.batch_set_v2([self._transfer(PoolName.KV, keys)])[
                PoolName.KV
            ][0]
        )
        result = self.backend.batch_get_v2(
            [self._transfer(PoolName.KV, [keys[0], keys[7]])]
        )
        self.assertEqual(result[PoolName.KV], [True, True])
        self.assertEqual(bytes(kv._pages[0]), bytes(values[0]))
        self.assertEqual(bytes(kv._pages[1]), bytes(values[7]))
        stats = self.backend.stats_snapshot()
        self.assertEqual(stats["last_batch_requested_bytes"], 2000)
        self.assertEqual(stats["last_batch_physical_bytes"], 2000)

    def test_counters_accumulate_with_per_batch_latest(self):
        # Cumulative counters sum across batches; last_batch_* hold the most
        # recent _read_pages call (single-key get and batch_get both count).
        self.assertTrue(self.backend.set("k0", self._page_bytes(1)))
        self.assertTrue(self.backend.set("k1", self._page_bytes(2)))
        self.backend.get("k0", bytearray(1000))
        stats = self.backend.stats_snapshot()
        self.assertEqual(stats["requested_bytes"], 1000)
        self.assertEqual(stats["physical_bytes_read"], 1000)
        self.assertEqual(stats["last_batch_requested_bytes"], 1000)
        got = self.backend.batch_get(["k0", "k1"], [bytearray(1000), bytearray(1000)])
        self.assertIsNotNone(got[0])
        stats = self.backend.stats_snapshot()
        self.assertEqual(stats["requested_bytes"], 3000)  # 1000 + 2000
        self.assertEqual(stats["physical_bytes_read"], 3000)
        self.assertEqual(stats["last_batch_requested_bytes"], 2000)
        self.assertEqual(stats["last_batch_physical_bytes"], 2000)

    def test_prefetch_bytes_read_stays_zero(self):
        # This backend has no prefetch logic: every pread is request-driven,
        # so the separate prefetch accounting must stay 0 after any reads.
        keys = self._write_keys(5)
        targets = [bytearray(1000) for _ in keys]
        self.backend.batch_get(keys, targets)
        self.backend.get(keys[0], bytearray(1000))
        stats = self.backend.stats_snapshot()
        self.assertEqual(stats["prefetch_bytes_read"], 0)
        self.assertGreater(stats["physical_bytes_read"], 0)


class _TensorPagePool:
    """Host-pool stand-in backed by real CPU tensors (engine flat pages)."""

    def __init__(self, page_bytes: int, slots: int = 256, dtype=None):
        self.page_size = 1
        self.dtype = dtype if dtype is not None else torch.bfloat16
        numel = page_bytes // torch.empty(0, dtype=self.dtype).element_size()
        self._pages = [torch.zeros(numel, dtype=self.dtype) for _ in range(slots)]

    def get_data_page(self, index, flat=True):
        return self._pages[index]

    def get_dummy_flat_data_page(self):
        return torch.zeros_like(self._pages[0])

    def set_from_flat_data_page(self, index, data_page):
        self._pages[index].copy_(data_page)


class TestExtentTensorPages(ExtentBackendTestBase):
    """Real torch.Tensor page round-trips (bf16/fp16/fp32 CPU).

    Bug regression (3090 M2 crash): ``_page_byte_view`` called ``.numpy()``
    on the engine's bf16 CPU flat pages, which raises
    ``TypeError: Got unsupported ScalarType BFloat16``; backup_thread died on
    its first batch_set, storage never filled, and restore served nothing.
    The byte view must be a bit-level uint8 reinterpretation (valid for any
    CPU dtype), and the read-back fill must restore the same dtype and bit
    pattern.  Requires torch; skips itself when torch is unavailable.
    """

    def _torch_or_skip(self):
        if torch is None:
            self.skipTest("torch not available")
        return torch

    def _transfer(self, name, keys, indices=None, hit_policy=None):
        return PoolTransfer(
            name=name,
            host_indices=indices if indices is not None else list(range(len(keys))),
            keys=list(keys),
            hit_policy=hit_policy,
        )

    def _assert_bit_equal(self, got, expected, torch):
        # dtype must survive the round trip, values must be numerically
        # equal, and the uint8 views must match byte for byte (contiguous()
        # first: view(dtype) needs a unit last stride).
        self.assertEqual(got.dtype, expected.dtype)
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(
            torch.equal(
                got.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def test_bf16_tensor_round_trip(self):
        torch_ = self._torch_or_skip()
        t = torch_.randn(4096, dtype=torch_.bfloat16)
        self.assertTrue(self.backend.set("tk", t))
        got = self.backend.get("tk", torch_.empty_like(t))
        self.assertIsNotNone(got)
        self._assert_bit_equal(got, t, torch_)

    def test_fp16_tensor_round_trip(self):
        torch_ = self._torch_or_skip()
        t = torch_.randn(4096, dtype=torch_.float16)
        self.assertTrue(self.backend.set("tk", t))
        got = self.backend.get("tk", torch_.empty_like(t))
        self.assertIsNotNone(got)
        self._assert_bit_equal(got, t, torch_)

    def test_fp32_tensor_round_trip(self):
        torch_ = self._torch_or_skip()
        t = torch_.randn(4096, dtype=torch_.float32)
        self.assertTrue(self.backend.set("tk", t))
        got = self.backend.get("tk", torch_.empty_like(t))
        self.assertIsNotNone(got)
        self._assert_bit_equal(got, t, torch_)

    def test_batch_mixed_dtype_tensors_round_trip(self):
        torch_ = self._torch_or_skip()
        keys = self._make_keys(9)
        values = [
            torch_.randn(64, dtype=dtype)
            for dtype in (
                torch_.bfloat16,
                torch_.float16,
                torch_.float32,
            ) * 3
        ]
        self.assertTrue(self.backend.batch_set(keys, values))
        targets = [torch_.empty_like(v) for v in values]
        got = self.backend.batch_get(keys, targets)
        self.assertEqual(len(got), len(values))
        for g, v in zip(got, values):
            self.assertIsNotNone(g)
            self._assert_bit_equal(g, v, torch_)

    def test_non_contiguous_tensor_round_trip(self):
        torch_ = self._torch_or_skip()
        # Write side: a strided (non-contiguous) page must be read by its
        # logical element order (contiguous() copies, then uint8 view).
        t = torch_.randn(8, 16, dtype=torch_.bfloat16).t()  # (16, 8), strided
        self.assertTrue(self.backend.set("nct", t))
        # Read back into a contiguous target (the engine's flat pages are
        # contiguous; a strided target cannot be filled in place).
        got = self.backend.get(
            "nct", torch_.empty(16, 8, dtype=torch_.bfloat16)
        )
        self.assertIsNotNone(got)
        self._assert_bit_equal(got, t, torch_)

    def test_tensor_page_view_is_zero_copy(self):
        torch_ = self._torch_or_skip()
        t = torch_.randn(4096, dtype=torch_.bfloat16)
        view = _page_byte_view(t)
        self.assertEqual(view.nbytes, t.numel() * t.element_size())
        # The byte view must alias the tensor storage, not copy it: the
        # exported numpy buffer and the tensor data pointer must agree
        # (a 20K-page batch would double memory if copied).
        self.assertEqual(
            view.obj.__array_interface__["data"][0], t.data_ptr()
        )

    def test_v2_bf16_tensor_pool_round_trip(self):
        torch_ = self._torch_or_skip()
        kv = _TensorPagePool(2000, dtype=torch_.bfloat16)
        self.backend.register_mem_host_pool_v2(kv, PoolName.KV)
        keys = self._make_keys(4)
        originals = [torch_.randn(1000, dtype=torch_.bfloat16) for _ in range(4)]
        for i, o in enumerate(originals):
            kv._pages[i][:] = o
        result = self.backend.batch_set_v2([self._transfer(PoolName.KV, keys)])
        self.assertEqual(result[PoolName.KV], [True] * 4)
        # Wipe the pages so a silent miss is not masked by stale bytes.
        for p in kv._pages[:4]:
            p.zero_()
        result = self.backend.batch_get_v2([self._transfer(PoolName.KV, keys)])
        self.assertEqual(result[PoolName.KV], [True] * 4)
        for i, o in enumerate(originals):
            self._assert_bit_equal(kv._pages[i], o, torch_)


class TestExtentV2TensorGroupCommitCrash(ExtentBackendTestBase):
    """GROUP_COMMIT temporal barrier over real torch.Tensor KV + Mamba pools.

    The bytearray ``_PagePool`` v2 cases prove group staging/crash semantics
    and ``TestExtentTensorPages`` proves real-tensor round-trips, but no case
    drives BOTH real-tensor components through one staged group and then omits
    the commit.  The crash case is the load-bearing negative: a staged
    two-member group with durable SET/BATCH_COMMIT frames but no durable
    GROUP_COMMIT must leave the KV *and* the Mamba member absent after reopen.
    The committed sibling is the positive control on the identical fixture
    (same two components, same crash+reopen): it recovers both members
    bit-identically through ``batch_get_v2``, so a failing crash case cannot be
    explained by broken tensor plumbing.  Requires torch; skips itself when
    torch is unavailable (``_torch_or_skip`` pattern).
    """

    # Reuse the existing bit-level oracle (dtype + numeric + uint8-view check).
    _assert_bit_equal = TestExtentTensorPages._assert_bit_equal

    def _torch_or_skip(self):
        if torch is None:
            self.skipTest("torch not available")
        return torch

    def _transfer(self, name, keys, indices=None, hit_policy=None):
        return PoolTransfer(
            name=name,
            host_indices=indices if indices is not None else list(range(len(keys))),
            keys=list(keys),
            hit_policy=hit_policy,
        )

    def _register_tensor_pools(self, torch_, page_bytes=2000):
        kv = _TensorPagePool(page_bytes, dtype=torch_.bfloat16)
        mamba = _TensorPagePool(page_bytes, dtype=torch_.bfloat16)
        self.backend.register_mem_host_pool_v2(kv, PoolName.KV)
        self.backend.register_mem_host_pool_v2(mamba, PoolName.MAMBA)
        return kv, mamba

    def _stage_two_component_group(
        self, kv, mamba, kv_key, mamba_key, kv_original, mamba_original
    ):
        kv._pages[0][:] = kv_original
        mamba._pages[0][:] = mamba_original
        group_id = self.backend.open_group()
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.KV, [kv_key])], group_id=group_id
            ),
            {PoolName.KV: [True]},
        )
        self.assertEqual(
            self.backend.batch_set_v2(
                [self._transfer(PoolName.MAMBA, [mamba_key])], group_id=group_id
            ),
            {PoolName.MAMBA: [True]},
        )
        return group_id

    def _crash_close_and_reopen(self):
        # Crash close: preserve the durable WAL prefix and skip the normal
        # close-time checkpoint/abort path (mirrors TestExtentV1 crash cases).
        self.backend._constructor_complete = False
        self.backend.close()
        self.backend = ExtentFileBackend(
            _make_config(
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                }
            )
        )

    def test_tensor_group_crash_before_commit_leaves_kv_and_mamba_absent(self):
        torch_ = self._torch_or_skip()
        kv, mamba = self._register_tensor_pools(torch_)
        kv_key, mamba_key = self._make_keys(2)
        kv_original = torch_.randn(1000, dtype=torch_.bfloat16)
        mamba_original = torch_.randn(1000, dtype=torch_.bfloat16)
        self._stage_two_component_group(
            kv, mamba, kv_key, mamba_key, kv_original, mamba_original
        )

        kv_storage_key = self.backend._storage_key(kv_key)
        mamba_storage_key = self.backend._component_key(PoolName.MAMBA, mamba_key)
        # Staged group is invisible before its GROUP_COMMIT frame.
        self.assertFalse(self.backend.exists(kv_key))
        self.assertFalse(self.backend.exists(mamba_key))
        self.assertIsNone(self.backend._allocator.lookup(kv_storage_key))
        self.assertIsNone(self.backend._allocator.lookup(mamba_storage_key))
        # Non-vacuity: both members' batches are already durable on disk
        # (SET+BATCH_COMMIT per member) and only the GROUP_COMMIT marker is
        # absent, so recovery drops real members rather than an empty prefix.
        frames = iter_segment_frames(
            segment_path(
                self.tmpdir, self.backend.config_suffix, self.backend._wal_generation
            ),
            active=True,
        )
        frame_types = [frame.frame_type for frame in frames]
        self.assertEqual(frame_types.count(FRAME_BATCH_COMMIT), 2)
        self.assertNotIn(FRAME_GROUP_COMMIT, frame_types)

        self._crash_close_and_reopen()
        # No durable GROUP_COMMIT: recovery must drop BOTH components.
        self.assertFalse(self.backend.exists(kv_key))
        self.assertFalse(self.backend.exists(mamba_key))
        self.assertIsNone(self.backend._allocator.lookup(kv_storage_key))
        self.assertIsNone(self.backend._allocator.lookup(mamba_storage_key))

    def test_tensor_group_commit_survives_crash_reopen_bit_identically(self):
        torch_ = self._torch_or_skip()
        kv, mamba = self._register_tensor_pools(torch_)
        kv_key, mamba_key = self._make_keys(2)
        kv_original = torch_.randn(1000, dtype=torch_.bfloat16)
        mamba_original = torch_.randn(1000, dtype=torch_.bfloat16)
        group_id = self._stage_two_component_group(
            kv, mamba, kv_key, mamba_key, kv_original, mamba_original
        )
        self.assertFalse(self.backend.exists(kv_key))
        self.assertFalse(self.backend.exists(mamba_key))
        self.assertTrue(self.backend.commit_group(group_id))
        self.assertTrue(self.backend.exists(kv_key))
        self.assertIsNotNone(
            self.backend._allocator.lookup(
                self.backend._component_key(PoolName.MAMBA, mamba_key)
            )
        )
        group_commits = [
            decode_group_commit_payload(frame.payload)
            for frame in iter_segment_frames(
                segment_path(
                    self.tmpdir,
                    self.backend.config_suffix,
                    self.backend._wal_generation,
                ),
                active=True,
            )
            if frame.frame_type == FRAME_GROUP_COMMIT
        ]
        self.assertEqual(group_commits[-1].member_count, 2)

        self._crash_close_and_reopen()
        restored_kv, restored_mamba = self._register_tensor_pools(torch_)
        self.assertTrue(self.backend.exists(kv_key))
        self.assertIsNotNone(
            self.backend._allocator.lookup(
                self.backend._component_key(PoolName.MAMBA, mamba_key)
            )
        )
        self.assertEqual(
            self.backend.batch_get_v2([self._transfer(PoolName.KV, [kv_key])])[
                PoolName.KV
            ],
            [True],
        )
        self.assertEqual(
            self.backend.batch_get_v2([self._transfer(PoolName.MAMBA, [mamba_key])])[
                PoolName.MAMBA
            ],
            [True],
        )
        self._assert_bit_equal(restored_kv._pages[0], kv_original, torch_)
        self._assert_bit_equal(restored_mamba._pages[0], mamba_original, torch_)


class TestExtentFullFile(ExtentBackendTestBase):
    def test_multi_batch_round_trip_across_extents(self):
        # 3 batches of 128 keys (384 pages, 24 extents): the pack cursor
        # carries across batches and page spans cross batch boundaries.
        backend = ExtentFileBackend(
            _make_config(
                model_name="full-file-round-trip",
                extra_config={
                    "storage_dir": self.tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "1Mi",
                }
            )
        )
        try:
            keys_all, values_all = [], []
            for batch in range(3):
                start = batch * 128
                keys = self._make_keys(128, start=start)
                values = [self._page_bytes(i) for i in range(start, start + 128)]
                self.assertTrue(backend.batch_set(keys, values))
                keys_all += keys
                values_all += values
            targets = [bytearray(1000) for _ in keys_all]
            got = backend.batch_get(keys_all, targets)
            self._assert_pages_equal(got, values_all)
        finally:
            backend.close()


class TestExtentFactory(CustomTestCase):
    def test_factory_creates_real_extent_backend(self):
        from sglang.srt.mem_cache.storage.backend_factory import (
            StorageBackendFactory,
        )

        tmpdir = tempfile.mkdtemp(prefix="extent_factory_")
        try:
            backend = StorageBackendFactory.create_backend(
                "extent",
                _make_config(
                    extra_config={
                        "storage_dir": tmpdir,
                        "extent_size": "16Ki",
                        "file_size": "64Ki",
                    }
                ),
                None,
            )
            self.assertEqual(type(backend).__name__, "ExtentFileBackend")
            self.assertTrue(isinstance(backend, ExtentFileBackend))
            # The instance is fully functional, not a stub.
            self.assertTrue(backend.set("k", bytearray(b"x" * 1000)))
            self.assertTrue(backend.exists("k"))
            backend.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _refhash_page_hashes(token_ids, prior_hex, page_size):
    """Pure-Python SHA256 chained page hash, byte-faithful to the native
    hash_binding.cpp regular path (page hash = SHA256(prior_digest ||
    uint32_le(tokens)); prior digest carries across pages, seeded by the
    caller's prior_hash for page 0).  Used as the deterministic oracle for
    the write->read round-trip assertion when the native extension cannot be
    built (missing libssl-dev); validated byte-for-byte against the native
    hashes in p2io-tester-e7k (CASE A/B)."""
    import hashlib
    import struct

    prior = bytes.fromhex(prior_hex) if prior_hex else None
    has_prior = prior is not None
    out = []
    for start in range(0, len(token_ids), page_size):
        chunk = b"".join(
            struct.pack("<I", t & 0xFFFFFFFF) for t in token_ids[start : start + page_size]
        )
        ctx = hashlib.sha256()
        if has_prior:
            ctx.update(prior)
        if chunk:
            ctx.update(chunk)
        prior = ctx.digest()
        has_prior = True
        out.append(prior.hex())
    return out


class TestExtentWriteQueryRoundTrip(CustomTestCase):
    """Deterministic write->read storage_hit=4 round trip through the real
    controller _storage_hit_query + real backend (e7l/e22 Task 4).

    Encodes the invariant e7l proved at 8-hex (write chain page0 == query
    key[0]) but whose full-64-hex byte-identity is the open question: when
    the write registers a page-hash chain and the read recomputes the same
    span with the same seed, batch_exists must return the full 4 pages.
    The hash function is the validated pure-Python oracle (native module is
    not buildable in this environment)."""

    def _engine_style_backend(self):
        tmpdir = tempfile.mkdtemp(prefix="extent_roundtrip_")
        backend = ExtentFileBackend(
            _make_config(
                model_name="qwen3.5-9b-gptq",
                extra_config={
                    "storage_dir": tmpdir,
                    "extent_size": "16Ki",
                    "file_size": "64Ki",
                },
            )
        )
        self.addCleanup(backend.close)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        return backend

    def _make_query_controller(self, backend, page_size):
        # _storage_hit_query touches only last_hash/token_ids/prefix_keys/
        # all_hash_values plus self.get_hash_str/page_size/storage_backend;
        # instantiate the real method on a bare object (established pattern
        # for method-in-isolation tests) bound to the refhash oracle.
        from sglang.srt.managers.cache_controller import HiCacheController

        ctl = object.__new__(HiCacheController)
        ctl.page_size = page_size
        ctl.storage_backend = backend
        ctl.get_hash_str = _refhash_page_hashes
        return ctl

    def test_same_span_write_then_query_storage_hit_4(self):
        from sglang.srt.managers.cache_controller import StorageOperation

        page_size = 4
        seed_hex = "9d373db168e3e6a03e12ea791ce33b00590b4652dd4618889639c8095ac017f7"
        span = list(range(100, 100 + page_size * 4))  # 4 pages of page_size
        # Write side: the engine registers spec.hash_value = chain seeded by
        # parent's last hash.  Register it like _page_backup does.
        write_keys = _refhash_page_hashes(span, seed_hex, page_size)
        backend = self._engine_style_backend()
        values = [bytearray(b"x" * 1000) for _ in write_keys]
        self.assertTrue(backend.batch_set(write_keys, values))
        self.assertEqual(backend.batch_exists(write_keys), 4)

        # Read side: _storage_hit_query recomputes the same span with the
        # same seed and must hit all 4 pages.
        ctl = self._make_query_controller(backend, page_size)
        op = StorageOperation(None, span, last_hash=seed_hex)
        hash_value, storage_hit_count = ctl._storage_hit_query(op)
        self.assertEqual(len(hash_value), 4)
        self.assertEqual(storage_hit_count, 4 * page_size)
        self.assertEqual(hash_value, write_keys)


if __name__ == "__main__":
    unittest.main()
