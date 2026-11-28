"""Group commit and restorable-prefix tests on the production call chain.

Two properties are pinned here that the backend-level suite cannot see:

* ``HybridCacheController._page_backup`` writes FULL KV, SWA KV and Conv state
  of one backup under a single ``group_id`` and only ``commit_group`` makes the
  backup restorable — including through the real ``BaseHiCacheController`` KV
  batch writer (``_page_set_for_operation``).
* ``ExtentFileBackend.batch_exists_v2`` answers with the *intersection* of the
  pools' restorable prefix lengths, not the minimum of their maxima. SWA and
  Conv windows can be individually fine at lengths that never coincide.

Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_extent_group_commit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import contextlib
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.storage.extent.extent_file_backend import ExtentFileBackend
from sglang.srt.mem_cache.storage.extent.extent_metadata import (
    FRAME_GROUP_COMMIT,
    iter_segment_frames,
    segment_path,
)
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 1  # pages are single slots; host_indices index straight into the pool
PAGE_BYTES = 1024


class _Pool:
    """Minimal host-pool stand-in: one flat page per slot, bytearray backed."""

    def __init__(self, page_bytes=PAGE_BYTES, slots=64):
        self.page_size = PAGE_SIZE
        self._page_bytes = page_bytes
        self._pages = [bytearray(page_bytes) for _ in range(slots)]

    def get_data_page(self, index, flat=True):
        return self._pages[int(index)]

    def get_dummy_flat_data_page(self):
        return bytearray(self._page_bytes)

    def set_from_flat_data_page(self, index, data_page):
        self._pages[int(index)][:] = data_page


def _config(tmpdir):
    return HiCacheStorageConfig(
        extra_config={
            "storage_dir": tmpdir,
            "extent_size": "64Ki",
            "file_size": "512Ki",
        },
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


class ExtentGroupCommitChainTest(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="extent_group_chain_")
        self.backend = ExtentFileBackend(_config(self.tmpdir))
        self.kv = _Pool()
        self.swa = _Pool()
        self.conv = _Pool()
        self.backend.register_mem_host_pool_v2(self.kv, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.swa, PoolName.SWA)
        self.backend.register_mem_host_pool_v2(self.conv, PoolName.MAMBA)

    def tearDown(self):
        if getattr(self, "backend", None) is not None:
            self.backend.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- scaffolding -----------------------------------------------------

    def _controller(self):
        """A real HybridCacheController over the real extent backend.

        Only process state is supplied; every method exercised below is the
        production one.
        """
        ctl = HybridCacheController.__new__(HybridCacheController)
        ctl.storage_backend = self.backend
        ctl.storage_backend_type = "extent"
        ctl.storage_config = SimpleNamespace(is_storage_writer=True)
        ctl.page_size = PAGE_SIZE
        ctl.mem_pool_host = SimpleNamespace(
            layout_lease=lambda: contextlib.nullcontext()
        )
        ctl.page_set_func = lambda hashes, indices, extra_info: all(
            self.backend.batch_set_v1(hashes, indices, extra_info)
        )
        return ctl

    def _operation(self, kv_keys, swa_keys, conv_keys, prefix_keys=None):
        op = StorageOperation(
            host_indices=torch.arange(len(kv_keys)),
            token_ids=[0] * len(kv_keys),
            last_hash=kv_keys[-1] if kv_keys else None,
            hash_value=list(kv_keys),
            prefix_keys=prefix_keys,
            pool_transfers=[
                PoolTransfer(
                    name=name,
                    host_indices=list(range(len(pool_keys))),
                    keys=list(pool_keys),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
                for name, pool_keys in (
                    (PoolName.SWA, swa_keys),
                    (PoolName.MAMBA, conv_keys),
                )
                if pool_keys
            ],
        )
        op.id = 0
        op.node_id = "node-0"
        op.pool_storage_result = PoolTransferResult.empty()
        return op

    def _fill(self, kv_keys, swa_keys, conv_keys):
        for i in range(len(kv_keys)):
            self.kv._pages[i][:] = bytes([10 + i]) * PAGE_BYTES
        for i in range(len(swa_keys)):
            self.swa._pages[i][:] = bytes([60 + i]) * PAGE_BYTES
        for i in range(len(conv_keys)):
            self.conv._pages[i][:] = bytes([90 + i]) * PAGE_BYTES

    def _wal_frames(self):
        return list(
            iter_segment_frames(
                segment_path(
                    self.tmpdir, self.backend.config_suffix, self.backend._wal_generation
                ),
                active=True,
            )
        )

    def _reopen(self):
        self.backend.close()
        self.backend = ExtentFileBackend(_config(self.tmpdir))
        return self.backend

    # ---- 1. the real chain produces a GROUP_COMMIT -----------------------

    def test_page_backup_commits_one_group_for_kv_swa_conv(self):
        kv_keys = ["h0", "h1", "h2"]
        swa_keys = ["h2"]
        conv_keys = ["h2"]
        self._fill(kv_keys, swa_keys, conv_keys)
        ctl = self._controller()
        op = self._operation(kv_keys, swa_keys, conv_keys)

        self.assertTrue(ctl._page_backup(op))
        self.assertIsNone(op.group_id, "the transient group must be cleared")

        frames = self._wal_frames()
        self.assertIn(FRAME_GROUP_COMMIT, [frame.frame_type for frame in frames])

        # Every member of the backup is visible, and all three pools agree on
        # the prefix, so the backup is restorable end to end.
        self.assertTrue(self.backend.exists(kv_keys[-1]))
        result = self.backend.batch_exists_v2(kv_keys, op.pool_transfers)
        self.assertEqual(result.kv_hit_pages, len(kv_keys))
        # A trailing window that covers the whole span is restorable exactly at
        # that span length, so the joint answer is the tail, not every prefix.
        self.assertEqual(result.restorable_prefix_pages, [len(kv_keys)])

        # Survives a restart (the WAL record is what makes it durable).
        backend = self._reopen()
        backend.register_mem_host_pool_v2(_Pool(), PoolName.KV)
        self.assertTrue(backend.exists(kv_keys[-1]))

    def test_page_backup_without_group_api_keeps_legacy_path(self):
        """A backend with no group surface must not be forced through v2."""
        kv_keys = ["g0", "g1"]
        self._fill(kv_keys, [], [])
        ctl = self._controller()
        # Stand-in for the zero-copy v1 writer: the minimal pool here carries no
        # v1 buffer metadata, and this test only pins the controller branch.
        ctl.page_set_func = lambda hashes, indices, extra_info: bool(
            self.backend.batch_set(hashes, [bytes([7]) * PAGE_BYTES for _ in hashes])
        )
        op = self._operation(kv_keys, [], [])
        with mock.patch.object(self.backend, "open_group", None):
            self.assertTrue(ctl._page_backup(op))
        self.assertTrue(self.backend.exists(kv_keys[-1]))

    # ---- 2. failure injection: an uncommitted group is not restorable ----

    def test_sidecar_failure_aborts_group_and_leaves_nothing_restorable(self):
        kv_keys = ["h0", "h1"]
        swa_keys = ["h1"]
        conv_keys = ["h1"]
        self._fill(kv_keys, swa_keys, conv_keys)
        ctl = self._controller()
        op = self._operation(kv_keys, swa_keys, conv_keys)

        # The Conv member fails: its indices do not match its keys, which the
        # backend rejects before staging anything for that pool.
        op.pool_transfers[1].host_indices = []
        self.assertFalse(ctl._page_backup(op))
        self.assertIsNone(op.group_id)

        frames = self._wal_frames()
        self.assertNotIn(FRAME_GROUP_COMMIT, [frame.frame_type for frame in frames])
        # KV staged in the abandoned group must not be visible either.
        self.assertFalse(self.backend.exists(kv_keys[-1]))
        self.assertEqual(self.backend._staged_groups, {})
        self.assertEqual(self.backend._staged_reserved, {})
        result = self.backend.batch_exists_v2(kv_keys, op.pool_transfers)
        self.assertEqual(result.kv_hit_pages, 0)

        backend = self._reopen()
        backend.register_mem_host_pool_v2(_Pool(), PoolName.KV)
        self.assertFalse(backend.exists(kv_keys[-1]))

    def test_kv_failure_aborts_group_after_sidecars_staged(self):
        kv_keys = ["h0", "h1"]
        self._fill(kv_keys, ["h1"], ["h1"])
        ctl = self._controller()
        op = self._operation(kv_keys, ["h1"], ["h1"])
        # KV batch write fails (v2 refuses the mismatched batch).
        op.host_indices = []
        self.assertFalse(ctl._page_backup(op))
        self.assertNotIn(
            FRAME_GROUP_COMMIT, [frame.frame_type for frame in self._wal_frames()]
        )
        self.assertFalse(self.backend.exists(kv_keys[-1]))
        self.assertEqual(self.backend._staged_groups, {})
        # Sidecars staged for this backup were released too.
        self.assertEqual(self.backend._staged_reserved, {})

    def test_commit_failure_aborts_group(self):
        kv_keys = ["h0", "h1"]
        self._fill(kv_keys, ["h1"], ["h1"])
        ctl = self._controller()
        op = self._operation(kv_keys, ["h1"], ["h1"])
        with mock.patch.object(self.backend, "commit_group", return_value=False):
            self.assertFalse(ctl._page_backup(op))
        self.assertFalse(self.backend.exists(kv_keys[-1]))
        self.assertEqual(self.backend._staged_groups, {})

    def test_group_is_aborted_when_a_member_raises(self):
        kv_keys = ["h0"]
        self._fill(kv_keys, ["h0"], ["h0"])
        ctl = self._controller()
        op = self._operation(kv_keys, ["h0"], ["h0"])
        with mock.patch.object(
            self.backend, "batch_set_v2", side_effect=RuntimeError("boom")
        ):
            self.assertFalse(ctl._page_backup(op))
        self.assertEqual(self.backend._staged_groups, {})
        self.assertFalse(self.backend.exists(kv_keys[-1]))


class ExtentRestorablePrefixIntersectionTest(CustomTestCase):
    """SWA and Conv windows that do not coincide must not yield a hit."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="extent_intersection_")
        self.backend = ExtentFileBackend(_config(self.tmpdir))
        self.kv = _Pool(slots=32)
        self.swa = _Pool(slots=32)
        self.conv = _Pool(slots=32)
        self.backend.register_mem_host_pool_v2(self.kv, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.swa, PoolName.SWA)
        self.backend.register_mem_host_pool_v2(self.conv, PoolName.MAMBA)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _put_pages(self, pool, name, page_indexes):
        """Store *pool*'s component for the keys at ``page_indexes``."""
        for index in page_indexes:
            self.kv  # keep pool identity mapping explicit
            self._pool = pool
            self.assertTrue(
                self.backend.batch_set_v2(
                    [
                        PoolTransfer(
                            name=name,
                            host_indices=[index],
                            keys=[f"k{index}"],
                        )
                    ]
                )[name][0]
            )

    def _put_kv(self, keys):
        self._put_pages(self.kv, PoolName.KV, range(len(keys)))

    def _transfers(self, swa_window, conv_window):
        return [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=list(range(swa_window)),
                keys=[f"w{i}" for i in range(swa_window)],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=list(range(conv_window)),
                keys=[f"c{i}" for i in range(conv_window)],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]

    def test_disjoint_trailing_windows_report_no_restorable_prefix(self):
        """SWA is restorable at 4, Conv at 6, nothing jointly.

        Folding each pool to its maximum and taking the minimum would report 4,
        a length Conv cannot serve, and the prefetch would fail after
        committing to it.
        """
        keys = [f"k{i}" for i in range(8)]
        self._put_kv(keys)
        # SWA (window 2): only pages 2,3 are on disk => restorable at length 4.
        self._put_pages(self.swa, PoolName.SWA, (2, 3))
        # Conv (window 1): only page 5 is on disk => restorable at length 6.
        self._put_pages(self.conv, PoolName.MAMBA, (5,))

        result = self.backend.batch_exists_v2(keys, self._transfers(2, 1))
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.restorable_prefix_pages, [])
        # Each pool still reports its own boundary for belief healing.
        self.assertEqual(result.extra_pool_hit_pages[PoolName.SWA], 4)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], 6)

    def test_intersection_is_smaller_than_every_pool_maximum(self):
        """SWA restorable at {4,6}, Conv at {4,5} => joint answer 4, not 5."""
        keys = [f"k{i}" for i in range(8)]
        self._put_kv(keys)
        self._put_pages(self.swa, PoolName.SWA, (3, 5))  # window 1
        self._put_pages(self.conv, PoolName.MAMBA, (3, 4))  # window 1

        result = self.backend.batch_exists_v2(keys, self._transfers(1, 1))
        self.assertEqual(result.extra_pool_hit_pages[PoolName.SWA], 6)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], 5)
        self.assertEqual(result.kv_hit_pages, 4)
        self.assertEqual(result.restorable_prefix_pages, [4])

    def test_missing_kv_page_caps_every_pool(self):
        keys = [f"k{i}" for i in range(8)]
        self._put_kv(keys[:5])  # pages 5..7 absent
        self._put_pages(self.swa, PoolName.SWA, range(5))
        self._put_pages(self.conv, PoolName.MAMBA, range(5))
        result = self.backend.batch_exists_v2(keys, self._transfers(1, 1))
        self.assertEqual(result.kv_hit_pages, 5)
        self.assertEqual(result.restorable_prefix_pages, [1, 2, 3, 4, 5])

    def test_all_pages_pool_intersects_like_the_file_backend(self):
        """A DSA-style ALL_PAGES pool gates the KV prefix contiguously."""
        keys = [f"k{i}" for i in range(4)]
        self._put_kv(keys)
        self._put_pages(self.swa, PoolName.SWA, (0, 1))
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=[0, 1],
            keys=["k0", "k1"],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.backend.batch_exists_v2(keys, [transfer])
        self.assertEqual(result.extra_pool_hit_pages[PoolName.SWA], 2)
        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(result.restorable_prefix_pages, [1, 2])


def restorable(keys):
    return list(range(1, len(keys) + 1))


def pool_name_of(pool, testcase):
    return {
        id(testcase.kv): PoolName.KV,
        id(testcase.swa): PoolName.SWA,
        id(testcase.conv): PoolName.MAMBA,
    }[id(pool)]


if __name__ == "__main__":
    unittest.main()
