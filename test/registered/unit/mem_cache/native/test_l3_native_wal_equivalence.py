# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""WAL byte-equivalence oracle for the L3 native write-path A/B seam.

Arm A is the untouched Python reference path; arm B is the native/GIL-free
seam (``SGLANG_L3_NATIVE_WRITE_PATH=1``).  No existing test compares the WAL
bytes produced by the two arms, and the backend has **no data-page checksum**,
so a silent divergence would only show up as wrong data on recovery.  This
file closes that gap:

* identical write sequences on two fresh backends must produce **byte-identical
  WAL segments and data files** (sha256 compared, not just CRCs);
* the same holds on the zero-intermediate-gather-copy raw `get_page_buffer_meta`
  path against a real CPU ``MHATokenToKVPoolHost``;
* pointer-range validation rejects an out-of-arena raw `(ptr, len)` before any
  dereference;
* the submit -> completion -> release page-lease invariants hold.

Runs CPU-only.  ``CUDA_VISIBLE_DEVICES`` must be a non-GPU ordinal (e.g. 999):
the frozen ``sglang.test.test_utils`` indexes ``CUDA_VISIBLE_DEVICES[0]``, so an
empty value is not usable for the pytest suite.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.extent.extent_file_backend import (
    ExtentFileBackend,
)
from sglang.srt.mem_cache.storage.extent import extent_file_backend as efb
from sglang.test.test_utils import CustomTestCase

try:
    import torch
except ImportError:  # pragma: no cover - torch is present in the frozen env
    torch = None

_NATIVE_ENV = "SGLANG_L3_NATIVE_WRITE_PATH"


def _make_config(storage_dir: str, model_name: str = "wal-eq") -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        extra_config={
            "storage_dir": storage_dir,
            "extent_size": "16Ki",
            "file_size": "64Ki",
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
        model_name=model_name,
    )



def _host_pool_kwargs(device_pool):
    """v0.5.23 host pools take a MHABufferInfo instead of the device pool."""
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

class _PagePool:
    """Minimal host-pool stand-in (page_size + flat page API), mirroring the
    frozen unit suite's stand-in: no ``get_page_buffer_meta`` => arm B uses the
    buffer submit path."""

    def __init__(self, page_bytes: int, slots: int = 8):
        self.page_size = 1
        self._page_bytes = page_bytes
        self._pages = [bytearray(page_bytes) for _ in range(slots)]

    def get_data_page(self, index, flat=True):
        return self._pages[index]


def _wal_and_data_bytes(backend: ExtentFileBackend):
    wal_path = Path(backend._storage_dir) / (
        f"hicache.meta{backend.config_suffix}.wal.0"
    )
    data_path = Path(backend._storage_dir) / (
        f"hicache.data{backend.config_suffix}"
    )
    return wal_path.read_bytes(), data_path.read_bytes()


class WalByteEquivalenceTests(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="l3_native_wal_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop(_NATIVE_ENV, None)

    def _new_backend(self, name: str, native: bool) -> ExtentFileBackend:
        sub = Path(self.tmpdir) / name
        sub.mkdir()
        if native:
            os.environ[_NATIVE_ENV] = "1"
        else:
            os.environ.pop(_NATIVE_ENV, None)
        # Same model_name on both arms: config_suffix is embedded in the storage
        # key, so a differing suffix would legitimately change the frame bytes.
        return ExtentFileBackend(_make_config(str(sub), model_name="wal-eq"))

    def _write_sequence(self, backend: ExtentFileBackend):
        keys = [f"k{i:04d}" for i in range(6)]
        values = [bytes([i % 251]) * 1000 for i in range(6)]
        self.assertTrue(backend.batch_set(keys, values))
        self.assertTrue(
            backend.batch_set(["z1", "z2"], [bytes([9]) * 1000, bytes([8]) * 1000])
        )

    def test_wal_and_data_bytes_are_identical_python_vs_native(self):
        """Same ungrouped writes => identical WAL segment and data file bytes."""
        backend_a = self._new_backend("a", native=False)
        self._write_sequence(backend_a)
        wal_a, data_a = _wal_and_data_bytes(backend_a)
        stats_a = backend_a.stats_snapshot()
        backend_a.close()

        backend_b = self._new_backend("b", native=True)
        self._write_sequence(backend_b)
        wal_b, data_b = _wal_and_data_bytes(backend_b)
        stats_b = backend_b.stats_snapshot()
        backend_b.close()

        print(
            "WAL: A %d B %d shaA=%s shaB=%s"
            % (
                len(wal_a),
                len(wal_b),
                hashlib.sha256(wal_a).hexdigest(),
                hashlib.sha256(wal_b).hexdigest(),
            )
        )
        print(
            "DATA: A %d B %d shaA=%s shaB=%s"
            % (
                len(data_a),
                len(data_b),
                hashlib.sha256(data_a).hexdigest(),
                hashlib.sha256(data_b).hexdigest(),
            )
        )
        self.assertGreater(len(wal_a), 0)
        self.assertEqual(wal_a, wal_b, "WAL bytes differ between arm A and arm B")
        self.assertEqual(data_a, data_b, "data bytes differ between arm A and arm B")
        # The same durable cursor: no renumbering or double-advance.
        self.assertEqual(stats_a["wal_active_bytes"], stats_b["wal_active_bytes"])
        self.assertEqual(stats_a["bytes_written"], stats_b["bytes_written"])

    @unittest.skipIf(torch is None, "torch not installed")
    def test_wal_bytes_identical_on_raw_get_page_buffer_meta_path(self):
        """Arm B raw zero-intermediate-gather-copy path is byte-equivalent."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.pool_host import base
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        page_size, layer_num, head_num, head_dim = 2, 2, 2, 4

        def build_pool():
            budget = mock.patch.object(
                base,
                "available_host_memory_bytes",
                return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3),
            )
            budget.start()
            device_pool = MHATokenToKVPool(
                size=page_size * 2,
                page_size=page_size,
                dtype=torch.float16,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=layer_num,
                device="cpu",
                enable_memory_saver=False,
            )
            host_pool = MHATokenToKVPoolHost(
                **_host_pool_kwargs(device_pool),
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout="page_first",
                pin_memory=False,
                device="cpu",
                allocator_type="default",
            )
            return budget, host_pool

        def run(name: str, native: bool):
            budget, host_pool = build_pool()
            backend = self._new_backend(name, native=native)
            backend.register_mem_host_pool_v2(host_pool, PoolName.KV)
            pages = host_pool.page_num
            for page in range(pages):
                host_pool.kv_buffer[0, page * page_size : (page + 1) * page_size] = (
                    float(page + 1)
                )
                host_pool.kv_buffer[1, page * page_size : (page + 1) * page_size] = (
                    float(page + 100)
                )
            keys = [f"k{i:04d}" for i in range(pages)]
            indices = torch.arange(pages * page_size, dtype=torch.int64)
            result = backend.batch_set_v2(
                [PoolTransfer(name=PoolName.KV, host_indices=indices, keys=keys)]
            )
            self.assertEqual(result[PoolName.KV], [True] * pages)
            wal, data = _wal_and_data_bytes(backend)
            arenas = len(backend._native_registered_arenas)
            backend.close()
            budget.stop()
            return wal, data, arenas, pages

        wal_a, data_a, arenas_a, pages = run("meta_a", native=False)
        wal_b, data_b, arenas_b, _ = run("meta_b", native=True)
        print(
            "META pages=%d arenas A=%d B=%d walA=%s walB=%s"
            % (
                pages,
                arenas_a,
                arenas_b,
                hashlib.sha256(wal_a).hexdigest(),
                hashlib.sha256(wal_b).hexdigest(),
            )
        )
        # Arm B must have taken the raw path (an arena was registered), and the
        # durable bytes must still be identical to the arm-A gather.
        self.assertEqual(arenas_a, 0)
        self.assertGreaterEqual(arenas_b, 1)
        self.assertEqual(wal_a, wal_b)
        self.assertEqual(data_a, data_b)


    def test_native_set_and_commit_payload_bytes_match_python(self):
        """Native frame bytes are exact for a nonzero group epoch/seq too (the
        grouped-write case, where a backend-random epoch prevents cross-backend
        WAL comparison)."""
        from sglang.srt.mem_cache.storage.extent import extent_metadata as em

        module = efb._load_native_module()
        epoch = bytes(range(16))
        entries = [
            (b"key-a", 3, 128, 4096, 7, 1),
            (b"key-b" * 10, 0, 0, 8192, 9, 255),
        ]
        python_payload = em.encode_set_payload(
            em.SetPayload(
                group_epoch=epoch,
                group_seq=42,
                batch_id=5,
                member_index=0,
                member_count=1,
                entries=[em.SetEntry(*entry) for entry in entries],
            )
        )
        native_payload = module.encode_set_payload(epoch, 42, 5, 0, 1, entries)
        self.assertEqual(native_payload, python_payload)
        seq = 123
        self.assertEqual(
            module.encode_frame(em.FRAME_SET, seq, native_payload),
            em.encode_frame(em.FRAME_SET, seq, python_payload),
        )
        self.assertEqual(
            module.encode_batch_commit_payload(5, 1),
            em.encode_batch_commit_payload(em.BatchCommitPayload(5, 1)),
        )


class NativeSeamInvariantTests(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="l3_native_inv_")
        self._keepalive = []

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop(_NATIVE_ENV, None)

    def _leak_bytearray(self, size: int) -> int:
        page = bytearray(size)
        self._keepalive.append(page)  # keep the mapping stable for the test
        return ctypes.addressof(ctypes.c_char.from_buffer(page))

    def test_range_validation_rejects_out_of_arena_before_dereference(self):
        module = efb._load_native_module()
        arena = bytearray(64 * 1024)
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        self.assertTrue(module.validate_range(base, 1024, base, len(arena)))
        self.assertFalse(module.validate_range(base + len(arena) - 8, 64, base, len(arena)))

        writer = module.Writer()
        try:
            writer.register_arena(base, len(arena))
            out_fd = os.open(Path(self.tmpdir) / "oob.bin", os.O_CREAT | os.O_RDWR, 0o644)
            try:
                with self.assertRaises(ValueError):
                    writer.submit_write_raw(
                        data_fd=out_fd,
                        plans=[(0, [(base + len(arena) + 4096, 64)])],
                        wal_fd=out_fd,
                        wal_offset=0,
                        wal_seq=1,
                        group_epoch=b"\x00" * 16,
                        group_seq=0,
                        batch_id=1,
                        member_index=0,
                        member_count=1,
                        entries=[],
                        wal_max_record_bytes=8 << 20,
                    )
            finally:
                os.close(out_fd)
            stats = dict(writer.stats())
            self.assertEqual(stats["range_validation_refusals"], 1)
            self.assertEqual(writer.submitted(), 0, "rejected job must not be enqueued")
        finally:
            writer.shutdown()

    def test_lease_established_at_submit_and_released_only_after_completion(self):
        module = efb._load_native_module()
        data_fd = os.open(Path(self.tmpdir) / "data.bin", os.O_CREAT | os.O_RDWR, 0o644)
        wal_fd = os.open(Path(self.tmpdir) / "wal.bin", os.O_CREAT | os.O_RDWR, 0o644)
        writer = module.Writer()
        try:
            base = self._leak_bytearray(4096)
            writer.register_arena(base, 4096)
            job = writer.submit_write_raw(
                data_fd=data_fd,
                plans=[(0, [(base, 4096)])],
                wal_fd=wal_fd,
                wal_offset=0,
                wal_seq=1,
                group_epoch=b"\x00" * 16,
                group_seq=0,
                batch_id=1,
                member_index=0,
                member_count=1,
                entries=[(b"k", 0, 0, 4096, 1, 0)],
                wal_max_record_bytes=8 << 20,
            )
            # Invariant 1: the lease exists from submit.
            self.assertTrue(writer.lease_active(job))
            completion = writer.wait(job)
            self.assertIsNotNone(completion)
            self.assertTrue(completion[1], completion[2])
            # Invariant 2: a recycle overlapping the leased page is refused even
            # after completion, until the owner releases.
            self.assertEqual(writer.recompute_conflict(base, 4096), job)
            writer.release(job)
            # Invariant 3: release follows completion; the lease then clears.
            self.assertFalse(writer.lease_active(job))
            self.assertEqual(writer.recompute_conflict(base, 4096), 0)
            stats = dict(writer.stats())
            self.assertEqual(stats["leases_established"], 1)
            self.assertEqual(stats["leases_released"], 1)
        finally:
            writer.shutdown()
            os.close(data_fd)
            os.close(wal_fd)

    def test_lease_registry_refuses_recycle_while_leased(self):
        """Negative control carried into the real crate: with a lease the
        recycle is refused and writes nothing; without it the copy happens."""
        module = efb._load_native_module()
        arena = bytearray(64 * 1024)
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        page = bytearray(4096)
        src = ctypes.addressof(ctypes.c_char.from_buffer(page))
        registry = module.LeaseRegistry(base, len(arena))
        slot, generation = registry.alloc_slot(base, 4096)
        lease = registry.submit_lease(slot, generation)
        self.assertFalse(registry.try_recycle(slot, generation + 1, src, 4096))
        with self.assertRaises(ValueError):
            registry.commit_install_and_release(lease)
        registry.mark_completed(lease)
        registry.commit_install_and_release(lease)
        self.assertTrue(registry.try_recycle(slot, generation + 2, src, 4096))
        stats = dict(registry.stats())
        self.assertEqual(stats["recycle_refused_leased"], 1)
        self.assertEqual(stats["recycle_allowed"], 1)
        self.assertEqual(stats["release_before_completion_refused"], 1)

    def test_data_pwrite_failure_appends_no_wal_frame(self):
        """Durable ordering: data pwritev precedes the WAL frames, so a failed
        data write must leave the WAL cursor and file untouched."""
        module = efb._load_native_module()
        base = self._leak_bytearray(4096)
        wal_fd = os.open(Path(self.tmpdir) / "order.wal", os.O_CREAT | os.O_RDWR, 0o644)
        writer = module.Writer()
        try:
            writer.register_arena(base, 4096)
            job = writer.submit_write_raw(
                data_fd=-1,  # guaranteed pwritev failure
                plans=[(0, [(base, 4096)])],
                wal_fd=wal_fd,
                wal_offset=0,
                wal_seq=1,
                group_epoch=b"\x00" * 16,
                group_seq=0,
                batch_id=1,
                member_index=0,
                member_count=1,
                entries=[(b"k", 0, 0, 4096, 1, 0)],
                wal_max_record_bytes=8 << 20,
            )
            completion = writer.wait(job)
            self.assertFalse(completion[1])
            self.assertEqual(completion[5], 0, "WAL offset advanced after data failure")
            self.assertEqual(completion[6], 1, "WAL seq advanced after data failure")
            self.assertEqual(os.fstat(wal_fd).st_size, 0, "WAL frame written before data")
        finally:
            writer.shutdown()
            os.close(wal_fd)

    def test_iov_max_chunking_mandatory_at_2048_iovecs(self):
        """A kv extent needs 2048 iovecs; the worker must shard at IOV_MAX."""
        module = efb._load_native_module()
        chunk = 64
        count = 2048
        arena = bytearray(chunk * count)
        for index in range(count):
            arena[index * chunk] = index % 251
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        self._keepalive.append(arena)
        data_fd = os.open(Path(self.tmpdir) / "iov.data", os.O_CREAT | os.O_RDWR, 0o644)
        wal_fd = os.open(Path(self.tmpdir) / "iov.wal", os.O_CREAT | os.O_RDWR, 0o644)
        writer = module.Writer()
        try:
            writer.register_arena(base, len(arena))
            plans = [(0, [(base + index * chunk, chunk) for index in range(count)])]
            job = writer.submit_write_raw(
                data_fd=data_fd,
                plans=plans,
                wal_fd=wal_fd,
                wal_offset=0,
                wal_seq=1,
                group_epoch=b"\x00" * 16,
                group_seq=1,
                batch_id=1,
                member_index=0,
                member_count=1,
                entries=[(b"k", 0, 0, chunk * count, 1, 0)],
                wal_max_record_bytes=8 << 20,
            )
            completion = writer.wait(job)
            self.assertTrue(completion[1], completion[2])
            self.assertEqual(completion[4], chunk * count)
            self.assertGreaterEqual(
                completion[7], 2, "2048 iovecs must shard into >= 2 pwritev syscalls"
            )
            self.assertGreaterEqual(module.iov_max(), 1)
            written = os.pread(data_fd, chunk, 0)
            self.assertEqual(written[0], 0)
            self.assertEqual(os.pread(data_fd, 1, (count - 1) * chunk), bytes([(count - 1) % 251]))
        finally:
            writer.shutdown()
            os.close(data_fd)
            os.close(wal_fd)



def _flat_page_bytes(host_pool, page_index: int, page_size: int) -> bytes:
    page = host_pool.kv_buffer[:, page_index * page_size : (page_index + 1) * page_size]
    return page.flatten().view(torch.uint8).numpy().tobytes()


def _build_mha_pool(pages: int, page_size=2, layer_num=2, head_num=2, head_dim=4):
    """Real CPU page_first MHA host pool with at least ``pages`` host pages."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host import base
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    budget = mock.patch.object(
        base,
        "available_host_memory_bytes",
        return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3),
    )
    budget.start()
    device_pool = MHATokenToKVPool(
        size=pages * page_size,
        page_size=page_size,
        dtype=torch.float16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )
    host_pool = MHATokenToKVPoolHost(
        **_host_pool_kwargs(device_pool),
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=page_size,
        layout="page_first",
        pin_memory=False,
        device="cpu",
        allocator_type="default",
    )
    return budget, host_pool


class ChunkAtomicityTests(CustomTestCase):
    """H1 -- a mid-way chunk failure must fail the whole batch closed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="l3_h1_")
        self._keepalive = []
        os.environ[_NATIVE_ENV] = "1"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop(_NATIVE_ENV, None)

    def test_midway_chunk_failure_writes_no_wal_frame(self):
        """Chunk 1 succeeds, chunk 2 fails => no WAL frame, whole job failed.

        The first plan needs more than IOV_MAX iovecs so it is itself split into
        chunks; the second plan points past ``off_t`` range so its very first
        ``pwritev`` fails with EINVAL. Bytes written by the earlier chunks stay
        in the data file (the documented partial window), but because no
        FRAME_SET / FRAME_BATCH_COMMIT is ever appended they are unreachable.
        """
        module = efb._load_native_module()
        iov_max = int(module.iov_max())
        chunk = 64
        count = iov_max + 200  # 1224 pairs -> 2 chunks (1024 + 200)
        arena = bytearray(chunk * count)
        for index in range(count):
            arena[index * chunk] = index % 251
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        self._keepalive.append(arena)
        data_fd = os.open(
            Path(self.tmpdir) / "chunk.data", os.O_CREAT | os.O_RDWR, 0o644
        )
        wal_fd = os.open(
            Path(self.tmpdir) / "chunk.wal", os.O_CREAT | os.O_RDWR, 0o644
        )
        writer = module.Writer()
        try:
            writer.register_arena(base, len(arena))
            plans = [
                (0, [(base + index * chunk, chunk) for index in range(count)]),
                # u64::MAX casts to off_t -1 => pwritev EINVAL on the 2nd plan.
                ((1 << 64) - 1, [(base, chunk)]),
            ]
            job = writer.submit_write_raw(
                data_fd=data_fd,
                plans=plans,
                wal_fd=wal_fd,
                wal_offset=0,
                wal_seq=1,
                group_epoch=b"\x00" * 16,
                group_seq=0,
                batch_id=1,
                member_index=0,
                member_count=1,
                entries=[(b"k", 0, 0, chunk * count, 1, 0)],
                wal_max_record_bytes=8 << 20,
            )
            completion = writer.wait(job)
            _jid, ok, error, errno, _written, wal_offset, wal_seq, syscalls = completion
            self.assertFalse(ok, "a failed chunk must fail the whole job")
            self.assertEqual(errno, 22, f"expected EINVAL from the 2nd plan: {error}")
            self.assertGreaterEqual(syscalls, 2)
            self.assertEqual(wal_offset, 0, "WAL offset advanced on a data failure")
            self.assertEqual(wal_seq, 1, "WAL seq advanced on a data failure")
            self.assertEqual(os.fstat(wal_fd).st_size, 0, "no WAL frame may be written")
            # Documented partial window: the first chunk's bytes ARE on disk.
            self.assertEqual(os.pread(data_fd, chunk, 0), bytes([0]) * chunk)
        finally:
            writer.shutdown()
            os.close(data_fd)
            os.close(wal_fd)

    def test_partial_chunk_write_is_not_recovered_as_committed(self):
        """End-to-end H1: partial data from a refused batch never becomes visible."""
        backend = ExtentFileBackend(_make_config(self.tmpdir, model_name="h1-partial"))
        key = backend._storage_key("partial")
        try:
            original_wal_fd = backend._wal_fd
            backend._wal_fd = os.open(
                Path(self.tmpdir) / "h1-partial-ro.wal", os.O_CREAT | os.O_RDONLY, 0o644
            )
            try:
                ok = backend.batch_set([key], [bytearray(b"\x5a" * 1000)])
            finally:
                os.close(backend._wal_fd)
                backend._wal_fd = original_wal_fd
            self.assertFalse(ok, "no ACK for a batch whose WAL frame never landed")
            stats = backend.stats_snapshot()
            self.assertGreaterEqual(stats["native_write_failures"], 1)
            self.assertIn("errno=9", stats["native_write_last_error"] or "")
            self.assertFalse(backend.exists(key))
        finally:
            backend.close()
        reopened = ExtentFileBackend(_make_config(self.tmpdir, model_name="h1-partial"))
        try:
            self.assertEqual(reopened.batch_exists([key]), 0)
            self.assertFalse(reopened.exists(key))
        finally:
            reopened.close()


class NativeErrorPropagationTests(CustomTestCase):
    """H2 -- every native failure surfaces exactly once and fails closed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="l3_h2_")
        self._keepalive = []

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop(_NATIVE_ENV, None)

    def _submit(self, writer, data_fd, wal_fd, pairs, entries=None, wal_max=8 << 20):
        return writer.submit_write_raw(
            data_fd=data_fd,
            plans=[(0, pairs)],
            wal_fd=wal_fd,
            wal_offset=0,
            wal_seq=1,
            group_epoch=b"\x00" * 16,
            group_seq=0,
            batch_id=1,
            member_index=0,
            member_count=1,
            entries=entries if entries is not None else [(b"k", 0, 0, 4096, 1, 0)],
            wal_max_record_bytes=wal_max,
        )

    def test_pwritev_ebadf_propagates_with_errno_and_no_wal_advance(self):
        arena = bytearray(4096)
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        self._keepalive.append(arena)
        wal_fd = os.open(Path(self.tmpdir) / "h2.wal", os.O_CREAT | os.O_RDWR, 0o644)
        writer = efb._load_native_module().Writer()
        try:
            writer.register_arena(base, len(arena))
            job = self._submit(writer, -1, wal_fd, [(base, 4096)])
            _jid, ok, error, errno, written, wal_offset, wal_seq, _syscalls = writer.wait(
                job
            )
            self.assertFalse(ok, "EBADF must be reported as failure, never success")
            self.assertEqual(errno, 9, error)
            self.assertIn("pwritev", error)
            self.assertEqual(written, 0)
            self.assertEqual((wal_offset, wal_seq), (0, 1), "WAL cursor advanced")
            self.assertEqual(os.fstat(wal_fd).st_size, 0)
            # Failure must not release the lease before the owner observes it.
            self.assertTrue(writer.lease_active(job))
            writer.release(job)
            self.assertFalse(writer.lease_active(job))
            stats = dict(writer.stats())
            self.assertEqual(stats["failed"], 1)
            self.assertEqual(stats["completed"], 0)
        finally:
            writer.shutdown()
            os.close(wal_fd)

    def test_wal_pwrite_failure_after_data_leaves_no_committed_batch(self):
        arena = bytearray(4096)
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        self._keepalive.append(arena)
        data_fd = os.open(
            Path(self.tmpdir) / "h2d.data", os.O_CREAT | os.O_RDWR, 0o644
        )
        wal_fd = os.open(
            Path(self.tmpdir) / "h2d-ro.wal", os.O_CREAT | os.O_RDONLY, 0o644
        )
        writer = efb._load_native_module().Writer()
        try:
            writer.register_arena(base, len(arena))
            job = self._submit(writer, data_fd, wal_fd, [(base, 4096)])
            _jid, ok, error, errno, written, wal_offset, wal_seq, _syscalls = writer.wait(
                job
            )
            self.assertFalse(ok)
            self.assertEqual(errno, 9, error)
            self.assertEqual(written, 4096, "data chunk should have been written")
            self.assertEqual((wal_offset, wal_seq), (0, 1), "WAL cursor advanced")
            self.assertEqual(os.fstat(wal_fd).st_size, 0)
        finally:
            writer.shutdown()
            os.close(data_fd)
            os.close(wal_fd)

    def test_encoder_limit_failure_surfaces_and_writes_nothing(self):
        arena = bytearray(4096)
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        self._keepalive.append(arena)
        data_fd = os.open(
            Path(self.tmpdir) / "h2e.data", os.O_CREAT | os.O_RDWR, 0o644
        )
        wal_fd = os.open(
            Path(self.tmpdir) / "h2e.wal", os.O_CREAT | os.O_RDWR, 0o644
        )
        writer = efb._load_native_module().Writer()
        try:
            writer.register_arena(base, len(arena))
            job = self._submit(
                writer,
                data_fd,
                wal_fd,
                [(base, 4096)],
                entries=[(b"k" * 64, 0, 0, 4096, 1, 0)],
                wal_max=16,  # smaller than the encoded SET payload
            )
            _jid, ok, error, errno, written, wal_offset, wal_seq, _syscalls = writer.wait(
                job
            )
            self.assertFalse(ok)
            self.assertIn("exceeds wal_max_record_bytes", error)
            self.assertEqual(errno, 0)
            self.assertEqual(written, 4096, "data write happened before the encoder")
            self.assertEqual((wal_offset, wal_seq), (0, 1))
            self.assertEqual(os.fstat(wal_fd).st_size, 0)
        finally:
            writer.shutdown()
            os.close(data_fd)
            os.close(wal_fd)

    def test_exactly_one_completion_per_submitted_request(self):
        """H2: no silent drop -- N submits produce exactly N completions."""
        module = efb._load_native_module()
        writer = module.Writer()
        data_fd = os.open(
            Path(self.tmpdir) / "h2n.data", os.O_CREAT | os.O_RDWR, 0o644
        )
        wal_fd = os.open(
            Path(self.tmpdir) / "h2n.wal", os.O_CREAT | os.O_RDWR, 0o644
        )
        arenas = []
        try:
            jobs = []
            for index in range(5):
                arena = bytearray(1024)
                arenas.append(arena)
                base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
                writer.register_arena(base, len(arena))
                fd = data_fd if index % 2 == 0 else -1  # 3 ok, 2 forced failures
                jobs.append(
                    writer.submit_write_raw(
                        data_fd=fd,
                        plans=[(index * 1024, [(base, 1024)])],
                        wal_fd=wal_fd,
                        wal_offset=index * 64,
                        wal_seq=index + 1,
                        group_epoch=b"\x00" * 16,
                        group_seq=index + 1,
                        batch_id=index + 1,
                        member_index=0,
                        member_count=1,
                        entries=[(b"k", 0, index * 1024, 1024, 1, 0)],
                        wal_max_record_bytes=8 << 20,
                    )
                )
            for job in jobs:
                self.assertIsNotNone(writer.wait(job))
                writer.release(job)
            drained = writer.poll()
            self.assertEqual(len(drained), 5)
            self.assertEqual(sorted(entry[0] for entry in drained), sorted(jobs))
            self.assertEqual(len({entry[0] for entry in drained}), 5)
            self.assertEqual(writer.submitted(), 5)
            self.assertEqual(writer.outstanding_leases(), 0)
        finally:
            writer.shutdown()
            os.close(data_fd)
            os.close(wal_fd)


class NativeShutdownLeaseTests(CustomTestCase):
    """H3 -- leases must also cover shutdown / cancel, not only reuse."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="l3_h3_")
        self._keepalive = []
        os.environ[_NATIVE_ENV] = "1"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop(_NATIVE_ENV, None)

    def test_cancel_midflight_resolves_every_lease_without_uaf(self):
        """Park the worker mid-job, cancel the queue, resume, then verify:
        (a) no use-after-free -- the parked pointer stayed valid and the bytes
        written are correct; (b) no lease leak; (c) the parked job still gets
        exactly one completion and the cancelled jobs each get one too."""
        module = efb._load_native_module()
        chunk = 256
        arena = bytearray(chunk * 4)
        for index in range(4):
            arena[index * chunk] = index + 1
        base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        self._keepalive.append(arena)
        data_fd = os.open(Path(self.tmpdir) / "h3.data", os.O_CREAT | os.O_RDWR, 0o644)
        wal_fd = os.open(Path(self.tmpdir) / "h3.wal", os.O_CREAT | os.O_RDWR, 0o644)
        writer = module.Writer()

        def submit(index):
            return writer.submit_write_raw(
                data_fd=data_fd,
                plans=[(index * chunk, [(base + index * chunk, chunk)])],
                wal_fd=wal_fd,
                wal_offset=index * 128,
                wal_seq=index + 1,
                group_epoch=b"\x00" * 16,
                group_seq=index + 1,
                batch_id=index + 1,
                member_index=0,
                member_count=1,
                entries=[(b"k", 0, index * chunk, chunk, 1, 0)],
                wal_max_record_bytes=8 << 20,
            )

        try:
            writer.register_arena(base, len(arena))
            writer.arm_test_rendezvous()
            parked = submit(0)
            self.assertTrue(writer.wait_parked(5.0), "worker never reached rendezvous")
            self.assertEqual(writer.inflight(), 1)
            self.assertEqual(writer.outstanding_leases(), 1)
            self.assertTrue(writer.lease_active(parked))

            queued = [submit(1), submit(2)]
            self.assertEqual(writer.cancel_pending(), 2)
            for job in queued:
                completion = writer.wait(job)
                self.assertFalse(completion[1])
                self.assertIn("cancelled", completion[2])
                writer.release(job)
            # The running job was not cancelled; its lease still blocks teardown.
            self.assertTrue(writer.lease_active(parked))
            self.assertEqual(writer.outstanding_leases(), 1)
            self.assertFalse(writer.drain(0.1), "an in-flight lease must block teardown")

            writer.resume()
            completion = writer.wait(parked)
            self.assertTrue(completion[1], completion[2])
            writer.release(parked)
            self.assertTrue(writer.drain(5.0))
            self.assertEqual(writer.outstanding_leases(), 0)
            stats = dict(writer.stats())
            self.assertEqual(
                stats["leases_released"], stats["leases_established"], "lease leak"
            )
            self.assertEqual(stats["cancelled"], 2)
            # (a) no UAF: source bytes were read while parked and written
            # correctly, and the buffer is only dropped after the lease cleared.
            self.assertEqual(os.pread(data_fd, chunk, 0), arena[0:chunk])
            # The two cancelled jobs never touched the data file: only the
            # parked job's chunk is present.
            self.assertEqual(os.fstat(data_fd).st_size, chunk)
            self.assertEqual(writer.submitted(), 3)
            self.assertEqual(len(writer.poll()), 3)
            del arena
        finally:
            writer.shutdown()
            os.close(data_fd)
            os.close(wal_fd)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_backend_teardown_waits_for_inflight_lease(self):
        """Backend-level H3: `_native_shutdown` refuses to tear down while a
        lease is outstanding, and resolves cleanly once the write completes."""
        backend = ExtentFileBackend(_make_config(self.tmpdir, model_name="h3-teardown"))
        budget, host_pool = _build_mha_pool(5)
        try:
            backend.register_mem_host_pool_v2(host_pool, PoolName.KV)
            pages = host_pool.page_num
            for page in range(pages):
                host_pool.kv_buffer[0, page * 2 : (page + 1) * 2] = float(page + 1)
                host_pool.kv_buffer[1, page * 2 : (page + 1) * 2] = float(page + 100)
            keys = [f"k{i:04d}" for i in range(pages)]
            indices = torch.arange(pages * host_pool.page_size, dtype=torch.int64)
            transfer = PoolTransfer(name=PoolName.KV, host_indices=indices, keys=keys)
            expected = [
                _flat_page_bytes(host_pool, page, host_pool.page_size)
                for page in range(pages)
            ]

            writer = backend._ensure_native_writer()
            writer.arm_test_rendezvous()
            outcome = {}

            def do_write():
                try:
                    outcome["result"] = backend.batch_set_v2([transfer])
                except Exception as exc:  # pragma: no cover - surfaced below
                    outcome["error"] = exc

            worker = threading.Thread(target=do_write, daemon=True)
            worker.start()
            self.assertTrue(writer.wait_parked(5.0), "backend write never parked")
            # (c) while the lease is outstanding, teardown is refused.
            self.assertEqual(backend._native_outstanding_leases(), 1)
            self.assertEqual(backend._native_inflight(), 1)
            self.assertFalse(
                backend._native_shutdown(timeout_s=0.3),
                "teardown must refuse while a page lease is outstanding",
            )
            # The pinned pool allocation is still alive and untouched.
            self.assertEqual(
                _flat_page_bytes(host_pool, 0, host_pool.page_size), expected[0]
            )

            writer.resume()
            worker.join(timeout=30)
            self.assertFalse(worker.is_alive())
            self.assertNotIn("error", outcome)
            self.assertEqual(outcome["result"][PoolName.KV], [True] * pages)
            self.assertEqual(backend._native_outstanding_leases(), 0)
            self.assertTrue(backend._native_shutdown(timeout_s=5.0))
            # (a)/(b): no UAF, no lease leak, recovered bytes correct.
            self.assertEqual(backend._native_outstanding_leases(), 0)
            got = backend.batch_get_v2(
                [PoolTransfer(name=PoolName.KV, host_indices=indices, keys=keys)]
            )
            self.assertEqual(got[PoolName.KV], [True] * pages)
            for page in range(pages):
                self.assertEqual(
                    _flat_page_bytes(host_pool, page, host_pool.page_size),
                    expected[page],
                    f"page {page} corrupted",
                )
            stats = dict(writer.stats())
            self.assertEqual(stats["leases_released"], stats["leases_established"])
        finally:
            backend.close()
            budget.stop()


_CRASH_CHILD = r'''
import os
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig, PoolName, PoolTransfer,
)
from sglang.srt.mem_cache.storage.extent.extent_file_backend import ExtentFileBackend


class PagePool:
    page_size = 1

    def __init__(self, page_bytes, slots=8):
        self._pages = [bytearray([i % 251]) * page_bytes for i in range(slots)]

    def get_data_page(self, index, flat=True):
        return self._pages[index]


storage = os.environ["L3_CRASH_STORAGE"]
cfg = HiCacheStorageConfig(
    extra_config={"storage_dir": storage, "extent_size": "16Ki", "file_size": "64Ki"},
    tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
    attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
    enable_storage_metrics=False, is_page_first_layout=True,
    model_name="crash-parity",
)
backend = ExtentFileBackend(cfg)
pool = PagePool(1000)
backend.register_mem_host_pool_v2(pool, PoolName.KV)
# Data pwrite succeeds; only the WAL fd is broken, so the crash lands in the
# window "data written, no FRAME_SET/BATCH_COMMIT" for BOTH arms.
segment = os.path.join(storage, "hicache.meta%s.wal.0" % backend.config_suffix)
backend._wal_fd = os.open(segment, os.O_RDONLY)
result = backend.batch_set_v2(
    [PoolTransfer(name=PoolName.KV, host_indices=[0], keys=["crashkey"])]
)
with open(os.environ["L3_CRASH_RESULT"], "w") as handle:
    handle.write(repr(result))
# Hard crash: no close(), no checkpoint, no manifest update.
os._exit(0)
'''


class CrashRecoveryParityTests(CustomTestCase):
    """H4 -- arm A vs arm B crash/recovery parity on identical synthetic batches."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="l3_h4_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        os.environ.pop(_NATIVE_ENV, None)

    @staticmethod
    def _wal_and_data(directory: Path, suffix: str):
        wal = directory / f"hicache.meta{suffix}.wal.0"
        data = directory / f"hicache.data{suffix}"
        return wal.read_bytes(), data.read_bytes()

    @unittest.skipIf(torch is None, "torch not installed")
    def test_chunked_batch_arm_parity_after_restart(self):
        """600 pages x 2 blocks = 1200 iovecs (> IOV_MAX) written, restarted,
        and recovered -- committed set and recovered bytes identical per arm."""
        results = {}
        for name, native in (("chunk_a", False), ("chunk_b", True)):
            results[name] = self._run_chunked_arm(name, native)
        arm_a, arm_b = results["chunk_a"], results["chunk_b"]
        self.assertGreaterEqual(
            arm_a["pwrites"], 2, "arm A must shard the 1100 flat pages (>IOV_MAX)"
        )
        self.assertGreaterEqual(
            arm_b["pwrites"], 2, "arm B must shard the 2200 raw blocks (>IOV_MAX)"
        )
        print(
            "H4 chunked: pwrites A=%d B=%d; wal sha A=%s B=%s; data sha A=%s B=%s; "
            "committed A=%d B=%d; recovered==expected A=%s B=%s"
            % (
                arm_a["pwrites"],
                arm_b["pwrites"],
                hashlib.sha256(arm_a["wal"]).hexdigest()[:24],
                hashlib.sha256(arm_b["wal"]).hexdigest()[:24],
                hashlib.sha256(arm_a["data"]).hexdigest()[:24],
                hashlib.sha256(arm_b["data"]).hexdigest()[:24],
                arm_a["committed"],
                arm_b["committed"],
                arm_a["recovered"] == arm_a["expected"],
                arm_b["recovered"] == arm_b["expected"],
            )
        )
        self.assertEqual(arm_a["committed"], 1100)
        self.assertEqual(arm_a["committed"], arm_b["committed"])
        self.assertEqual(arm_a["got_ok"], [True] * 1100)
        self.assertEqual(arm_b["got_ok"], [True] * 1100)
        self.assertEqual(arm_a["wal"], arm_b["wal"], "WAL bytes differ across arms")
        self.assertEqual(arm_a["data"], arm_b["data"], "data bytes differ across arms")
        for page in range(1100):
            self.assertEqual(arm_a["recovered"][page], arm_a["expected"][page])
            self.assertEqual(arm_b["recovered"][page], arm_b["expected"][page])
        self.assertEqual(arm_a["recovered"], arm_b["recovered"])

    def _run_chunked_arm(self, name: str, native: bool):
        directory = self.root / name
        directory.mkdir()
        os.environ[_NATIVE_ENV] = "1" if native else "0"
        # 1100 pages: arm B writes 2 raw blocks/page = 2200 iovecs (> IOV_MAX);
        # arm A gathers one flat buffer/page = 1100 iovecs. Both must chunk.
        pages = 1100
        config = HiCacheStorageConfig(
            extra_config={
                "storage_dir": str(directory),
                "extent_size": "64Mi",
                "file_size": "512Mi",
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
            model_name="parity-chunk",
        )
        budget, host_pool = _build_mha_pool(pages)
        backend = ExtentFileBackend(config)
        try:
            self.assertGreaterEqual(host_pool.page_num, pages)
            backend.register_mem_host_pool_v2(host_pool, PoolName.KV)
            for page in range(pages):
                host_pool.kv_buffer[0, page * 2 : (page + 1) * 2] = float(page + 1)
                host_pool.kv_buffer[1, page * 2 : (page + 1) * 2] = float(page + 100)
            keys = [f"k{i:04d}" for i in range(pages)]
            indices = torch.arange(pages * 2, dtype=torch.int64)
            before = backend.stats_snapshot()
            result = backend.batch_set_v2(
                [PoolTransfer(name=PoolName.KV, host_indices=indices, keys=keys)]
            )
            after = backend.stats_snapshot()
            self.assertEqual(result[PoolName.KV], [True] * pages)
            pwrites = after["pwrites"] - before["pwrites"]
            expected = [_flat_page_bytes(host_pool, page, 2) for page in range(pages)]
            wal_bytes, data_bytes = self._wal_and_data(directory, backend.config_suffix)
        finally:
            backend.close()
        # Restart/rebuild: brand new instance, fresh (zeroed) pool, read from disk.
        budget2, host_pool2 = _build_mha_pool(pages)
        reopened = ExtentFileBackend(config)
        try:
            reopened.register_mem_host_pool_v2(host_pool2, PoolName.KV)
            committed = reopened.batch_exists(keys)
            item = PoolTransfer(name=PoolName.KV, host_indices=indices, keys=keys)
            got = reopened.batch_get_v2([item])[PoolName.KV]
            recovered = [_flat_page_bytes(host_pool2, page, 2) for page in range(pages)]
        finally:
            reopened.close()
            budget.stop()
            budget2.stop()
        return {
            "pwrites": pwrites,
            "wal": wal_bytes,
            "data": data_bytes,
            "committed": committed,
            "recovered": recovered,
            "expected": expected,
            "got_ok": got,
        }

    def _run_crash_arm(self, name: str, native: bool):
        directory = self.root / name
        directory.mkdir()
        result_file = self.root / f"{name}.result"
        env = dict(os.environ)
        env[_NATIVE_ENV] = "1" if native else "0"
        env["L3_CRASH_STORAGE"] = str(directory)
        env["L3_CRASH_RESULT"] = str(result_file)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", _CRASH_CHILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("False", result_file.read_text())
        config = HiCacheStorageConfig(
            extra_config={
                "storage_dir": str(directory),
                "extent_size": "16Ki",
                "file_size": "64Ki",
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
            model_name="crash-parity",
        )
        backend = ExtentFileBackend(config)
        try:
            key = backend._storage_key("crashkey")
            committed = backend.batch_exists([key])
            visible = backend.exists(key)
            wal_bytes, data_bytes = self._wal_and_data(directory, backend.config_suffix)
        finally:
            backend.close()
        return {
            "committed": committed,
            "visible": visible,
            "wal": wal_bytes,
            "data": data_bytes,
        }

    def test_crash_before_commit_arm_parity(self):
        """Both arms crash in the same window (data written, no WAL frame);
        recovery must show the same non-committed state and identical bytes."""
        arm_a = self._run_crash_arm("crash_a", native=False)
        arm_b = self._run_crash_arm("crash_b", native=True)
        print(
            "H4 crash-before-commit: committed A=%d B=%d; visible A=%s B=%s; "
            "wal sha A=%s B=%s; data sha A=%s B=%s"
            % (
                arm_a["committed"],
                arm_b["committed"],
                arm_a["visible"],
                arm_b["visible"],
                hashlib.sha256(arm_a["wal"]).hexdigest()[:24],
                hashlib.sha256(arm_b["wal"]).hexdigest()[:24],
                hashlib.sha256(arm_a["data"]).hexdigest()[:24],
                hashlib.sha256(arm_b["data"]).hexdigest()[:24],
            )
        )
        self.assertEqual(arm_a["committed"], 0)
        self.assertEqual(arm_b["committed"], 0)
        self.assertFalse(arm_a["visible"])
        self.assertFalse(arm_b["visible"])
        self.assertEqual(arm_a["wal"], arm_b["wal"])
        self.assertEqual(arm_a["data"], arm_b["data"])
        self.assertGreater(
            len(arm_a["data"]),
            0,
            "the data pwrite must have happened before the crash",
        )


if __name__ == "__main__":
    unittest.main()
