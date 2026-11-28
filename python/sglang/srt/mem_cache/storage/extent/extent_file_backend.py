# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""ExtentFileBackend: HiCacheStorage over a single preallocated sparse file.

Replaces HiCacheFile's per-key .bin files (a 20K-page restore is ~63K
open/read/close syscalls) with aggregated IO: a batch of consecutive page
writes packs into fixed-size extents (default 64 MiB), so one pwritev covers
the whole batch.  The extent is an *allocation* unit, not a read unit: a
batch read sorts its requested refs by offset and preads each merged exact
range (adjacent ranges with a gap of at most one 4KiB sector merge), so only
the bytes the request covers are read.  The RAM index (``ExtentAllocator``)
maps storage key -> ExtentRef{extent_id, offset, length, generation}; slot
reuse or watermark GC bumps the generation and any stale entry fails the
read-side re-verify (generation mismatch = miss).

Durability contract is unchanged from HiCacheFile: the tree layer only trusts
the ACK, and ACK here means pwrite success + index registration (no fsync,
matching the current semantics; a crash leaves the file orphaned and the
restart starts cold -- no metadata persistence). v1 is the main KV path
(controller batches at STORAGE_BATCH_SIZE=1024): the batch_get_v1/batch_set_v1
zero-copy surface pread/pwrite directly into the host pool buffers that
``get_page_buffer_meta`` describes (k/v dual blocks per page_first MHA page,
one syscall per merged exact range), so the engine-side dummy-page allocation
and per-page scatter loops disappear. v2 keeps the sidecar-pool semantics with
per-pool component keys (``key.<pool>``) packed into their own extents.

This build uses buffered IO only: page lengths are not 4KiB aligned, so only
whole-extent aligned IO could use O_DIRECT, and that tier is not ported.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.storage.extent.extent_allocator import (
    ExtentAllocator,
    ExtentRef,
)
from sglang.srt.mem_cache.storage.extent.extent_metadata import (
    DEFAULT_WAL_MAX_RECORD_BYTES,
    FRAME_BATCH_COMMIT,
    FRAME_CLEAR,
    FRAME_DROP_SLOT,
    FRAME_GROUP_COMMIT,
    FRAME_SET,
    BatchCommitPayload,
    DropSlotPayload,
    GroupCommitPayload,
    ManifestEntry,
    ManifestState,
    RecoveryError,
    SEGMENT_HEADER_SIZE,
    SetEntry,
    SetPayload,
    acquire_meta_lock,
    checkpoint_state_from_allocator,
    component_from_pool_name,
    create_segment,
    current_path,
    decode_batch_commit_payload,
    decode_set_payload,
    deserialize_segment_header,
    encode_clear_payload,
    encode_batch_commit_payload,
    encode_drop_slot_payload,
    encode_frame,
    encode_group_commit_payload,
    encode_set_payload,
    iter_segment_frames,
    lock_path,
    manifest_path,
    promote_pending_rotation,
    publish_current,
    read_current,
    read_manifest,
    repair_active_tail,
    replay_records,
    resolve_manifest_absent,
    resolve_wal_range,
    segment_path,
    validate_geometry,
    write_manifest_atomic,
)
from sglang.srt.mem_cache.storage.file.lru_file_evictor import _parse_size_to_bytes
from sglang.srt.mem_cache.storage.extent.extent_trace import (
    attach_trace_context,
    marker_present,
)
from sglang.srt.utils.common import human_readable_int

logger = logging.getLogger(__name__)


def _extent_read_context(extra_info, pool: str) -> Optional[dict[str, Any]]:
    """Extract only the bounded request identity used for restore-read joins."""
    details = getattr(extra_info, "extra_info", None)
    if not isinstance(details, dict) or not details.get("rid"):
        return None
    return {
        "rid": str(details["rid"]),
        "attempt_id": details.get("attempt_id"),
        "pool": str(details.get("pool", pool)),
        "operation": str(details.get("operation", "prefetch_restore")),
    }


def _log_extent_read(context, keys_n, requested, physical, preads, succeeded):
    """Log request-scoped aggregate Extent reads, never keys or cache data."""
    if not context:
        return
    logger.info(
        "[P3] ts_ms=%d evt=extent_read rid=%s attempt_id=%s operation=%s "
        "component=%s keys_n=%d succeeded_keys=%d requested_bytes=%d "
        "physical_bytes=%d preads=%d",
        int(time.time() * 1000),
        context["rid"],
        context.get("attempt_id") or "none",
        context["operation"],
        context["pool"],
        keys_n,
        succeeded,
        requested,
        physical,
        preads,
    )


# B1 fixed default (ledger: 64MiB tier of the 64/128 decision); env-switchable.
_DEFAULT_EXTENT_SIZE = 64 << 20
# B3 single-file layout: 64 GiB sparse preallocation per rank/suffix.
_DEFAULT_FILE_SIZE = 64 << 30
_DEFAULT_WAL_CHECKPOINT_BYTES = 1 << 30
_WAL_CHECKPOINT_MAX_RETRIES = 3
_WAL_CHECKPOINT_RETRY_DELAY = 0.1
# Read-side merge threshold: adjacent requested ranges whose gap is at most
# one 4KiB sector are merged into a single exact pread.  Absorbing a <=4KiB
# dead gap is bounded amplification (<= 1.13x on a ~31KB page) and keeps
# contiguous page clusters at one syscall; larger gaps stay separate
# exact-range reads so sparse requests never drag in extent fill.
_MERGE_GAP_THRESHOLD = 4096
# Linux caps iovecs per readv/pwritev syscall at IOV_MAX = UIO_MAXIOV =
# 1024 (measured on the 3090 box and locally: 1024 iovecs OK, 1025 EINVAL).
# The v1 batch packs 2 blocks per page, so a 1024-page batch is 2048 iovecs
# and the IO helpers shard at this boundary.
_IOV_MAX = 1024

# --------------------------------------------------------------------------- #
# Native (GIL-free) write path -- L3-NATIVE-WRITEPATH-PROTOTYPE A/B seam.
#
# Select with SGLANG_L3_NATIVE_WRITE_PATH=1. Default 0 = the untouched Python
# reference path (arm A). Arm B moves the page ``pwritev``, the
# FRAME_SET/FRAME_BATCH_COMMIT encoding + append and the ``fdatasync`` onto a
# Rust worker thread that never takes the GIL and never calls back into Python.
# The durable order is unchanged:
#   data pwritev -> WAL frames -> fdatasync -> commit_install.
# Only the seam to <worktree>/rust (the l3wal crate) is added; allocator/index,
# tree and controller layers are untouched.
# --------------------------------------------------------------------------- #
_NATIVE_WRITE_ENV = "SGLANG_L3_NATIVE_WRITE_PATH"
_NATIVE_WRITE_WORKSPACE_ENV = "SGLANG_L3_NATIVE_WRITE_PATH_WORKSPACE"
_NATIVE_WRITE_BUILD_MODE_ENV = "SGLANG_L3_NATIVE_WRITE_BUILD_MODE"
_NATIVE_MODULE = None
_NATIVE_MODULE_LOCK = threading.Lock()


def _native_write_path_enabled() -> bool:
    value = os.environ.get(_NATIVE_WRITE_ENV, "0")
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _native_workspace() -> Path:
    override = os.environ.get(_NATIVE_WRITE_WORKSPACE_ENV)
    if override:
        return Path(override)
    # <worktree>/rust, resolved from <worktree>/python/sglang/srt/.../this file.
    return Path(__file__).resolve().parents[6] / "rust"


def _load_native_module():
    """Import (building offline if needed) the l3wal Rust extension."""
    global _NATIVE_MODULE
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE
    with _NATIVE_MODULE_LOCK:
        if _NATIVE_MODULE is None:
            from sglang.srt.rust_extensions.loader import load_rust_extension

            workspace = _native_workspace()
            _NATIVE_MODULE = load_rust_extension(
                "sglang.srt.rust_extensions._l3wal",
                mode=os.environ.get(_NATIVE_WRITE_BUILD_MODE_ENV, "auto"),
                workspace=workspace,
                # None defers to loader._cache_root() -> SGLANG_CACHE_DIR/rust_extensions,
                # i.e. outside the source tree. Never default inside the repo.
                cache_dir=None,
            )
    return _NATIVE_MODULE


class _NativePageSource:
    """One page's source bytes for the native arm.

    ``blocks`` (raw) is the zero-intermediate-gather-copy form exported by
    ``pool_host.get_page_buffer_meta``: the ``(ptr, len)`` pool blocks in flat
    page order.  ``buffers`` is the fallback form for a host pool without that
    API (or for the explicit ``batch_set(values=...)`` surface): the page
    memoryview is handed straight to the worker, which takes its buffer pointer.
    """

    __slots__ = ("length", "blocks", "buffers")

    def __init__(self, length, blocks=None, buffers=None):
        self.length = length
        self.blocks = blocks
        self.buffers = buffers


def _host_pool_arena_regions(pool):
    """``(base, nbytes)`` regions covering a host pool's stable allocations.

    Every raw ``(ptr, len)`` produced by ``get_page_buffer_meta`` must fall
    inside one of these, so a wrong address is rejected before dereference,
    never silently written.
    """
    regions = []
    for name in ("kv_buffer", "temporal_buffer"):
        tensor = getattr(pool, name, None)
        if tensor is not None and hasattr(tensor, "data_ptr"):
            regions.append(
                (
                    int(tensor.data_ptr()),
                    int(tensor.numel()) * int(tensor.element_size()),
                )
            )
    conv = getattr(pool, "conv_buffer", None)
    if conv is not None:
        try:
            conv_buffers = list(conv)
        except TypeError:
            conv_buffers = []
        for tensor in conv_buffers:
            if tensor is not None and hasattr(tensor, "data_ptr"):
                regions.append(
                    (
                        int(tensor.data_ptr()),
                        int(tensor.numel()) * int(tensor.element_size()),
                    )
                )
    return regions


_T38C_DURABLE_SEQ = {}
_T38C_DURABLE_SEQ_LOCK = threading.Lock()
_PROCESS_META_LOCKS = {}
_PROCESS_META_LOCKS_LOCK = threading.Lock()


@dataclass
class _StagedGroup:
    """Live-only grouped writes, kept out of the allocator until commit."""

    members: list = field(default_factory=list)

    @property
    def entry_count(self) -> int:
        return sum(len(member[0]) for member in self.members)


# T38C paired Mamba trace.  Diagnostic-only and completely disabled unless
# the explicit marker exists; no token text or state bytes are recorded.
def _t38c_log(rec):
    try:
        import json as _json

        if not marker_present("/tmp/t38c-on"):
            return
        rec["ts"] = round(time.time(), 3)
        with open("/tmp/t38c-probe.log", "a") as _f:
            _f.write(_json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _t38c_durable_log(rec):
    """Marker-gated p0_durable.v2 trace; identities/counters only."""
    try:
        if not marker_present("/tmp/t38c-durable-trace"):
            return
        thread_id = threading.get_ident()
        with _T38C_DURABLE_SEQ_LOCK:
            local_seq = _T38C_DURABLE_SEQ.get(thread_id, 0) + 1
            _T38C_DURABLE_SEQ[thread_id] = local_seq
        run_id = os.environ.get("T38C_RUN_ID") or f"pid-{os.getpid()}"
        record = attach_trace_context(dict(rec))
        record.setdefault("schema_version", "p0_durable.v2")
        record.setdefault("run_id", run_id)
        record.setdefault(
            "event_id",
            f"{run_id}:{os.getpid()}:{thread_id}:extent:{local_seq}",
        )
        record.setdefault("process_id", os.getpid())
        record.setdefault("rank", os.environ.get("RANK"))
        record.setdefault("thread_id", thread_id)
        record.setdefault("local_seq", local_seq)
        record.setdefault("operation", record.get("operation_id"))
        record.setdefault("attempt", record.get("attempt_id"))
        for field in (
            "operation_id",
            "attempt_id",
            "node_id",
            "anchor_id",
            "pool",
            "tier",
            "final_kv_hash",
            "storage_key",
            "mamba_key",
            "extent_id",
            "extent_generation",
            "old_generation",
            "new_generation",
            "slot_digest",
            "requested_pages",
            "succeeded_pages",
            "reason",
            "predecessor_event_id",
            "evidence_ref",
        ):
            record.setdefault(field, None)
        record.setdefault("result", None)
        record["ts_wall_ns"] = time.time_ns()
        record["ts_mono_ns"] = time.monotonic_ns()
        record["trace"] = "p0_durable"
        record["ts_ms"] = int(time.time() * 1000)
        with open("/tmp/t38c-probe.log", "a") as _f:
            _f.write(json.dumps(record, default=str) + "\n")
        return record["event_id"]
    except Exception:
        pass


def _t38c_extent_commits(keys, refs, groups, context):
    for extent_id, indexes in groups.items():
        generations = sorted(
            {
                generation
                for index in indexes
                if (generation := getattr(refs[index], "generation", None))
                is not None
            }
        )
        _t38c_durable_log(
            {
                "evt": "extent_commit",
                **(context or {}),
                "extent_id": extent_id,
                "extent_generation": generations[0]
                if len(generations) == 1
                else generations,
                "pages": len(indexes),
            }
        )


def _merge_requested_ranges(items):
    """Coalesce requested (offset, length, index) refs into exact pread ranges.

    Items are sorted by offset; the running range [start, end) absorbs the
    next item when the gap (item.start - end) is at most _MERGE_GAP_THRESHOLD
    bytes.  Returns a list of (start, end, [index...]) where each returned
    range is the exact byte span one pread must cover -- never a whole
    extent, never a span over unrequested bytes beyond the threshold.
    """
    ranges = []
    for offset, length, idx in sorted(items):
        start, end = offset, offset + length
        if ranges and start - ranges[-1][1] <= _MERGE_GAP_THRESHOLD:
            prev = ranges[-1]
            prev[1] = max(prev[1], end)
            prev[2].append(idx)
        else:
            ranges.append([start, end, [idx]])
    return ranges


def _page_byte_view(value) -> memoryview:
    """Zero-copy raw-byte view of a page value.

    torch CPU tensors (the host pool's flat pages) share memory via a
    bit-level uint8 reinterpretation -- ``contiguous().cpu().view(torch.uint8)``
    aliases the storage without copying, and unlike a bare ``.numpy()`` it
    also accepts bf16/fp16 (numpy raises ``TypeError: Got unsupported
    ScalarType BFloat16`` on those dtypes; M2 crash).  Non-CPU tensors are
    copied to CPU only if needed; byte-like buffers are used as-is.
    No torch import at module top.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return memoryview(value)
    if hasattr(value, "contiguous"):
        # torch.Tensor (or tensor-like): bit-level uint8 view, zero-copy for
        # contiguous CPU tensors (the host pool's flat pages).  Same path as
        # HiCacheFile.set (hicache_storage.py), which writes
        # value.contiguous().view(dtype=torch.uint8).numpy().
        import torch

        value = value.contiguous().cpu().view(torch.uint8)
    arr = value.numpy()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy()
    return memoryview(arr).cast("B")


def _target_byte_view(target) -> memoryview:
    """Writable zero-copy byte view of a flat page target."""
    if isinstance(target, (bytearray, memoryview)):
        return memoryview(target)
    if hasattr(target, "contiguous"):
        # torch.Tensor target: bit-level uint8 view so the stored bytes can
        # be filled back into a bf16/fp16/fp32 flat page (mirror of the
        # write side; HiCacheFile.get fills the same way).  Writable for
        # contiguous no-grad CPU tensors.
        import torch

        target = target.contiguous().cpu().view(torch.uint8)
    arr = target.numpy()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy()
    return memoryview(arr).cast("B")


def _config_value(extra: dict, key: str, env_field):
    """extra_config (per-backend) takes precedence over the env var."""
    val = extra.get(key)
    return env_field.get() if val is None else val


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _parse_wal_checkpoint_bytes(value, source: str) -> int:
    """Parse the live-WAL checkpoint threshold without silently disabling it."""
    if isinstance(value, bool):
        raise ValueError(
            f"wal_checkpoint_bytes from {source} must be a non-negative size, not bool"
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"wal_checkpoint_bytes from {source} must not be empty")
        try:
            parsed = human_readable_int(raw)
        except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
            raise ValueError(
                f"invalid wal_checkpoint_bytes from {source}: {value!r}"
            ) from exc
    else:
        raise ValueError(
            f"wal_checkpoint_bytes from {source} must be an integer or size string"
        )
    if parsed < 0:
        raise ValueError(
            f"wal_checkpoint_bytes from {source} must be non-negative, got {parsed}"
        )
    return parsed


class ExtentFileBackend(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, file_path=None):
        extra = storage_config.extra_config or {}
        # --- config_suffix: exact replica of HiCacheFile (hicache_storage.py) ---
        tp_rank, tp_size, pp_rank, pp_size = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
        )
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        model_name = storage_config.model_name
        is_mla_model = storage_config.is_mla_model
        model_name = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = pp_size > 1
        self.config_suffix = f"_{model_name}"
        if not is_mla_model:
            self.config_suffix += f"_{tp_rank}_{tp_size}"
        if enable_pp:
            self.config_suffix += f"_{pp_size}_{pp_rank}"
        # Under NSA context parallel each CP rank holds a disjoint slice of
        # every page, so give each rank its own file to avoid a cross-rank
        # write race (same isolation rationale as HiCacheFile's key suffix).
        if attn_cp_size > 1:
            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"

        self._storage_dir = (
            _config_value(
                extra, "storage_dir", envs.SGLANG_HICACHE_EXTENT_BACKEND_STORAGE_DIR
            )
            or file_path
            or "/tmp/hicache"
        )
        if not os.path.exists(self._storage_dir) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self._storage_dir)
            logger.info(
                f"Created ExtentFileBackend storage directory at {self._storage_dir}"
            )

        extent_size = (
            _parse_size_to_bytes(
                _config_value(
                    extra, "extent_size", envs.SGLANG_HICACHE_EXTENT_BACKEND_EXTENT_SIZE
                )
            )
            or _DEFAULT_EXTENT_SIZE
        )
        file_size = (
            _parse_size_to_bytes(
                _config_value(
                    extra, "file_size", envs.SGLANG_HICACHE_EXTENT_BACKEND_FILE_SIZE
                )
            )
            or _DEFAULT_FILE_SIZE
        )
        if extent_size % 4096 != 0:
            raise ValueError(
                f"extent_size must be a multiple of 4096, got {extent_size}"
            )
        if file_size < extent_size:
            raise ValueError(
                f"file_size {file_size} must be >= extent_size {extent_size}"
            )
        self._extent_size = extent_size
        self._file_size = file_size
        self._max_size_bytes = _parse_size_to_bytes(
            _config_value(extra, "max_size", envs.SGLANG_HICACHE_EXTENT_BACKEND_MAX_SIZE)
        )
        self._min_free_bytes = _parse_size_to_bytes(
            _config_value(
                extra, "min_free_space", envs.SGLANG_HICACHE_EXTENT_BACKEND_MIN_FREE_SPACE
            )
        )
        ratio_raw = extra.get("eviction_ratio")
        try:
            self._eviction_ratio = float(ratio_raw) if ratio_raw is not None else 0.9
        except (TypeError, ValueError):
            self._eviction_ratio = 0.9
        # Buffered IO only: the O_DIRECT calibration tier is not ported (page
        # lengths are not 4KiB aligned, so only whole-extent IO could use it).
        if _as_bool(extra.get("direct_io", False)):
            logger.warning(
                "Extent backend: direct_io is not supported in this build; "
                "continuing with buffered IO."
            )

        threshold_raw = extra.get("wal_checkpoint_bytes")
        if threshold_raw is not None:
            threshold_source = "extra_config"
        elif envs.SGLANG_HICACHE_EXTENT_BACKEND_WAL_CHECKPOINT_BYTES.is_set():
            threshold_raw = envs.SGLANG_HICACHE_EXTENT_BACKEND_WAL_CHECKPOINT_BYTES.get()
            threshold_source = "env"
        else:
            threshold_raw = _DEFAULT_WAL_CHECKPOINT_BYTES
            threshold_source = "default"
        self._wal_checkpoint_bytes = _parse_wal_checkpoint_bytes(
            threshold_raw, threshold_source
        )
        self._wal_checkpoint_source = threshold_source
        self._wal_checkpoint_requested = threshold_raw

        # One data file per rank/config suffix: ranks never share an allocator.
        self._data_path = os.path.join(
            self._storage_dir, f"hicache.data{self.config_suffix}"
        )
        self._manifest_path = manifest_path(self._storage_dir, self.config_suffix)
        self._wal_current_path = current_path(self._storage_dir, self.config_suffix)
        self._meta_lock_path = lock_path(self._storage_dir, self.config_suffix)
        # The lock is itself persistent metadata.  Capture its pre-open
        # existence so the lock file created by this constructor cannot make a
        # genuinely fresh namespace look like a prior initialized one.
        self._meta_lock_existed_before_open = os.path.exists(self._meta_lock_path)
        self._meta_lock_fd = None
        self._meta_lock_owner = False
        self._fd = None
        self._wal_fd = None
        self._wal_generation = None
        self._wal_offset = 0
        self._wal_next_seq = 1
        self._next_batch_id = 1
        self._next_group_seq = 1
        self._staged_groups = {}
        self._staged_reserved = {}
        self._file_uuid = self._new_file_uuid()
        self._namespace_epoch = self._new_nonzero_epoch()
        self._components = {}
        # Cache entries invalidated by durable DROP_SLOT commits are consumed
        # by the scheduler to clear the corresponding Radix L3 beliefs.
        self._evicted_storage_keys = []
        self._constructor_complete = False
        self._closed = False
        self._lifecycle_state = "OPENING"
        self._lifecycle_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._checkpoint_event = threading.Event()
        self._checkpoint_stop = threading.Event()
        self._checkpoint_thread = None
        self._lock = threading.Lock()
        # Lock order is deliberately fixed at _write_gate -> _lock.  The gate
        # quiesces writers while checkpoint rotates and reclaims WAL segments;
        # reads continue to use _lock only.
        self._write_gate = threading.RLock()
        # Native write-path A/B switch (arm A = 0, the Python reference path).
        # The worker is created lazily on the first arm-B write, so an arm-A
        # run neither builds nor imports the Rust extension.
        self._native_write_path = _native_write_path_enabled()
        self._native_writer = None
        self._native_writer_lock = threading.Lock()
        self._native_registered_arenas = set()
        self._stats = {
            "opens": 0,
            "preads": 0,
            "pwrites": 0,
            "bytes_read": 0,
            "bytes_written": 0,
            # Read-amplification accounting (P0 fix: exact-range reads).
            # requested_bytes = ref lengths of read batches; physical_bytes_read
            # = merged pread spans; the ratio target is < 1.2x.  prefetch_bytes_read
            # stays 0: this backend has no prefetch logic, every pread is
            # request-driven (engine-side proactive prefetch arrives as ordinary
            # batch_get/batch_get_v2 and is counted like any other read).
            "requested_bytes": 0,
            "physical_bytes_read": 0,
            "prefetch_bytes_read": 0,
            "last_batch_requested_bytes": 0,
            "last_batch_physical_bytes": 0,
            "last_batch_preads": 0,
            "wal_checkpoint_bytes": self._wal_checkpoint_bytes,
            "wal_checkpoint_requested_bytes": self._wal_checkpoint_requested,
            "wal_checkpoint_source": self._wal_checkpoint_source,
            "wal_active_bytes": 0,
            "wal_checkpoint_requested": 0,
            "wal_checkpoint_successes": 0,
            "wal_checkpoint_failures": 0,
            "wal_checkpoint_retries": 0,
            "wal_checkpoint_last_error": None,
            # Native seam (arm B) failure accounting (H2).
            "native_write_failures": 0,
            "native_write_last_error": None,
        }
        try:
            (
                self._meta_lock_fd,
                self._meta_lock_owner,
                geometry_matches,
            ) = self._acquire_backend_meta_lock()
            self._fd = os.open(self._data_path, os.O_CREAT | os.O_RDWR, 0o644)
            self._stats["opens"] += 1
            os.ftruncate(self._fd, self._file_size)
            self._allocator = ExtentAllocator(
                extent_size, self._file_size // extent_size
            )
            if geometry_matches:
                self._startup_metadata()
            self._open_active_wal()
            self._constructor_complete = True
            self._lifecycle_state = "OPEN"
            self._start_checkpoint_worker()
            logger.info(
                f"ExtentFileBackend: file={self._data_path} extent={extent_size} B "
                f"slots={self._file_size // extent_size} "
                f"max_size={self._max_size_bytes} min_free={self._min_free_bytes} "
                f"wal_checkpoint_bytes={self._wal_checkpoint_bytes} "
                f"wal_checkpoint_requested={self._wal_checkpoint_requested!r} "
                f"wal_checkpoint_source={self._wal_checkpoint_source}"
            )
        except Exception:
            self._checkpoint_stop.set()
            self._checkpoint_event.set()
            if self._checkpoint_thread is not None:
                self._checkpoint_thread.join()
                self._checkpoint_thread = None
            if self._wal_fd is not None:
                os.close(self._wal_fd)
                self._wal_fd = None
            self._release_backend_meta_lock()
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None
            raise

    def _start_checkpoint_worker(self) -> None:
        if self._wal_checkpoint_bytes == 0:
            return
        checkpoint_thread = threading.Thread(
            target=self._checkpoint_worker,
            name=f"extent-wal-checkpoint-{self.config_suffix}",
            daemon=True,
        )
        checkpoint_thread.start()
        self._checkpoint_thread = checkpoint_thread

    def _checkpoint_worker(self) -> None:
        while not self._checkpoint_stop.is_set():
            if not self._checkpoint_event.wait(0.5):
                continue
            # Clear the request under the same lifecycle lock used by the
            # signaler.  A signal that arrives during checkpoint processing is
            # not cleared afterward, so it remains a distinct retry request.
            with self._lifecycle_lock:
                self._checkpoint_event.clear()
                if self._lifecycle_state != "OPEN":
                    return
            if self._checkpoint_stop.is_set():
                return
            self._stats["wal_checkpoint_requested"] += 1
            succeeded = False
            for attempt in range(_WAL_CHECKPOINT_MAX_RETRIES):
                if self._checkpoint_stop.is_set():
                    return
                try:
                    # checkpoint() takes _write_gate itself; no backend lock is
                    # held across this call, preserving the existing lock order.
                    if not self.checkpoint(preserve_incomplete_groups=True):
                        break
                    self._stats["wal_checkpoint_successes"] += 1
                    self._stats["wal_active_bytes"] = max(
                        0, self._wal_offset - SEGMENT_HEADER_SIZE
                    )
                    succeeded = True
                    break
                except Exception as exc:
                    self._stats["wal_checkpoint_failures"] += 1
                    self._stats["wal_checkpoint_last_error"] = repr(exc)
                    if attempt + 1 < _WAL_CHECKPOINT_MAX_RETRIES:
                        self._stats["wal_checkpoint_retries"] += 1
                        if self._checkpoint_stop.wait(_WAL_CHECKPOINT_RETRY_DELAY):
                            return
            # Do not clear here: a signal received while the attempt was in
            # flight is a newer request and must survive this failure/decline.

    def _signal_wal_checkpoint_if_needed(self) -> None:
        """Signal only after the writer's _lock-protected boundary completes."""
        if self._wal_checkpoint_bytes <= 0:
            return
        if self._wal_offset - SEGMENT_HEADER_SIZE < self._wal_checkpoint_bytes:
            return
        with self._lifecycle_lock:
            if self._lifecycle_state == "OPEN":
                self._checkpoint_event.set()

    @staticmethod
    def _new_nonzero_epoch() -> bytes:
        while True:
            epoch = secrets.token_bytes(16)
            if any(epoch):
                return epoch

    @staticmethod
    def _new_file_uuid() -> bytes:
        while True:
            file_uuid = secrets.token_bytes(16)
            if any(file_uuid):
                return file_uuid

    def _acquire_backend_meta_lock(self):
        """Acquire the per-namespace fence for this backend instance."""
        with _PROCESS_META_LOCKS_LOCK:
            held = _PROCESS_META_LOCKS.get(self._meta_lock_path)
            if held is not None:
                raise RecoveryError(
                    "metadata lock is already held by this process",
                    path=self._meta_lock_path,
                    failure_class="lock-busy",
                )
            fd = acquire_meta_lock(self._meta_lock_path)
            _PROCESS_META_LOCKS[self._meta_lock_path] = (
                fd,
                1,
                (self._extent_size, self._file_size),
            )
            return fd, True, True

    def _release_backend_meta_lock(self) -> None:
        fd = self._meta_lock_fd
        if fd is None:
            return
        self._meta_lock_fd = None
        with _PROCESS_META_LOCKS_LOCK:
            owner_fd, count, geometry = _PROCESS_META_LOCKS[self._meta_lock_path]
            count -= 1
            if fd != owner_fd:
                os.close(fd)
            if count == 0:
                _PROCESS_META_LOCKS.pop(self._meta_lock_path, None)
                os.close(owner_fd)
            else:
                _PROCESS_META_LOCKS[self._meta_lock_path] = (owner_fd, count, geometry)

    def _validate_segment(self, path: str, generation: int) -> None:
        try:
            with open(path, "rb") as stream:
                raw = stream.read(SEGMENT_HEADER_SIZE)
            header = deserialize_segment_header(raw)
            if header.segment_generation != generation:
                raise RecoveryError(
                    "WAL segment generation does not match its name",
                    path=path,
                    failure_class="segment-corruption",
                )
        except RecoveryError:
            raise
        except Exception as exc:
            raise RecoveryError(
                "WAL segment has an invalid header",
                path=path,
                failure_class="segment-corruption",
            ) from exc

    def _restore_allocator(self, state: ManifestState, replay) -> None:
        num_slots = self._file_size // self._extent_size
        if len(replay.generations) != num_slots:
            raise RecoveryError(
                "replay generation geometry mismatch",
                path=self._manifest_path,
                failure_class="wal-replay-state",
            )
        refs = []
        keys = []
        slot_keys = [set() for _ in range(num_slots)]
        slot_bytes = [0] * num_slots
        cursors = [0] * num_slots
        index = {}
        component_map = {}
        for entry in replay.entries:
            try:
                key = entry.storage_key.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RecoveryError(
                    "WAL storage key is not valid UTF-8",
                    path=self._manifest_path,
                    failure_class="wal-replay-corruption",
                ) from exc
            if not 0 <= entry.extent_id < num_slots:
                raise RecoveryError(
                    "replayed entry is outside allocator geometry",
                    path=self._manifest_path,
                    failure_class="wal-replay-state",
                )
            if (
                entry.offset < 0
                or entry.length < 0
                or entry.offset + entry.length > self._extent_size
                or entry.generation != replay.generations[entry.extent_id]
            ):
                raise RecoveryError(
                    "replayed entry does not match allocator state",
                    path=self._manifest_path,
                    failure_class="wal-replay-state",
                )
            ref = ExtentRef(
                entry.extent_id, entry.offset, entry.length, entry.generation
            )
            index[key] = ref
            component_map[key] = entry.component

        for key, ref in index.items():
            slot_keys[ref.extent_id].add(key)
            slot_bytes[ref.extent_id] += ref.length
            cursors[ref.extent_id] = max(cursors[ref.extent_id], ref.offset + ref.length)

        if slot_bytes != list(replay.slot_bytes) or cursors != list(replay.cursors):
            raise RecoveryError(
                "replayed allocator accounting mismatch",
                path=self._manifest_path,
                failure_class="wal-replay-state",
            )
        if not 0 <= state.next_slot < num_slots:
            raise RecoveryError(
                "manifest next_slot is outside allocator geometry",
                path=self._manifest_path,
                failure_class="manifest-corruption",
            )

        recency = []
        seen = set()
        for entry_index in state.recency_order:
            if not 0 <= entry_index < len(replay.entries):
                raise RecoveryError(
                    "manifest recency entry is outside the entry table",
                    path=self._manifest_path,
                    failure_class="manifest-corruption",
                )
            try:
                key = replay.entries[entry_index].storage_key.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RecoveryError(
                    "manifest recency key is not valid UTF-8",
                    path=self._manifest_path,
                    failure_class="manifest-corruption",
                ) from exc
            if key in index and key not in seen:
                recency.append(key)
                seen.add(key)
        recency.extend(key for key in index if key not in seen)

        with self._allocator._lock:
            self._allocator._generations = list(replay.generations)
            self._allocator._cursors = cursors
            self._allocator._slot_bytes = slot_bytes
            self._allocator._slot_keys = slot_keys
            self._allocator._index = index
            self._allocator._next_slot = state.next_slot
            self._allocator._recency = OrderedDict((key, None) for key in recency)
        self._components = component_map

    def _startup_metadata(self) -> None:
        data_stat = os.fstat(self._fd)
        num_slots = self._file_size // self._extent_size
        manifest_exists = os.path.exists(self._manifest_path)
        current_exists = os.path.exists(self._wal_current_path)
        wal_zero_path = segment_path(self._storage_dir, self.config_suffix, 0)
        wal_exists_without_manifest = os.path.exists(wal_zero_path)

        state = None
        active_generation = None
        if not manifest_exists:
            if (
                not current_exists
                and not wal_exists_without_manifest
                and not self._meta_lock_existed_before_open
            ):
                file_uuid = self._new_file_uuid()
                namespace_epoch = self._new_nonzero_epoch()
                try:
                    create_segment(wal_zero_path, 0, 0)
                    publish_current(self._wal_current_path, 0)
                    state = ManifestState(
                        checkpoint_lsn=0,
                        wal_floor=0,
                        wal_active_gen=0,
                        extent_size=self._extent_size,
                        file_size=self._file_size,
                        num_slots=num_slots,
                        data_dev=data_stat.st_dev,
                        data_ino=data_stat.st_ino,
                        file_uuid=file_uuid,
                        namespace_epoch=namespace_epoch,
                        next_slot=0,
                        generations=[0] * num_slots,
                        cursors=[0] * num_slots,
                        slot_bytes=[0] * num_slots,
                    )
                    write_manifest_atomic(self._manifest_path, state)
                except RecoveryError:
                    raise
                except OSError as exc:
                    raise RecoveryError(
                        "unable to initialize metadata",
                        path=self._manifest_path,
                        failure_class="metadata-io",
                    ) from exc
                self._restore_allocator(
                    state,
                    replay_records([], state, num_slots=num_slots),
                )
                self._file_uuid = state.file_uuid
                self._namespace_epoch = state.namespace_epoch
                return
            if not current_exists:
                resolve_manifest_absent(None, prior_metadata_exists=True)
            active_generation = read_current(self._wal_current_path)
            resolve_manifest_absent(active_generation, prior_metadata_exists=True)
        else:
            state = read_manifest(self._manifest_path)
            try:
                validate_geometry(
                    state,
                    extent_size=self._extent_size,
                    file_size=self._file_size,
                    num_slots=num_slots,
                    data_dev=data_stat.st_dev,
                    data_ino=data_stat.st_ino,
                )
            except Exception as exc:
                if isinstance(exc, RecoveryError):
                    raise
                raise RecoveryError(
                    str(exc),
                    path=self._manifest_path,
                    failure_class="manifest-corruption",
                ) from exc
            if not any(state.namespace_epoch):
                raise RecoveryError(
                    "manifest namespace_epoch is all zero",
                    path=self._manifest_path,
                    failure_class="manifest-corruption",
                )
            active_generation = read_current(self._wal_current_path)
            wal_range = resolve_wal_range(state, active_generation)
            if wal_range.pending_rotation:
                active_generation = promote_pending_rotation(
                    self._storage_dir,
                    self.config_suffix,
                    active_generation,
                    state.checkpoint_lsn,
                )
                wal_range = resolve_wal_range(state, active_generation)

        if state is None:
            state = ManifestState(
                extent_size=self._extent_size,
                file_size=self._file_size,
                num_slots=num_slots,
                data_dev=data_stat.st_dev,
                data_ino=data_stat.st_ino,
                file_uuid=self._new_file_uuid(),
                namespace_epoch=self._new_nonzero_epoch(),
                generations=[0] * num_slots,
                cursors=[0] * num_slots,
                slot_bytes=[0] * num_slots,
            )
            wal_range = resolve_manifest_absent(active_generation, prior_metadata_exists=False)

        segment_frames = []
        for generation in wal_range.generations:
            path = segment_path(self._storage_dir, self.config_suffix, generation)
            self._validate_segment(path, generation)
            frames = iter_segment_frames(
                path,
                active=generation == active_generation,
            )
            if frames.torn_tail is not None:
                repair_active_tail(path, frames.torn_tail.last_valid_offset)
            segment_frames.append(frames)
        replay = replay_records(
            segment_frames,
            state if manifest_exists else None,
            num_slots=num_slots,
        )
        self._restore_allocator(state, replay)
        self._wal_recovered_next_seq = max(
            replay.seen_max_seq, state.checkpoint_lsn
        ) + 1

        persisted = state
        persisted.namespace_epoch = self._new_nonzero_epoch()
        persisted.data_dev = data_stat.st_dev
        persisted.data_ino = data_stat.st_ino
        self._file_uuid = persisted.file_uuid
        self._namespace_epoch = persisted.namespace_epoch
        try:
            write_manifest_atomic(self._manifest_path, persisted)
        except RecoveryError:
            raise
        except OSError as exc:
            raise RecoveryError(
                "unable to persist startup namespace epoch",
                path=self._manifest_path,
                failure_class="manifest-io",
            ) from exc

    def _open_active_wal(self) -> None:
        """Open exactly the generation named by the durable current pointer."""
        generation = read_current(self._wal_current_path)
        path = segment_path(self._storage_dir, self.config_suffix, generation)
        fd = os.open(path, os.O_RDWR)
        try:
            frames = iter_segment_frames(fd, active=True)
            if frames.torn_tail is not None:
                repair_active_tail(path, frames.torn_tail.last_valid_offset)
                frames = iter_segment_frames(fd, active=True)
            self._wal_fd = fd
            self._wal_generation = generation
            self._wal_offset = os.fstat(fd).st_size
            self._wal_next_seq = max(
                (frames[-1].seq + 1) if frames else 1,
                getattr(self, "_wal_recovered_next_seq", 1),
            )
            self._wal_recovered_next_seq = self._wal_next_seq
            self._stats["wal_active_bytes"] = max(
                0, self._wal_offset - SEGMENT_HEADER_SIZE
            )
            max_batch_id = 0
            for frame in frames:
                if frame.frame_type == FRAME_SET:
                    max_batch_id = max(max_batch_id, decode_set_payload(frame.payload).batch_id)
                elif frame.frame_type == FRAME_BATCH_COMMIT:
                    max_batch_id = max(
                        max_batch_id,
                        decode_batch_commit_payload(frame.payload).batch_id,
                    )
            self._next_batch_id = max_batch_id + 1
        except Exception:
            os.close(fd)
            raise

    def _append_wal_frame(self, frame_type: int, payload: bytes) -> None:
        if self._wal_fd is None:
            raise OSError("active WAL is not open")
        frame = encode_frame(frame_type, self._wal_next_seq, payload)
        written = 0
        while written < len(frame):
            count = os.pwrite(self._wal_fd, frame[written:], self._wal_offset + written)
            if count <= 0:
                raise OSError("short WAL append")
            written += count
        self._wal_offset += written
        self._wal_next_seq += 1
        self._stats["wal_active_bytes"] = max(
            0, self._wal_offset - SEGMENT_HEADER_SIZE
        )

    def _sync_wal(self) -> None:
        if self._wal_fd is None:
            raise OSError("active WAL is not open")
        os.fdatasync(self._wal_fd)

    @staticmethod
    def _validate_group_id(group_id) -> tuple:
        if not isinstance(group_id, tuple) or len(group_id) != 2:
            raise ValueError("group_id must be a (16-byte epoch, positive sequence) tuple")
        epoch, seq = group_id
        if not isinstance(epoch, bytes) or len(epoch) != 16 or not any(epoch):
            raise ValueError("group_id epoch must be a nonzero 16-byte value")
        if not isinstance(seq, int) or isinstance(seq, bool) or not 1 <= seq <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("group_id sequence must be a positive u64")
        return epoch, seq

    def open_group(self) -> tuple:
        """Mint a backend-owned group identity for one runtime incarnation."""
        with self._write_gate:
            group_id = (self._namespace_epoch, self._next_group_seq)
            self._next_group_seq += 1
            self._staged_groups[group_id] = _StagedGroup()
            return group_id

    def _plan_split_with_staged(self, key_lengths, pending_drops=None):
        """Plan against allocator cursors plus invisible staged reservations."""
        saved_cursors = self._allocator._cursors
        saved_next_slot = self._allocator._next_slot
        cursors = list(saved_cursors)
        next_slot = saved_next_slot
        for refs in self._staged_reserved.values():
            for ref in refs:
                cursors[ref.extent_id] = max(
                    cursors[ref.extent_id], ref.offset + ref.length
                )
                next_slot = ref.extent_id
        self._allocator._cursors = cursors
        self._allocator._next_slot = next_slot
        try:
            return self._allocator.plan_split(key_lengths, pending_drops)
        finally:
            self._allocator._cursors = saved_cursors
            self._allocator._next_slot = saved_next_slot

    def _staged_slots(self):
        return {ref.extent_id for refs in self._staged_reserved.values() for ref in refs}

    def _stage_group_members(self, group_id, storage_keys, refs, component, drop_events):
        staged = self._staged_groups.get(group_id)
        if staged is None:
            return False
        if any(ref is None for ref in refs):
            return False
        staged.members.append((list(storage_keys), list(refs), component, list(drop_events)))
        self._staged_reserved.setdefault(group_id, []).extend(refs)
        return True

    def _abort_group_locked(self, group_id) -> bool:
        staged = self._staged_groups.pop(group_id, None)
        self._staged_reserved.pop(group_id, None)
        return staged is not None

    def commit_group(self, group_id) -> bool:
        """Durably complete and atomically install one staged group."""
        group_id = self._validate_group_id(group_id)
        with self._write_gate:
            if group_id[0] != self._namespace_epoch or group_id not in self._staged_groups:
                return False
            staged = self._staged_groups[group_id]
            if not staged.members or any(not member[0] for member in staged.members):
                return False
            member_count = staged.entry_count
            self._append_wal_frame(
                FRAME_GROUP_COMMIT,
                encode_group_commit_payload(
                    GroupCommitPayload(group_id[0], group_id[1], member_count)
                ),
            )
            self._sync_wal()
            keys = [key for member in staged.members for key in member[0]]
            refs = [ref for member in staged.members for ref in member[1]]
            with self._lock:
                if not self._allocator.commit_install(keys, refs):
                    return False
                for member in staged.members:
                    self._components.update(
                        (key, member[2]) for key in member[0]
                    )
            self._abort_group_locked(group_id)
            self._signal_wal_checkpoint_if_needed()
            return True

    def abort_group(self, group_id) -> bool:
        group_id = self._validate_group_id(group_id)
        with self._write_gate:
            aborted = self._abort_group_locked(group_id)
            if aborted:
                self._signal_wal_checkpoint_if_needed()
            return aborted

    def _abort_all_groups_locked(self) -> None:
        self._staged_groups.clear()
        self._staged_reserved.clear()

    def _append_drops(self, drop_events) -> None:
        for slot, new_generation in drop_events:
            self._append_wal_frame(
                FRAME_DROP_SLOT,
                encode_drop_slot_payload(DropSlotPayload(slot, new_generation)),
            )
        if drop_events:
            self._sync_wal()

    def _append_batch_commit(
        self, storage_keys, refs, component: int, group_id=None
    ) -> None:
        if group_id is None:
            group_epoch, group_seq = b"\x00" * 16, 0
        else:
            group_epoch, group_seq = self._validate_group_id(group_id)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        entries = [
            SetEntry(
                storage_key=key.encode("utf-8"),
                extent_id=ref.extent_id,
                offset=ref.offset,
                length=ref.length,
                generation=ref.generation,
                component=component,
            )
            for key, ref in zip(storage_keys, refs)
        ]
        self._append_wal_frame(
            FRAME_SET,
            encode_set_payload(
                SetPayload(
                    group_epoch=group_epoch,
                    group_seq=group_seq,
                    batch_id=batch_id,
                    member_index=0,
                    member_count=1,
                    entries=entries,
                )
            ),
        )
        self._append_wal_frame(
            FRAME_BATCH_COMMIT,
            encode_batch_commit_payload(BatchCommitPayload(batch_id, 1)),
        )
        # A staged member is invisible until GROUP_COMMIT makes it durable, so
        # its own frame needs no fsync; one sync at commit covers the group.
        if group_id is None:
            self._sync_wal()

    def _append_ungrouped_commit(self, storage_keys, refs, component: int) -> None:
        self._append_batch_commit(storage_keys, refs, component)

    def _append_relocation_commit(self, relocated) -> None:
        """Durably publish relocated refs with the existing WAL protocol.

        ``relocated`` is an ordered list of ``(key, ExtentRef, component)``.
        Exactly one SET frame plus one BATCH_COMMIT frame are appended and
        fsynced (no new frame type, no new protocol): replay folds the batch
        only when its commit is present, and the later SET deterministically
        wins over the victim's earlier SET for the same key.  Components are
        carried per entry so KV and Mamba payloads keep their opaque pool
        identity across the move.
        """
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        entries = [
            SetEntry(
                storage_key=key.encode("utf-8"),
                extent_id=ref.extent_id,
                offset=ref.offset,
                length=ref.length,
                generation=ref.generation,
                component=component,
            )
            for key, ref, component in relocated
        ]
        self._append_wal_frame(
            FRAME_SET,
            encode_set_payload(
                SetPayload(
                    group_epoch=b"\x00" * 16,
                    group_seq=0,
                    batch_id=batch_id,
                    member_index=0,
                    member_count=1,
                    entries=entries,
                )
            ),
        )
        self._append_wal_frame(
            FRAME_BATCH_COMMIT,
            encode_batch_commit_payload(BatchCommitPayload(batch_id, 1)),
        )
        self._sync_wal()

    # ---- key scoping (same semantics as HiCacheFile) ----
    # v1 and v2 share the same transform: the engine writes via v1 batch_set
    # and queries via v2 batch_exists_v2, so both sides must produce the same
    # storage key (bare keys on one side only = storage_hit always 0).

    def _storage_key(self, key: str) -> str:
        return key + self.config_suffix

    def _component_key(self, pool_name, key: str) -> str:
        if pool_name is None or pool_name == "__default__" or pool_name == PoolName.KV:
            return self._storage_key(key)
        return self._storage_key(f"{key}.{pool_name}")

    # ---- v1: single-key surface ----

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ):
        storage_key = self._storage_key(key)
        page = self._read_pages([storage_key])[0]
        if page is None or target_location is None:
            return None
        view = _target_byte_view(target_location)
        if len(view) != len(page):
            logger.warning(
                f"ExtentFileBackend: size mismatch for {storage_key}: "
                f"stored {len(page)} B vs target {len(view)} B"
            )
            return None
        view[:] = page
        return target_location

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if value is None:
            return False
        storage_keys = [self._storage_key(key)]
        ok = self._write_pages(storage_keys, [value])
        if ok:
            # Telemetry parity (P1 hygiene): the single-key set() must be
            # metered exactly like batch_set / batch_set_v1 / batch_set_v2,
            # or a future single-key write path (e.g. a mamba 51.5MB page)
            # would silently undercount the extent_write bytes (~34% of the
            # write volume at 90G/1341 slots in the A/B evidence). The write
            # path ignores node identity, so None is passed (same as the
            # batch kv path).
            self._log_extent_write("kv", None, storage_keys, [value])
        return ok

    def exists(self, key: str) -> bool:
        with self._lock:
            return self._allocator.lookup(self._storage_key(key)) is not None

    # ---- v1: batched surface (main KV path) ----

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[Any]:
        storage_keys = [self._storage_key(key) for key in keys]
        pages = self._read_pages(storage_keys)
        targets = target_locations or [None] * len(keys)
        out = []
        for page, target in zip(pages, targets):
            if page is None or target is None:
                out.append(None)
                continue
            view = _target_byte_view(target)
            if len(view) != len(page):
                logger.warning(
                    f"ExtentFileBackend: size mismatch for a page: "
                    f"stored {len(page)} B vs target {len(view)} B"
                )
                out.append(None)
                continue
            view[:] = page
            out.append(target)
        return out

    def _log_extent_write(
        self, component, node_id, storage_keys, sizes
    ) -> None:
        """[P3] extent_write instrumentation: one record per successful
        extent (component) batch. ``sizes`` are the per-key byte lengths.
        Node identity comes from the caller's extra_info (instrumentation-
        only; the write path ignores it)."""
        try:
            total = sum(len(v) for v in sizes)
        except TypeError:
            total = sum(sizes)
        logger.info(
            "[P3] ts_ms=%d evt=extent_write node=%s component=%s "
            "keys_n=%d bytes=%d",
            int(time.time() * 1000),
            node_id,
            component,
            len(storage_keys),
            total,
        )

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if values is None or len(values) != len(keys):
            return False
        storage_keys = [self._storage_key(key) for key in keys]
        ok = self._write_pages(storage_keys, values)
        if ok:
            self._log_extent_write("kv", None, storage_keys, values)
        return ok

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        with self._lock:
            for i, key in enumerate(keys):
                if self._allocator.lookup(self._storage_key(key)) is None:
                    return i
            return len(keys)

    # ---- v1: zero-copy surface (anchor KV pool, direct-to-buffer IO) ----

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Zero-copy read: preadv merged exact ranges into the host pool.

        The target buffers come from ``get_page_buffer_meta(host_indices)``;
        the stored flat page bytes are the concatenation of the per-page
        blocks in pool order (k block then v block for a page_first MHA
        pool), so one preadv per merged exact range scatters the whole batch
        into the pool with no intermediate page buffers.  Dead gap bytes
        inside a merged range land in scratch bytearrays and are discarded.
        A generation mismatch (slot reused since the snapshot) or a short
        read marks the affected keys as misses; the engine never consumes
        those host slots, so partially written bytes are harmless.  Returns
        one bool per key; never raises, never returns None.
        """
        if len(keys) == 0:
            return []
        meta = self._v1_buffer_meta(keys, host_indices, "batch_get_v1")
        if meta is None:
            return [False] * len(keys)
        ptr_list, size_list, blocks_per_page, _ = meta
        storage_keys = [self._storage_key(k) for k in keys]
        page_lens = [
            sum(size_list[i * blocks_per_page : (i + 1) * blocks_per_page])
            for i in range(len(keys))
        ]
        blocks = self._v1_host_blocks(ptr_list, size_list, blocks_per_page)
        with self._lock:
            refs = self._allocator.snapshot(storage_keys)
        # A ref whose length differs from the current pool layout was written
        # under a different layout; refusing beats scattering bytes into the
        # wrong blocks.
        items = [
            (ref.offset, ref.length, i)
            for i, ref in enumerate(refs)
            if ref is not None and ref.length == page_lens[i]
        ]
        groups: dict = {}
        for offset, length, i in items:
            groups.setdefault(refs[i].extent_id, []).append((offset, length, i))
        ok = [False] * len(keys)
        requested = 0
        physical = 0
        preads = 0
        for extent_id, triples in groups.items():
            requested += sum(t[1] for t in triples)
            for start, end, idx_list in _merge_requested_ranges(triples):
                iovecs = []
                prev_end = start
                for i in idx_list:
                    ref = refs[i]
                    gap = ref.offset - prev_end
                    if gap > 0:
                        iovecs.append(bytearray(gap))
                    iovecs.extend(blocks[i])
                    prev_end = ref.offset + ref.length
                if (
                    self._preadv_full(extent_id * self._extent_size + start, iovecs)
                    != end - start
                ):
                    logger.error(
                        f"ExtentFileBackend: short pread for extent {extent_id}"
                    )
                    continue
                physical += end - start
                preads += 1
                for i in idx_list:
                    ok[i] = True
        with self._lock:
            verify_ok = self._allocator.verify(storage_keys, refs)
            for i, key in enumerate(storage_keys):
                if ok[i] and verify_ok[i]:
                    self._allocator.touch(key)
        self._stats["requested_bytes"] += requested
        self._stats["physical_bytes_read"] += physical
        self._stats["last_batch_requested_bytes"] = requested
        self._stats["last_batch_physical_bytes"] = physical
        self._stats["last_batch_preads"] = preads
        _log_extent_read(
            _extent_read_context(extra_info, "kv"),
            len(keys),
            requested,
            physical,
            preads,
            sum(bool(o and v) for o, v in zip(ok, verify_ok)),
        )
        return [o and v for o, v in zip(ok, verify_ok)]

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
        group_id=None,
    ) -> List[bool]:
        """Zero-copy write; ``group_id`` stages the pages for a later commit."""
        with self._write_gate:
            if group_id is not None:
                try:
                    group_id = self._validate_group_id(group_id)
                except ValueError:
                    return [False] * len(keys)
                if (
                    group_id[0] != self._namespace_epoch
                    or group_id not in self._staged_groups
                ):
                    return [False] * len(keys)
            result = self._batch_set_v1_locked(
                keys, host_indices, extra_info, group_id
            )
            self._signal_wal_checkpoint_if_needed()
            return result

    def _batch_set_v1_locked(
        self,
        keys: List[str],
        host_indices,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
        group_id=None,
    ) -> List[bool]:
        """Zero-copy write: pwritev the host pool blocks into packed extents.

        The per-page blocks (k block then v block for page_first MHA) are
        pwritten in pool order at each ref, reconstructing the flat page
        bytes the read paths and the engine's flat-page API expect.  The
        plan -> pwrite -> commit sequence holds the index lock (same
        all-or-nothing semantics as ``_write_pages``); a refused batch
        returns all False.
        """
        if len(keys) == 0:
            return []
        meta = self._v1_buffer_meta(keys, host_indices, "batch_set_v1")
        if meta is None:
            return [False] * len(keys)
        ptr_list, size_list, blocks_per_page, _ = meta
        storage_keys = [self._storage_key(k) for k in keys]
        page_lens = [
            sum(size_list[i * blocks_per_page : (i + 1) * blocks_per_page])
            for i in range(len(keys))
        ]
        blocks = self._v1_host_blocks(ptr_list, size_list, blocks_per_page)
        with self._lock:
            pending_drops = self._select_gc_victims()
            refs, passed, drop_events = self._plan_split_with_staged(
                list(zip(storage_keys, page_lens)), pending_drops
            )
            if any(r is None for r in refs):
                logger.warning(
                    "ExtentFileBackend: batch_set_v1 of %s pages refused "
                    "(page larger than extent_size %s B)",
                    len(keys),
                    self._extent_size,
                )
                return [False] * len(keys)
            # Reclaim-time relocation: a circular-reclaim victim's live refs
            # must have a durable replacement before its generation is bumped.
            if not self._relocate_reclaim_victims_locked(passed, drop_events, refs):
                return [False] * len(keys)
            evicted_keys = self._keys_for_drop_events(drop_events)
            try:
                self._append_drops(drop_events)
                self._allocator.apply_drops(
                    drop_events,
                    trace_context=getattr(self, "_t38c_durable_context", None),
                )
                self._evicted_storage_keys.extend(evicted_keys)
                for slot, _generation in drop_events:
                    for key in list(self._components):
                        ref = self._allocator._index.get(key)
                        if ref is None and key not in self._allocator._index:
                            self._components.pop(key, None)
            except Exception as exc:
                logger.error("ExtentFileBackend: WAL invalidation failed: %s", exc)
                return [False] * len(keys)
            groups: dict = {}
            for i, ref in enumerate(refs):
                groups.setdefault(ref.extent_id, []).append(i)
            for extent_id, idxs in groups.items():
                ordered = sorted(idxs, key=lambda i: refs[i].offset)
                first = refs[ordered[0]].offset
                iovecs = []
                for i in ordered:
                    iovecs.extend(blocks[i])
                if not self._pwritev(extent_id * self._extent_size + first, iovecs):
                    return [False] * len(keys)
            kv_component = component_from_pool_name(PoolName.KV)
            try:
                self._append_batch_commit(
                    storage_keys, refs, kv_component, group_id=group_id
                )
            except Exception as exc:
                logger.error("ExtentFileBackend: WAL commit failed: %s", exc)
                return [False] * len(keys)
            if group_id is not None:
                return (
                    [True] * len(keys)
                    if self._stage_group_members(
                        group_id, storage_keys, refs, kv_component, drop_events
                    )
                    else [False] * len(keys)
                )
            trace_context = getattr(self, "_t38c_durable_context", None) or {
                "pool": "kv",
                "final_kv_hash": keys[-1] if keys else "",
            }
            if not self._allocator.commit_install(storage_keys, refs):
                return [False] * len(keys)
            self._components.update(
                zip(
                    storage_keys,
                    [component_from_pool_name(PoolName.KV)] * len(storage_keys),
                )
            )
            _t38c_extent_commits(
                storage_keys,
                refs,
                groups,
                getattr(self, "_t38c_durable_context", None)
                or {
                    "pool": "kv",
                    "final_kv_hash": keys[-1] if keys else "",
                },
            )
            self._log_extent_write(
                "kv", getattr(extra_info, "node_id", None), storage_keys, page_lens
            )
            return [True] * len(keys)

    def _v1_buffer_meta(self, keys, host_indices, op_name):
        """Resolve the v1 host-pool buffer contract; None on any violation.

        v1 targets the anchor KV pool (``register_mem_pool_host``): the pool
        must expose ``get_page_buffer_meta``, host_indices must carry
        page_size slots per key, and the per-key block count must divide
        evenly.  The scatter relies on the flat page bytes being the
        concatenation of the blocks in pool order -- true for the
        page-first family (MHA k/v dual blocks, Mamba temporal+conv, MLA
        single block) but not for layer_first, which interleaves per-layer
        k/v blocks; layer_first is refused rather than corrupted.
        """
        pool = getattr(self, "mem_pool_host", None)
        if pool is None or not callable(getattr(pool, "get_page_buffer_meta", None)):
            logger.error(
                "ExtentFileBackend %s: no host pool with get_page_buffer_meta "
                "registered (register_mem_pool_host).",
                op_name,
            )
            return None
        page_size = getattr(pool, "page_size", 1) or 1
        if len(host_indices) != len(keys) * page_size:
            logger.error(
                "ExtentFileBackend %s: host_indices length %s != keys %s x "
                "page_size %s",
                op_name,
                len(host_indices),
                len(keys),
                page_size,
            )
            return None
        if getattr(pool, "layout", "") == "layer_first":
            logger.error(
                "ExtentFileBackend %s: layer_first pool layout is not "
                "supported by the v1 flat-page scatter; refusing.",
                op_name,
            )
            return None
        try:
            ptr_list, size_list = pool.get_page_buffer_meta(host_indices)
        except Exception as e:
            logger.error(
                "ExtentFileBackend %s: get_page_buffer_meta failed: %s", op_name, e
            )
            return None
        if not ptr_list or len(ptr_list) != len(size_list):
            logger.error(
                "ExtentFileBackend %s: empty or mismatched buffer meta", op_name
            )
            return None
        blocks_per_page = len(ptr_list) // len(keys)
        if len(ptr_list) % len(keys) != 0 or blocks_per_page == 0:
            logger.error(
                "ExtentFileBackend %s: %s buffer blocks for %s keys is not a "
                "whole per-key count",
                op_name,
                len(ptr_list),
                len(keys),
            )
            return None
        return ptr_list, size_list, blocks_per_page, page_size

    def _v1_host_blocks(self, ptr_list, size_list, blocks_per_page):
        """ctypes arrays aliasing the host pool blocks for direct fd IO."""
        blocks = []
        for i in range(0, len(ptr_list), blocks_per_page):
            blocks.append(
                [
                    (ctypes.c_char * size_list[i + j]).from_address(ptr_list[i + j])
                    for j in range(blocks_per_page)
                ]
            )
        return blocks

    # ---- v2: sidecar-pool surface ----

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Restorable prefix lengths, as a set intersection over pools.

        A TRAILING_PAGES pool (SWA window, Mamba/Conv checkpoint) is only
        restorable at the discrete lengths whose trailing window is on disk, and
        two such pools can hold windows that never coincide. Folding each pool
        to one number and taking the minimum would then return a length that at
        least one pool cannot serve; the pools' restorable sets are intersected
        instead, so the answer is the longest *jointly* restorable prefix.

        Existence comes from the allocator index (``lookup`` honours the slot
        generation, so circular reuse reads as absent); everything else follows
        ``HiCacheFile.batch_exists_v2``.
        """
        with self._lock:

            def present(key: str) -> bool:
                return self._allocator.lookup(key) is not None

            def has_component(page_idx: int, name) -> bool:
                return present(self._component_key(name, keys[page_idx]))

            # Longest contiguous KV prefix present in storage.
            kv_pages = next(
                (i for i in range(len(keys)) if not present(self._storage_key(keys[i]))),
                len(keys),
            )
            hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
            restorable = list(range(1, kv_pages + 1))

            for transfer in pool_transfers or []:
                if not restorable:
                    break
                name = transfer.name
                # A PREFETCH transfer's keys are placeholders at build time
                # (mamba_component / swa_component write node-last checkpoints;
                # the real trailing hashes are filled only after the hit query
                # by _sync_trailing_keys). Such a transfer must use the
                # TRAILING scan: these pools store only a trailing window, so
                # reading them as ALL_PAGES -- the PoolTransfer default, which
                # takes over when a transfer reconstruction drops the policy --
                # fails at the first non-window page and clamps kv_hit to 0.
                placeholder_keys = bool(transfer.keys) and all(
                    k == "__placeholder__" for k in transfer.keys
                )
                trailing_pages = transfer.hit_policy in (
                    PoolHitPolicy.TRAILING_PAGES,
                    None,
                )
                if not trailing_pages and not placeholder_keys:
                    boundary = next(
                        (
                            i
                            for i in range(kv_pages)
                            if not has_component(i, name)
                        ),
                        kv_pages,
                    )
                    pool_restorable = set(range(1, boundary + 1))
                else:
                    trailing = max(
                        1, min(self._trailing_transfer_pages(transfer), kv_pages)
                    )
                    pool_restorable = {
                        prefix_len
                        for prefix_len in range(1, kv_pages + 1)
                        if all(
                            has_component(i, name)
                            for i in range(
                                max(0, prefix_len - trailing), prefix_len
                            )
                        )
                    }
                    boundary = max(pool_restorable, default=0)
                if boundary:
                    hit_count[name] = boundary
                restorable = [page for page in restorable if page in pool_restorable]

            final_pages = restorable[-1] if restorable else 0
            return PoolTransferResult(final_pages, hit_count, restorable)

    def _trailing_transfer_pages(self, transfer: PoolTransfer) -> int:
        """Return the number of sidecar pages represented by a transfer."""
        host_indices = transfer.host_indices
        if host_indices is not None:
            try:
                slots = int(host_indices.numel())
            except AttributeError:
                slots = len(host_indices)
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            page_size = getattr(host_pool, "page_size", 1) or 1
            if slots > 0 and slots % page_size == 0:
                return max(1, slots // page_size)
        return max(1, len(transfer.keys) if transfer.keys else 1)

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        results = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            if host_indices is None or len(host_indices) != len(keys) * page_size:
                logger.error(
                    "batch_get_v2 indices length mismatch for %s: "
                    "expected %s, got %s",
                    transfer.name,
                    len(keys) * page_size,
                    len(host_indices) if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                if transfer.name == PoolName.MAMBA:
                    _t38c_log(
                        {
                            "evt": "pc_mamba_write",
                            "node_id": getattr(transfer, "_t38c_node_id", ""),
                            "final_kv_hash": getattr(
                                transfer, "_t38c_final_kv_hash", ""
                            ),
                            "mamba_transfer_exists": True,
                            "mamba_key": keys[-1] if keys else "",
                            "batch_set_v2": False,
                        }
                    )
                continue
            storage_keys = [self._component_key(transfer.name, k) for k in keys]
            pages = self._read_pages(
                storage_keys,
                read_context=_extent_read_context(
                    extra_info, getattr(transfer.name, "value", transfer.name)
                ),
            )
            dummy = host_pool.get_dummy_flat_data_page()
            per_key = []
            for i, page in enumerate(pages):
                if page is None:
                    per_key.append(False)
                    continue
                dummy_view = _target_byte_view(dummy)
                if len(dummy_view) != len(page):
                    per_key.append(False)
                    continue
                dummy_view[:] = page
                host_pool.set_from_flat_data_page(
                    int(host_indices[i * page_size]), dummy
                )
                per_key.append(True)
            results[transfer.name] = per_key
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
        group_id=None,
    ) -> dict:
        if group_id is not None:
            try:
                group_id = self._validate_group_id(group_id)
            except ValueError:
                return {transfer.name: [False] * len(transfer.keys or []) for transfer in transfers}
        results = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            if host_indices is None or len(host_indices) != len(keys) * page_size:
                logger.error(
                    "batch_set_v2 indices length mismatch for %s: "
                    "expected %s, got %s",
                    transfer.name,
                    len(keys) * page_size,
                    len(host_indices) if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                _t38c_durable_log(
                    {
                        "evt": "batch_set_result",
                        "trace_context": getattr(
                            transfer, "_t38c_trace_context", None
                        ),
                        "operation_id": getattr(transfer, "_t38c_operation_id", None),
                        "attempt_id": getattr(transfer, "_t38c_attempt_id", None),
                        "node_id": getattr(transfer, "_t38c_node_id", ""),
                        "final_kv_hash": getattr(
                            transfer, "_t38c_final_kv_hash", ""
                        ),
                        "pool": getattr(transfer.name, "value", transfer.name),
                        "requested_pages": len(keys),
                        "succeeded_pages": 0,
                        "result": False,
                        "reason": "host_index_geometry",
                    }
                )
                continue
            storage_keys = [self._component_key(transfer.name, k) for k in keys]
            page_sources = None
            if self._native_write_path:
                page_sources = self._native_page_sources_from_meta(
                    host_pool, host_indices, len(keys), page_size
                )
            values = None
            if page_sources is None:
                values = [
                    host_pool.get_data_page(int(host_indices[i * page_size]), flat=True)
                    for i in range(len(keys))
                ]
            trace_context = {
                **(
                    getattr(transfer, "_t38c_trace_context", None) or {}
                ),
                "operation_id": getattr(transfer, "_t38c_operation_id", None),
                "attempt_id": getattr(transfer, "_t38c_attempt_id", None),
                "node_id": getattr(transfer, "_t38c_node_id", ""),
                "final_kv_hash": getattr(transfer, "_t38c_final_kv_hash", ""),
                "pool": getattr(transfer.name, "value", transfer.name),
            }
            ok = self._write_pages(
                storage_keys,
                values,
                trace_context,
                group_id=group_id,
                component=component_from_pool_name(transfer.name),
                page_sources=page_sources,
            )
            results[transfer.name] = [ok] * len(keys)
            _t38c_durable_log(
                {
                    "evt": "batch_set_result",
                    **trace_context,
                    "requested_pages": len(keys),
                    "succeeded_pages": len(keys) if ok else 0,
                    "result": bool(ok),
                }
            )
            if transfer.name == PoolName.MAMBA:
                _t38c_log(
                    {
                        "evt": "pc_mamba_write",
                        "node_id": getattr(transfer, "_t38c_node_id", ""),
                        "final_kv_hash": getattr(
                            transfer, "_t38c_final_kv_hash", ""
                        ),
                        "mamba_transfer_exists": True,
                        "mamba_key": keys[-1] if keys else "",
                        "batch_set_v2": bool(ok),
                    }
                )
            if ok:
                self._log_extent_write(
                    transfer.name,
                    getattr(extra_info, "node_id", None),
                    storage_keys,
                    values
                    if values is not None
                    else [source.length for source in page_sources],
                )
        return results

    # ---- core aggregated IO ----

    def _read_pages(
        self, keys: List[str], read_context: Optional[dict[str, Any]] = None
    ) -> List[Optional[memoryview]]:
        """Aggregate-read pages: one pread per merged exact range.

        Snapshot refs under the lock -> pread each merged range without the
        lock -> re-verify generation under the lock (a slot reused in between
        fails verification, and its half-read data is discarded = miss).

        Read amplification fix (P0, expert ruling): the extent is an
        allocation unit, not a read unit.  Requested refs are sorted by
        offset within each extent and merged only when the gap between
        adjacent ranges is at most ``_MERGE_GAP_THRESHOLD`` bytes; each
        merged range is pread exactly, so dead bytes between requested slots
        are no longer read (M3 measured ~30x physical/requested with the
        previous whole-extent-span read).
        """
        with self._lock:
            refs = self._allocator.snapshot(keys)
        groups: dict = {}
        for i, ref in enumerate(refs):
            if ref is None:
                continue
            groups.setdefault(ref.extent_id, []).append(i)
        requested = 0
        physical = 0
        preads = 0
        scratch: dict = {}
        for extent_id, idxs in groups.items():
            items = sorted((refs[i].offset, refs[i].length, i) for i in idxs)
            ranges = _merge_requested_ranges(items)
            requested += sum(length for _, length, _ in items)
            found: dict = {}
            for start, end, idx_list in ranges:
                buf = bytearray(end - start)
                if (
                    self._pread_full(buf, extent_id * self._extent_size + start)
                    != len(buf)
                ):
                    logger.error(
                        f"ExtentFileBackend: short pread for extent {extent_id}"
                    )
                    # Whole extent misses (same semantics as before); the
                    # ranges read so far are discarded with `found`.
                    found = {}
                    break
                physical += end - start
                preads += 1
                for i in idx_list:
                    found[i] = (buf, refs[i].offset - start)
            scratch.update(found)
        with self._lock:
            ok = self._allocator.verify(keys, refs)
            for i, key in enumerate(keys):
                if ok[i]:
                    self._allocator.touch(key)
        out: List[Optional[memoryview]] = [None] * len(keys)
        for i, ref in enumerate(refs):
            if ref is None or not ok[i]:
                continue
            entry = scratch.get(i)
            if entry is None:
                continue
            buf, start = entry
            out[i] = memoryview(buf)[start : start + ref.length]
        # Read-amplification accounting (physical/requested < 1.2x target).
        self._stats["requested_bytes"] += requested
        self._stats["physical_bytes_read"] += physical
        self._stats["last_batch_requested_bytes"] = requested
        self._stats["last_batch_physical_bytes"] = physical
        self._stats["last_batch_preads"] = preads
        _log_extent_read(
            read_context,
            len(keys),
            requested,
            physical,
            preads,
            sum(page is not None for page in out),
        )
        return out

    # ---- reclaim-time live-ref relocation ----

    def drain_evicted_storage_keys(self):
        """Return keys invalidated by committed slot drops since last drain."""
        with self._lock:
            keys = self._evicted_storage_keys
            self._evicted_storage_keys = []
            # A successful live-ref relocation publishes replacement refs
            # before the victim DROP_SLOT. Such keys remain readable and must
            # not invalidate their Radix durability claims.
            return [
                key[: -len(self.config_suffix)]
                for key in keys
                if key not in self._allocator._index
                and key.endswith(self.config_suffix)
            ]

    def _keys_for_drop_events(self, drop_events):
        slots = {slot for slot, _generation in drop_events}
        return [
            key
            for slot in sorted(slots)
            for key in sorted(self._allocator._slot_keys[slot])
            if (ref := self._allocator._index.get(key)) is not None
            and ref.extent_id == slot
            and ref.generation == self._allocator._generations[slot]
        ]

    def _reclaim_victims(self, passed, drop_events):
        """Enumerate the live refs held by circular-reclaim victim slots.

        A victim is a slot the pure planner put in ``passed``: its generation
        is about to be bumped and its bytes reused.  Only refs that currently
        resolve (present in the index, in this exact slot, at the slot's
        current generation) are returned; dead/stale entries are never copied.
        """
        drop_slots = {slot for slot, _ in drop_events}
        order = []
        seen = set()
        for slot in passed:
            if slot in seen or slot not in drop_slots:
                continue
            seen.add(slot)
            order.append(slot)
        victims = []
        for slot in sorted(order):
            generation = self._allocator._generations[slot]
            for key in sorted(self._allocator._slot_keys[slot]):
                ref = self._allocator._index.get(key)
                if ref is None or ref.extent_id != slot:
                    continue
                if ref.generation != generation:
                    continue
                victims.append((key, ref))
        return victims

    def _relocate_reclaim_victims_locked(self, passed, drop_events, planned_refs):
        """Move every live ref out of the reclaim victims before they drop.

        Preserves the existing durability order exactly: exact bytes are
        pread from the victim and pwritten to non-victim space, then the
        replacements are published with SET -> BATCH_COMMIT -> fsync and
        installed.  Only after that may the caller append the DROP_SLOT
        frames and bump the victim generation.  Returns False (fail closed)
        when no destination capacity can be secured; the victim is then left
        untouched and the whole write is refused.
        """
        victims = self._reclaim_victims(passed, drop_events)
        if not victims:
            return True
        # Destination space must never be a victim (about to be dropped), a
        # slot carrying this plan's own refs (its pwrite/`_cursors` update
        # follows), or a staged reservation (its bytes are already written
        # but its refs are not installed yet).
        excluded = {slot for slot, _ in drop_events}
        excluded.update(ref.extent_id for ref in planned_refs if ref is not None)
        excluded.update(self._staged_slots())
        start_slot = max(
            (ref.extent_id for ref in planned_refs if ref is not None),
            default=self._allocator._next_slot,
        )
        key_lengths = [(key, ref.length) for key, ref in victims]
        destination_refs = self._allocator.plan_relocation(
            [slot for slot in {ref.extent_id for _key, ref in victims}],
            key_lengths,
            excluded_slots=sorted(excluded),
            start_slot=start_slot,
        )
        if destination_refs is None:
            # Extent is a finite cache, not a permanent store. When safe
            # relocation cannot fit, drop this LRU victim atomically with the
            # incoming write. DROP_SLOT invalidates its generation before any
            # bytes are reused; concurrent readers verify their snapshot and
            # turn a raced read into a miss. The scheduler consumes the dropped
            # keys and clears every dependent Radix durability claim.
            logger.info(
                "ExtentFileBackend: evicting %d LRU live refs from slots %s "
                "to satisfy circular allocation (no relocation tail space)",
                len(victims),
                sorted({ref.extent_id for _key, ref in victims}),
            )
            return True
        # Exact bytes, read from the victim and written to the replacement.
        for (key, ref), destination in zip(victims, destination_refs):
            buffer = bytearray(ref.length)
            if (
                self._pread_full(
                    buffer, ref.extent_id * self._extent_size + ref.offset
                )
                != ref.length
            ):
                logger.error(
                    "ExtentFileBackend: reclaim relocation read failed for %s", key
                )
                return False
            if not self._pwritev(
                destination.extent_id * self._extent_size + destination.offset,
                [memoryview(buffer)],
            ):
                logger.error(
                    "ExtentFileBackend: reclaim relocation write failed for %s", key
                )
                return False
        relocated = [
            (
                key,
                destination,
                self._components.get(key, component_from_pool_name(PoolName.KV)),
            )
            for (key, _ref), destination in zip(victims, destination_refs)
        ]
        try:
            self._append_relocation_commit(relocated)
        except Exception as exc:
            logger.error(
                "ExtentFileBackend: reclaim relocation WAL commit failed: %s", exc
            )
            return False
        if not self._allocator.commit_install(
            [key for key, _ref in victims], destination_refs
        ):
            logger.error(
                "ExtentFileBackend: reclaim relocation install failed; "
                "refusing the write"
            )
            return False
        logger.info(
            "ExtentFileBackend: reclaim relocated %d live refs out of victim "
            "slots %s (%d bytes)",
            len(victims),
            sorted({ref.extent_id for _key, ref in victims}),
            sum(ref.length for _key, ref in victims),
        )
        return True

    def _write_pages(
        self,
        keys: List[str],
        values: List[Any],
        trace_context=None,
        group_id=None,
        component=None,
        page_sources=None,
    ) -> bool:
        with self._write_gate:
            result = self._write_pages_locked(
                keys,
                values,
                trace_context,
                group_id=group_id,
                component=component,
                page_sources=page_sources,
            )
            self._signal_wal_checkpoint_if_needed()
            return result

    def _write_pages_locked(
        self,
        keys: List[str],
        values: List[Any],
        trace_context=None,
        group_id=None,
        component=None,
        page_sources=None,
    ) -> bool:
        """Pack pages into extents and pwrite, all-or-nothing.

        Watermark GC runs first (under the lock); the whole plan->pwrite->
        commit sequence holds the index lock, so readers cannot snapshot a
        half-applied write and no state is mutated if a pwrite fails (ACK =
        pwrite success + index registration).
        """
        if page_sources is None:
            if len(keys) != len(values) or any(v is None for v in values):
                return False
        elif len(page_sources) != len(keys):
            return False
        if group_id is not None:
            try:
                group_id = self._validate_group_id(group_id)
            except ValueError:
                return False
            if group_id[0] != self._namespace_epoch or group_id not in self._staged_groups:
                return False
        with self._lock:
            if page_sources is not None:
                # Arm B with a raw meta source: skip the Python gather entirely.
                page_lens = [source.length for source in page_sources]
                views = None
            else:
                views = [_page_byte_view(v) for v in values]
                page_lens = [len(v) for v in views]
            pending_drops = self._select_gc_victims()
            refs, passed, drop_events = self._plan_split_with_staged(
                list(zip(keys, page_lens)), pending_drops
            )
            if any(slot in self._staged_slots() for slot, _ in drop_events):
                return False
            if any(r is None for r in refs):
                logger.warning(
                    "ExtentFileBackend: batch of %s pages refused (page "
                    "larger than extent_size %s B)",
                    len(keys),
                    self._extent_size,
                )
                return False
            # Reclaim-time relocation: a circular-reclaim victim's live refs
            # must have a durable replacement before its generation is bumped.
            if not self._relocate_reclaim_victims_locked(passed, drop_events, refs):
                return False
            evicted_keys = self._keys_for_drop_events(drop_events)
            try:
                self._append_drops(drop_events)
                self._allocator.apply_drops(drop_events, trace_context=trace_context)
                self._evicted_storage_keys.extend(evicted_keys)
            except Exception as exc:
                logger.error("ExtentFileBackend: WAL invalidation failed: %s", exc)
                return False
            groups: dict = {}
            for i, ref in enumerate(refs):
                groups.setdefault(ref.extent_id, []).append(i)
            pool = (trace_context or {}).get("pool", PoolName.KV)
            persisted_component = (
                component
                if component is not None
                else component_from_pool_name(pool)
            )
            if self._native_write_path:
                # Arm B: data pwritev + WAL frames + fdatasync on the Rust worker.
                if not self._native_write_locked(
                    keys,
                    page_sources,
                    views,
                    refs,
                    groups,
                    persisted_component,
                    group_id,
                ):
                    return False
            else:
                for extent_id, idxs in groups.items():
                    first = refs[idxs[0]].offset
                    bufs = [views[i] for i in idxs]
                    if not self._pwritev(extent_id * self._extent_size + first, bufs):
                        return False
                try:
                    self._append_batch_commit(
                        keys, refs, persisted_component, group_id=group_id
                    )
                except Exception as exc:
                    logger.error("ExtentFileBackend: WAL commit failed: %s", exc)
                    return False
            if group_id is not None:
                return self._stage_group_members(
                    group_id,
                    keys,
                    refs,
                    persisted_component,
                    drop_events,
                )
            if not self._allocator.commit_install(keys, refs):
                return False
            self._components.update(
                zip(keys, [persisted_component] * len(keys))
            )
            _t38c_extent_commits(keys, refs, groups, trace_context)
            return True

    # ---- native write path (arm B) ----

    def _ensure_native_writer(self):
        """Lazily create the Rust worker for this backend instance."""
        if self._native_writer is not None:
            return self._native_writer
        with self._native_writer_lock:
            if self._native_writer is None:
                module = _load_native_module()
                self._native_writer = module.Writer()
        return self._native_writer

    def _register_pool_arena(self, pool) -> bool:
        """Register the pool's stable allocation(s) as pointer-validity arenas."""
        regions = _host_pool_arena_regions(pool)
        if not regions:
            return False
        writer = self._ensure_native_writer()
        for base, length in regions:
            if (base, length) not in self._native_registered_arenas:
                writer.register_arena(int(base), int(length))
                self._native_registered_arenas.add((base, length))
        return True

    def _native_inflight(self) -> int:
        """Native jobs queued or running (0 == no worker can touch a pointer)."""
        if self._native_writer is None:
            return 0
        return int(self._native_writer.inflight())

    def _native_outstanding_leases(self) -> int:
        """Page leases established but not released; must be 0 before teardown."""
        if self._native_writer is None:
            return 0
        return int(self._native_writer.outstanding_leases())

    def _native_shutdown(self, timeout_s: float = -1.0) -> bool:
        """**H3** teardown barrier: drain in-flight native work, resolve every
        outstanding lease, join the worker, and only then close the fds it
        writes through. Returns False if a lease is still outstanding after the
        join (fail-closed: the caller must not free the pinned allocation).
        """
        writer = self._native_writer
        if writer is None:
            return True
        if self._native_inflight():
            if not writer.drain(timeout_s):
                logger.error(
                    "ExtentFileBackend: native worker did not drain within %ss; "
                    "refusing to tear down while %s lease(s) are outstanding",
                    timeout_s,
                    writer.outstanding_leases(),
                )
                return False
        writer.shutdown()
        outstanding = int(writer.outstanding_leases())
        if outstanding:
            logger.error(
                "ExtentFileBackend: %s native lease(s) unresolved after join; "
                "pinned memory must not be freed",
                outstanding,
            )
            return False
        return True

    def _native_page_sources_from_meta(self, pool, host_indices, count, page_size):
        """Raw per-page ``(ptr, len)`` blocks from ``get_page_buffer_meta``.

        Returns ``None`` when the pool cannot supply a validated raw source, in
        which case the caller falls back to the arm-A gather + buffer submit.
        """
        if not callable(getattr(pool, "get_page_buffer_meta", None)):
            return None
        if getattr(pool, "layout", "") == "layer_first":
            return None
        if not self._register_pool_arena(pool):
            return None
        try:
            ptr_list, size_list = pool.get_page_buffer_meta(host_indices)
        except Exception as exc:
            logger.error("ExtentFileBackend: native meta gather failed: %s", exc)
            return None
        if not ptr_list or len(ptr_list) != len(size_list):
            return None
        if len(ptr_list) % count != 0:
            return None
        blocks_per_page = len(ptr_list) // count
        if blocks_per_page == 0:
            return None
        sources = []
        for page in range(count):
            blocks = [
                (
                    int(ptr_list[page * blocks_per_page + j]),
                    int(size_list[page * blocks_per_page + j]),
                )
                for j in range(blocks_per_page)
            ]
            sources.append(
                _NativePageSource(
                    sum(length for _, length in blocks), blocks=blocks
                )
            )
        return sources

    def _native_write_locked(
        self, keys, page_sources, views, refs, groups, component, group_id
    ) -> bool:
        """Arm B: run the whole data + WAL durable sequence on the Rust worker.

        The worker owns page pwritev, frame encoding, WAL append and fdatasync;
        it never calls back into Python.  ``commit_install`` still runs in
        Python only after this returns (invariant 3).
        """
        if page_sources is None:
            page_sources = [
                _NativePageSource(len(view), buffers=[view]) for view in views
            ]
        raw = page_sources[0].blocks is not None
        plans = []
        for extent_id, idxs in groups.items():
            ordered = sorted(idxs, key=lambda index: refs[index].offset)
            first = refs[ordered[0]].offset
            if raw:
                spec = []
                for index in ordered:
                    spec.extend(page_sources[index].blocks)
            else:
                spec = []
                for index in ordered:
                    spec.extend(page_sources[index].buffers)
            plans.append((int(extent_id * self._extent_size + first), spec))
        if group_id is None:
            group_epoch, group_seq = b"\x00" * 16, 0
        else:
            group_epoch, group_seq = self._validate_group_id(group_id)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        entries = [
            (
                key.encode("utf-8"),
                int(ref.extent_id),
                int(ref.offset),
                int(ref.length),
                int(ref.generation),
                int(component),
            )
            for key, ref in zip(keys, refs)
        ]
        writer = self._ensure_native_writer()
        common = dict(
            data_fd=self._fd,
            wal_fd=self._wal_fd,
            wal_offset=int(self._wal_offset),
            wal_seq=int(self._wal_next_seq),
            group_epoch=group_epoch,
            group_seq=group_seq,
            batch_id=batch_id,
            member_index=0,
            member_count=1,
            entries=entries,
            wal_max_record_bytes=DEFAULT_WAL_MAX_RECORD_BYTES,
        )
        try:
            if raw:
                job_id = writer.submit_write_raw(plans=plans, **common)
            else:
                job_id = writer.submit_write_buffers(plans=plans, **common)
        except Exception as exc:
            logger.error("ExtentFileBackend: native submit failed: %s", exc)
            return False
        completion = writer.wait(job_id)
        if completion is None:
            # H2: a missing completion is a protocol violation, not success.
            logger.error(
                "ExtentFileBackend: native write job %s produced no completion; "
                "failing the batch closed",
                job_id,
            )
            return False
        (
            _job_id,
            ok,
            error,
            errno,
            bytes_written,
            wal_offset,
            wal_seq,
            data_syscalls,
        ) = completion
        # Advance the shared WAL cursor even on failure: a frame the worker
        # already appended must not be overwritten (same invariant as
        # _append_wal_frame's per-frame accounting).
        if wal_offset >= self._wal_offset:
            self._wal_offset = wal_offset
            self._wal_next_seq = wal_seq
            self._stats["wal_active_bytes"] = max(
                0, wal_offset - SEGMENT_HEADER_SIZE
            )
        if data_syscalls:
            self._stats["pwrites"] += data_syscalls
            self._stats["bytes_written"] += bytes_written
        # Invariant 3: completion arrived, so the page lease may (and must) be
        # released -- on failure too, otherwise a refused batch would keep a
        # stale lease on its pages forever.
        writer.release(job_id)
        if not ok:
            # H2: explicit failure (errno/message) is propagated as a whole-batch
            # refusal. The caller returns False to the engine, no WAL frame
            # beyond what the worker durably appended is assumed, and
            # commit_install is never reached.
            self._stats["native_write_failures"] += 1
            self._stats["native_write_last_error"] = (
                f"errno={errno} {error}" if errno else error
            )
            logger.error(
                "ExtentFileBackend: native write failed (errno=%s): %s; "
                "batch refused (no commit_install)",
                errno,
                error,
            )
            return False
        return True

    # ---- GC / lifecycle ----

    def _select_gc_victims(self):
        """Select LRU victim slots without mutating allocator state."""
        if self._max_size_bytes <= 0 and self._min_free_bytes <= 0:
            return []
        victims = []
        selected = set()
        used = self._allocator.used_bytes
        # Filesystem free space is invariant during this read-only victim
        # selection (the write has not started yet). Sample it once rather
        # than issuing statvfs for every LRU key. Walk OrderedDict directly:
        # copying every key and pop(0) made a full-pressure scan quadratic.
        free_bytes = None
        if self._min_free_bytes > 0:
            st = os.statvfs(self._storage_dir)
            free_bytes = st.f_bavail * st.f_frsize
        free_space_pressure = (
            free_bytes is not None and free_bytes < self._min_free_bytes
        )
        for key in self._allocator._recency:
            pressure = (
                self._max_size_bytes > 0
                and used > int(self._max_size_bytes * self._eviction_ratio)
            ) or free_space_pressure
            if not pressure:
                break
            ref = self._allocator._index.get(key)
            if ref is None or ref.extent_id in selected:
                continue
            selected.add(ref.extent_id)
            victims.append(ref.extent_id)
            used -= self._allocator._slot_bytes[ref.extent_id]
        return victims

    def _gc_evict(self, trace_context=None) -> None:
        """Lazy watermark GC: evict LRU slots until under both watermarks.

        Pressure = live bytes above max_size x eviction_ratio, or free disk
        below min_free_space; eviction is synchronous on the write path, no
        background daemon.
        """
        if self._max_size_bytes <= 0 and self._min_free_bytes <= 0:
            return
        # Free space is re-read after each reclaimed slot: this loop runs once
        # per evicted extent (bytes needed / extent_size, a handful), not once
        # per LRU key, so the statvfs cost stays bounded while the sweep still
        # stops as soon as free space recovers instead of over-evicting to a
        # stale pressure reading.
        while True:
            pressure = (
                self._max_size_bytes > 0
                and self._allocator.used_bytes
                > int(self._max_size_bytes * self._eviction_ratio)
            )
            if not pressure and self._min_free_bytes > 0:
                st = os.statvfs(self._storage_dir)
                pressure = st.f_bavail * st.f_frsize < self._min_free_bytes
            if not pressure:
                return
            evicted = self._allocator.evict_slot_lru(trace_context=trace_context)
            if evicted == 0:
                return

    def _checkpoint_state(self, wal_floor: int, wal_active_gen: int) -> ManifestState:
        """Snapshot the ungrouped allocator state while ``_lock`` is held."""
        allocator = self._allocator
        entries = []
        entry_indexes = {}
        for key, ref in allocator._index.items():
            entry_indexes[key] = len(entries)
            entries.append(
                ManifestEntry(
                    storage_key=key.encode("utf-8"),
                    extent_id=ref.extent_id,
                    offset=ref.offset,
                    length=ref.length,
                    generation=ref.generation,
                    component=self._components.get(
                        key, component_from_pool_name(PoolName.KV)
                    ),
                )
            )
        return checkpoint_state_from_allocator(
            checkpoint_lsn=self._wal_next_seq - 1,
            wal_floor=wal_floor,
            wal_active_gen=wal_active_gen,
            extent_size=self._extent_size,
            file_size=self._file_size,
            num_slots=self._file_size // self._extent_size,
            generations=list(allocator._generations),
            cursors=list(allocator._cursors),
            slot_bytes=list(allocator._slot_bytes),
            entries=entries,
            groups=(),
            data_dev=os.fstat(self._fd).st_dev,
            data_ino=os.fstat(self._fd).st_ino,
            file_uuid=self._file_uuid,
            namespace_epoch=self._namespace_epoch,
            next_slot=allocator._next_slot,
            recency_order=[entry_indexes[key] for key in allocator._recency if key in entry_indexes],
        )

    def checkpoint(self, preserve_incomplete_groups: bool = False) -> bool:
        """Publish a checkpoint, rotate WAL, then unlink covered segments.

        The write gate remains held for the full manifest-publish,
        current-pointer-publish, and exact-named unlink sequence.  No write can
        append to the old generation after the published floor.
        """
        with self._write_gate:
            if preserve_incomplete_groups and self._staged_groups:
                return False
            self._abort_all_groups_locked()
            with self._lock:
                old_generation = self._wal_generation
                new_generation = old_generation + 1
                state = self._checkpoint_state(new_generation, old_generation)
                write_manifest_atomic(self._manifest_path, state)

            create_segment(
                segment_path(self._storage_dir, self.config_suffix, new_generation),
                new_generation,
                state.checkpoint_lsn,
            )
            publish_current(self._wal_current_path, new_generation)

            old_fd = self._wal_fd
            self._wal_fd = None
            if old_fd is not None:
                os.close(old_fd)
            self._wal_generation = new_generation
            self._wal_offset = SEGMENT_HEADER_SIZE
            self._wal_next_seq = state.checkpoint_lsn + 1
            # _open_active_wal() max()es against this watermark; a stale value
            # renumbers the new segment below the published checkpoint lsn.
            self._wal_recovered_next_seq = self._wal_next_seq
            self._open_active_wal()

            # Recovery intentionally does not enumerate this directory.  The
            # checkpoint floor is the complete reclamation set.
            for generation in range(new_generation):
                path = segment_path(self._storage_dir, self.config_suffix, generation)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            return True

    def clear(self) -> bool:
        # Clear is quiescent-only: the gate prevents a concurrent write from
        # interleaving between the durable CLEAR and allocator reset.
        with self._write_gate:
            self._abort_all_groups_locked()
            with self._lock:
                self._append_wal_frame(FRAME_CLEAR, encode_clear_payload(self._wal_next_seq))
                self._sync_wal()
                self._allocator.clear(
                    trace_context=getattr(self, "_t38c_durable_context", None)
                )
                self._components.clear()
            self.checkpoint()
        logger.info("Cleared all entries in ExtentFileBackend storage.")
        return True

    def get_stats(self):
        # Engine contract (unified_radix_cache.check_hicache_events): must be
        # None or StorageMetrics; a dict crashes the scheduler with
        # AttributeError on the prefetch_stats assignment (3090 repro).  The
        # raw io counters live under stats_snapshot().
        return None

    def stats_snapshot(self) -> dict:
        """Raw io counters (opens/preads/pwrites/bytes_*) for tests/tooling."""
        with self._lock:
            return dict(self._stats)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            try:
                with self._lifecycle_lock:
                    if self._lifecycle_state == "OPEN":
                        self._lifecycle_state = "CLOSING"
                self._checkpoint_stop.set()
                self._checkpoint_event.set()
                checkpoint_thread = self._checkpoint_thread
                if checkpoint_thread is not None:
                    checkpoint_thread.join()
                    self._checkpoint_thread = None
                # H3 teardown barrier: drain + join the native worker before the
                # fds it writes through close, and before any host-pool
                # destroy()/cudaHostUnregister can run after close() returns.
                if self._native_writer is not None:
                    if not self._native_shutdown():
                        raise RuntimeError(
                            "ExtentFileBackend.close: native worker still holds "
                            "page leases; refusing to tear down pinned memory"
                        )
                    self._native_writer = None
                if self._constructor_complete and os.path.isdir(self._storage_dir):
                    with self._write_gate:
                        self._abort_all_groups_locked()
                    self.checkpoint()
                if self._wal_fd is not None:
                    os.close(self._wal_fd)
                    self._wal_fd = None
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
            finally:
                self._closed = True
                self._release_backend_meta_lock()

    # ---- raw fd IO (explicit offsets, thread-safe on a shared fd) ----

    def _pread_full(self, buf: bytearray, offset: int) -> int:
        mv = memoryview(buf)
        total = 0
        try:
            while total < len(mv):
                got = os.preadv(self._fd, [mv[total:]], offset + total)
                if got == 0:
                    break
                total += got
        except OSError as e:
            logger.error(f"ExtentFileBackend: pread failed at {offset}: {e}")
            return 0
        self._stats["preads"] += 1
        self._stats["bytes_read"] += total
        return total

    def _preadv_full(self, offset: int, buffers: List[Any]) -> int:
        """preadv ``buffers`` sequentially at ``offset`` until full or EOF.

        Buffers mix ctypes arrays aliasing host-pool memory (v1 zero-copy
        targets) and plain bytearrays (merged-range gap scratch); a partial
        read resumes past the already-consumed bytes.  Each syscall is
        capped at IOV_MAX iovecs (defensive: a merged range holds 2 blocks
        per page, so batches beyond 512 pages per range would otherwise
        fail with EINVAL; today's batches stay well under).
        """
        total = sum(len(b) for b in buffers)
        done = 0
        try:
            while done < total:
                iov = []
                skip = done
                for b in buffers:
                    if skip >= len(b):
                        skip -= len(b)
                        continue
                    if len(iov) >= _IOV_MAX:
                        break
                    iov.append(memoryview(b)[skip:])
                    skip = 0
                got = os.preadv(self._fd, iov, offset + done)
                if got == 0:
                    break
                done += got
        except OSError as e:
            logger.error(f"ExtentFileBackend: pread failed at {offset}: {e}")
            return 0
        self._stats["preads"] += 1
        self._stats["bytes_read"] += done
        return done

    def _pwritev(self, offset: int, buffers: List[memoryview]) -> bool:
        """pwritev ``buffers`` at ``offset``, sharded past the IOV_MAX cap.

        One syscall takes at most IOV_MAX(1024) iovecs (measured: 1024 OK,
        1025 EINVAL); the v1 path packs 2 blocks per page, so a 1024-page
        batch is 2048 iovecs and must split into consecutive shards whose
        offsets advance by the bytes already written.  Each shard is its
        own syscall and failure point: a short or failed shard returns
        False and the caller refuses the whole batch (all-or-nothing
        semantics unchanged).
        """
        total = sum(len(b) for b in buffers)
        written = 0
        try:
            for start in range(0, len(buffers), _IOV_MAX):
                chunk = buffers[start : start + _IOV_MAX]
                got = os.pwritev(self._fd, chunk, offset + written)
                self._stats["pwrites"] += 1
                if got != sum(len(b) for b in chunk):
                    logger.error(
                        f"ExtentFileBackend: short pwrite at {offset + written}"
                    )
                    return False
                written += got
        except OSError as e:
            logger.error(
                f"ExtentFileBackend: pwrite failed at {offset + written}: {e}"
            )
            self._stats["pwrites"] += 1
            return False
        self._stats["bytes_written"] += written
        return written == total
