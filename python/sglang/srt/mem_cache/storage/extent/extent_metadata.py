# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Durable metadata serialization primitives for the extent-file HiCache L3
store (G1 crash-consistency contract).

This module is pure and filesystem-free.  It implements only the byte-level
(de)serialization of:

* the checkpoint manifest (``hicache.meta{config_suffix}.manifest``), and
* metadata WAL segment headers and frames
  (``hicache.meta{config_suffix}.wal.<gen>``).

It deliberately contains **no** recovery algorithm, no file IO, no fencing and
no allocator state machine.  Recovery (a later increment) is expected to build
on these primitives and to call :func:`validate_geometry` explicitly, so the
runtime-geometry checks are kept OUT of ``deserialize_manifest``.

All integers are little-endian.  Every deserializer is strict: bad magic,
unsupported format version, CRC mismatch, short reads, oversized frames and
trailing bytes all raise :class:`MetadataError`.

Checksum choice
---------------
The contract calls for ``crc32c`` (Castagnoli).  The project has no crc32c
implementation or dependency (only ``zlib.crc32``, which is CRC-32/ISO-HDLC and
is *not* wire-compatible with CRC-32C), so a small table-driven reflected
Castagnoli implementation is provided here.  Known vector:
``crc32c(b"123456789") == 0xE3069283``.
"""

from __future__ import annotations

import struct
import errno
import fcntl
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from sglang.srt.mem_cache.hicache_storage import PoolName

__all__ = [
    "MetadataError",
    "RecoveryError",
    "crc32c",
    "MANIFEST_MAGIC",
    "SEGMENT_MAGIC",
    "FRAME_MAGIC",
    "FORMAT_VERSION",
    "MANIFEST_HEADER_SIZE",
    "SEGMENT_HEADER_SIZE",
    "FRAME_HEADER_SIZE",
    "FRAME_CRC_OFFSET",
    "DEFAULT_WAL_MAX_RECORD_BYTES",
    "COMPONENT_OTHER",
    "GROUP_STATE_COMPLETE",
    "FRAME_SET",
    "FRAME_BATCH_COMMIT",
    "FRAME_GROUP_COMMIT",
    "FRAME_DROP_SLOT",
    "FRAME_CLEAR",
    "FRAME_CHECKPOINT_MARK",
    "FRAME_TYPES",
    "component_from_pool_name",
    "pool_name_from_component",
    "ManifestEntry",
    "ManifestGroup",
    "ManifestState",
    "SegmentHeader",
    "SetEntry",
    "SetPayload",
    "BatchCommitPayload",
    "GroupCommitPayload",
    "DropSlotPayload",
    "serialize_manifest",
    "deserialize_manifest",
    "validate_geometry",
    "serialize_segment_header",
    "deserialize_segment_header",
    "encode_frame",
    "decode_frame",
    "encode_set_payload",
    "decode_set_payload",
    "encode_batch_commit_payload",
    "decode_batch_commit_payload",
    "encode_group_commit_payload",
    "decode_group_commit_payload",
    "encode_drop_slot_payload",
    "decode_drop_slot_payload",
    "encode_clear_payload",
    "decode_clear_payload",
    "encode_checkpoint_mark_payload",
    "decode_checkpoint_mark_payload",
    "WalRange",
    "WalFrame",
    "TornTail",
    "SegmentFrames",
    "ReplayResult",
    "manifest_path",
    "write_manifest_atomic",
    "read_manifest",
    "segment_path",
    "current_path",
    "lock_path",
    "create_segment",
    "publish_current",
    "read_current",
    "resolve_wal_range",
    "resolve_manifest_absent",
    "iter_segment_frames",
    "repair_active_tail",
    "replay_records",
    "checkpoint_state_from_allocator",
    "acquire_meta_lock",
    "promote_pending_rotation",
]


class MetadataError(Exception):
    """Raised on any malformed or inconsistent durable-metadata byte stream."""


class RecoveryError(MetadataError):
    """A fail-closed recovery error with durable-location context."""

    def __init__(
        self,
        message: str,
        *,
        path: Optional[Union[str, os.PathLike]] = None,
        lsn: Optional[int] = None,
        offset: Optional[int] = None,
        failure_class: str = "recovery-failure",
    ) -> None:
        self.path = os.fspath(path) if path is not None else None
        self.lsn = lsn
        self.offset = offset
        self.failure_class = failure_class
        details = [f"class={failure_class}"]
        if self.path is not None:
            details.append(f"path={self.path}")
        if lsn is not None:
            details.append(f"lsn={lsn}")
        if offset is not None:
            details.append(f"offset={offset}")
        super().__init__(f"{message} ({', '.join(details)})")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Manifest header magic, fixed by contract S3: 0x484D3145 (b"HM1E").
MANIFEST_MAGIC = 0x484D3145
#: WAL segment header magic.  Chosen distinct from the manifest/frame magics;
#: not pinned by the contract text (b"HM1S").
SEGMENT_MAGIC = 0x484D3153
#: WAL frame magic, distinct from the manifest/segment magics (b"HM1F").
FRAME_MAGIC = 0x484D3146
#: On-disk format version, fixed at 1 by the contract.
FORMAT_VERSION = 1

MANIFEST_HEADER_SIZE = 16
SEGMENT_HEADER_SIZE = 28
# magic(4) version(2) type(1) flags(1) reserved(2) payload_len(4) seq(8) crc(4)
FRAME_HEADER_SIZE = 26
#: CRC field starts here; the checksummed range is header[0..22) || payload.
FRAME_CRC_OFFSET = 22

#: Contract proposal: min(8 MiB, extent_size).  Callers may pass a tighter or
#: looser bound explicitly, but this is the default enforced by frame codecs.
DEFAULT_WAL_MAX_RECORD_BYTES = 8 * 1024 * 1024

UINT8_MAX = 0xFF
UINT16_MAX = 0xFFFF
UINT32_MAX = 0xFFFFFFFF
UINT64_MAX = 0xFFFFFFFFFFFFFFFF


# --------------------------------------------------------------------------- #
# CRC-32C (Castagnoli), reflected, polynomial 0x82F63B78
# --------------------------------------------------------------------------- #

_CRC32C_POLY = 0x82F63B78


def _build_crc32c_table() -> Tuple[int, ...]:
    table = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if (crc & 1) else 0)
        table.append(crc)
    return tuple(table)


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(data: bytes, crc: int = 0) -> int:
    """Return the CRC-32C (Castagnoli) checksum of ``data`` as a u32."""

    crc ^= UINT32_MAX
    for byte in data:
        crc = (crc >> 8) ^ _CRC32C_TABLE[(crc ^ byte) & 0xFF]
    return crc ^ UINT32_MAX


# --------------------------------------------------------------------------- #
# Component enum (contract S3.1)
# --------------------------------------------------------------------------- #

#: Reserved "unknown/future pool" component; persisted value 255.
COMPONENT_OTHER = 255
NUM_COMPONENTS = 13

# Persisted numbers are FIXED by the contract table, in this exact order.
_COMPONENT_POOLS: Tuple[PoolName, ...] = (
    PoolName.KV,
    PoolName.MAMBA,
    PoolName.SWA,
    PoolName.INDEXER,
    PoolName.DEEPSEEK_V4_C4,
    PoolName.DEEPSEEK_V4_C4_INDEXER,
    PoolName.DEEPSEEK_V4_C128,
    PoolName.DEEPSEEK_V4_C4_STATE,
    PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    PoolName.DEEPSEEK_V4_C128_STATE,
    PoolName.DRAFT,
    PoolName.DRAFT_INDEXER,
    PoolName.DRAFT_SWA,
)

COMPONENT_BY_POOL = {pool: code for code, pool in enumerate(_COMPONENT_POOLS)}
POOL_BY_COMPONENT = {code: pool for code, pool in enumerate(_COMPONENT_POOLS)}


def component_from_pool_name(name) -> int:
    """Map a :class:`PoolName` (or its string value) to its persisted u8 code.

    Any pool that is not one of the 13 contract pools maps to
    :data:`COMPONENT_OTHER` (255).  The mapping is lossy on purpose; the
    storage key still disambiguates the real pool at runtime.
    """

    if not isinstance(name, PoolName):
        try:
            name = PoolName(name)
        except ValueError:
            return COMPONENT_OTHER
    return COMPONENT_BY_POOL.get(name, COMPONENT_OTHER)


def pool_name_from_component(code: int) -> Optional[PoolName]:
    """Inverse of :func:`component_from_pool_name`.

    Returns ``None`` for :data:`COMPONENT_OTHER`, and raises
    :class:`MetadataError` for any other code outside the 13-value table.
    """

    if code == COMPONENT_OTHER:
        return None
    try:
        return POOL_BY_COMPONENT[code]
    except KeyError:
        raise MetadataError(f"unknown component code: {code!r}") from None


# --------------------------------------------------------------------------- #
# Bounds-checked reader / integer validation helpers
# --------------------------------------------------------------------------- #


def _check_uint(value: int, bits: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MetadataError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0 or value > (1 << bits) - 1:
        raise MetadataError(f"{name}={value} out of range for u{bits}")
    return value


def _check_fixed16(value: bytes, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise MetadataError(f"{name} must be bytes, got {type(value).__name__}")
    if len(value) != 16:
        raise MetadataError(f"{name} must be exactly 16 bytes, got {len(value)}")
    return bytes(value)


class _Reader:
    """Little-endian cursor with strict short-read detection."""

    __slots__ = ("_data", "_off")

    def __init__(self, data: bytes):
        self._data = data
        self._off = 0

    def _need(self, n: int) -> None:
        if n < 0 or self._off + n > len(self._data):
            raise MetadataError(
                f"short read: need {n} bytes at offset {self._off}, "
                f"have {len(self._data) - self._off}"
            )

    @property
    def remaining(self) -> int:
        return len(self._data) - self._off

    def u8(self) -> int:
        self._need(1)
        value = self._data[self._off]
        self._off += 1
        return value

    def u16(self) -> int:
        self._need(2)
        value = struct.unpack_from("<H", self._data, self._off)[0]
        self._off += 2
        return value

    def u32(self) -> int:
        self._need(4)
        value = struct.unpack_from("<I", self._data, self._off)[0]
        self._off += 4
        return value

    def u64(self) -> int:
        self._need(8)
        value = struct.unpack_from("<Q", self._data, self._off)[0]
        self._off += 8
        return value

    def raw(self, n: int) -> bytes:
        self._need(n)
        value = self._data[self._off : self._off + n]
        self._off += n
        return value

    def require_end(self) -> None:
        if self._off != len(self._data):
            raise MetadataError(
                f"trailing bytes: {len(self._data) - self._off} unconsumed"
            )


def _read_u64_list(reader: _Reader, count: int) -> List[int]:
    if count * 8 > reader.remaining:
        raise MetadataError(f"short read: {count} u64 values exceed payload")
    return [reader.u64() for _ in range(count)]


def _read_u32_list(reader: _Reader, count: int) -> List[int]:
    if count * 4 > reader.remaining:
        raise MetadataError(f"short read: {count} u32 values exceed payload")
    return [reader.u32() for _ in range(count)]


def _pack_u16_len(payload: bytearray, raw: bytes, name: str) -> None:
    if len(raw) > UINT16_MAX:
        raise MetadataError(f"{name} length {len(raw)} exceeds u16")
    payload += struct.pack("<H", len(raw))
    payload += raw


# --------------------------------------------------------------------------- #
# Manifest (contract S3)
# --------------------------------------------------------------------------- #


@dataclass
class ManifestEntry:
    """One live index entry (contract-minimum metadata)."""

    storage_key: bytes
    extent_id: int
    offset: int
    length: int
    generation: int
    component: int
    group_epoch: bytes = b"\x00" * 16
    group_seq: int = 0


#: A checkpoint may only contain complete groups (contract S3).
GROUP_STATE_COMPLETE = 1


@dataclass
class ManifestGroup:
    """One complete atomic group."""

    group_epoch: bytes
    group_seq: int
    member_count: int
    state: int = GROUP_STATE_COMPLETE


@dataclass
class ManifestState:
    """In-memory checkpoint state, in contract payload order."""

    checkpoint_lsn: int = 0
    wal_floor: int = 0
    wal_active_gen: int = 0
    extent_size: int = 0
    file_size: int = 0
    num_slots: int = 0
    data_dev: int = 0
    data_ino: int = 0
    file_uuid: bytes = b"\x00" * 16
    namespace_epoch: bytes = b"\x00" * 16
    next_slot: int = 0
    generations: List[int] = field(default_factory=list)
    cursors: List[int] = field(default_factory=list)
    slot_bytes: List[int] = field(default_factory=list)
    recency_order: List[int] = field(default_factory=list)
    entries: List[ManifestEntry] = field(default_factory=list)
    groups: List[ManifestGroup] = field(default_factory=list)
    flags: int = 0


def _encode_index_entry(
    storage_key: bytes,
    extent_id: int,
    offset: int,
    length: int,
    generation: int,
    component: int,
) -> bytes:
    if not isinstance(storage_key, (bytes, bytearray)):
        raise MetadataError("storage_key must be bytes")
    out = bytearray()
    _pack_u16_len(out, bytes(storage_key), "storage_key")
    out += struct.pack(
        "<IIIQB",
        _check_uint(extent_id, 32, "extent_id"),
        _check_uint(offset, 32, "offset"),
        _check_uint(length, 32, "length"),
        _check_uint(generation, 64, "generation"),
        _check_uint(component, 8, "component"),
    )
    return bytes(out)


def _decode_index_entry(reader: _Reader) -> SetEntry:
    key_len = reader.u16()
    key = reader.raw(key_len)
    extent_id = reader.u32()
    offset = reader.u32()
    length = reader.u32()
    generation = reader.u64()
    component = reader.u8()
    return SetEntry(key, extent_id, offset, length, generation, component)


def _encode_manifest_payload(state: ManifestState) -> bytes:
    num_slots = state.num_slots
    _check_uint(num_slots, 32, "num_slots")
    if not (
        len(state.generations)
        == len(state.cursors)
        == len(state.slot_bytes)
        == num_slots
    ):
        raise MetadataError(
            "num_slots does not match generations/cursors/slot_bytes lengths "
            f"({num_slots} vs "
            f"{len(state.generations)}/{len(state.cursors)}/{len(state.slot_bytes)})"
        )

    payload = bytearray()
    payload += struct.pack(
        "<QQQQQ",
        _check_uint(state.checkpoint_lsn, 64, "checkpoint_lsn"),
        _check_uint(state.wal_floor, 64, "wal_floor"),
        _check_uint(state.wal_active_gen, 64, "wal_active_gen"),
        _check_uint(state.extent_size, 64, "extent_size"),
        _check_uint(state.file_size, 64, "file_size"),
    )
    payload += struct.pack("<I", num_slots)
    payload += struct.pack(
        "<QQ",
        _check_uint(state.data_dev, 64, "data_dev"),
        _check_uint(state.data_ino, 64, "data_ino"),
    )
    payload += _check_fixed16(state.file_uuid, "file_uuid")
    payload += _check_fixed16(state.namespace_epoch, "namespace_epoch")
    payload += struct.pack("<I", _check_uint(state.next_slot, 32, "next_slot"))

    for generation in state.generations:
        payload += struct.pack("<Q", _check_uint(generation, 64, "generation"))
    for cursor in state.cursors:
        payload += struct.pack("<I", _check_uint(cursor, 32, "cursor"))
    for slot_bytes in state.slot_bytes:
        payload += struct.pack("<Q", _check_uint(slot_bytes, 64, "slot_bytes"))

    payload += struct.pack(
        "<I", _check_uint(len(state.recency_order), 32, "recency_count")
    )
    for key in state.recency_order:
        payload += struct.pack("<I", _check_uint(key, 32, "recency_key"))

    payload += struct.pack("<I", _check_uint(len(state.entries), 32, "entry_count"))
    for entry in state.entries:
        payload += _encode_index_entry(
            entry.storage_key,
            entry.extent_id,
            entry.offset,
            entry.length,
            entry.generation,
            entry.component,
        )
        payload += _check_fixed16(entry.group_epoch, "group_epoch")
        payload += struct.pack("<Q", _check_uint(entry.group_seq, 64, "group_seq"))

    payload += struct.pack("<I", _check_uint(len(state.groups), 32, "group_count"))
    for group in state.groups:
        payload += _check_fixed16(group.group_epoch, "group_epoch")
        payload += struct.pack("<Q", _check_uint(group.group_seq, 64, "group_seq"))
        payload += struct.pack(
            "<I", _check_uint(group.member_count, 32, "member_count")
        )
        payload += struct.pack("<B", _check_uint(group.state, 8, "group_state"))

    return bytes(payload)


def serialize_manifest(state: ManifestState) -> bytes:
    """Serialize a :class:`ManifestState` to its canonical byte form."""

    payload = _encode_manifest_payload(state)
    _check_uint(state.flags, 16, "flags")
    header = struct.pack(
        "<IHHI", MANIFEST_MAGIC, FORMAT_VERSION, state.flags, len(payload)
    )
    checksum = crc32c(header + payload)
    return header + struct.pack("<I", checksum) + payload


def deserialize_manifest(data: bytes) -> ManifestState:
    """Strictly parse a manifest; raises :class:`MetadataError` on any fault."""

    if len(data) < MANIFEST_HEADER_SIZE:
        raise MetadataError(
            f"short manifest header: {len(data)} < {MANIFEST_HEADER_SIZE}"
        )
    magic, version, flags, payload_len = struct.unpack_from("<IHHI", data, 0)
    if magic != MANIFEST_MAGIC:
        raise MetadataError(f"bad manifest magic: 0x{magic:08X}")
    if version != FORMAT_VERSION:
        raise MetadataError(f"unsupported manifest format_version: {version}")

    total = MANIFEST_HEADER_SIZE + payload_len
    if len(data) < total:
        raise MetadataError(
            f"truncated manifest: {len(data)} < {total} declared bytes"
        )
    if len(data) > total:
        raise MetadataError(
            f"trailing bytes: {len(data) - total} bytes after manifest payload"
        )

    stored_crc = struct.unpack_from("<I", data, 12)[0]
    payload = data[MANIFEST_HEADER_SIZE:]
    if crc32c(data[0:12] + payload) != stored_crc:
        raise MetadataError("manifest crc32c mismatch")

    reader = _Reader(payload)
    state = ManifestState()
    state.flags = flags
    state.checkpoint_lsn = reader.u64()
    state.wal_floor = reader.u64()
    state.wal_active_gen = reader.u64()
    state.extent_size = reader.u64()
    state.file_size = reader.u64()
    state.num_slots = reader.u32()
    state.data_dev = reader.u64()
    state.data_ino = reader.u64()
    state.file_uuid = reader.raw(16)
    state.namespace_epoch = reader.raw(16)
    state.next_slot = reader.u32()
    state.generations = _read_u64_list(reader, state.num_slots)
    state.cursors = _read_u32_list(reader, state.num_slots)
    state.slot_bytes = _read_u64_list(reader, state.num_slots)
    recency_count = reader.u32()
    state.recency_order = _read_u32_list(reader, recency_count)

    entry_count = reader.u32()
    # Each entry is at least a u16 key length + 4+4+4+8+1 bytes + 16 + 8.
    min_entry_bytes = 2 + 4 + 4 + 4 + 8 + 1 + 16 + 8
    if entry_count * min_entry_bytes > reader.remaining:
        raise MetadataError(
            f"short read: {entry_count} manifest entries exceed payload"
        )
    entries: List[ManifestEntry] = []
    for _ in range(entry_count):
        core = _decode_index_entry(reader)
        group_epoch = reader.raw(16)
        group_seq = reader.u64()
        entries.append(
            ManifestEntry(
                storage_key=core.storage_key,
                extent_id=core.extent_id,
                offset=core.offset,
                length=core.length,
                generation=core.generation,
                component=core.component,
                group_epoch=group_epoch,
                group_seq=group_seq,
            )
        )
    state.entries = entries

    group_count = reader.u32()
    if group_count * (16 + 8 + 4 + 1) > reader.remaining:
        raise MetadataError(
            f"short read: {group_count} manifest groups exceed payload"
        )
    groups: List[ManifestGroup] = []
    for _ in range(group_count):
        group_epoch = reader.raw(16)
        group_seq = reader.u64()
        member_count = reader.u32()
        state_byte = reader.u8()
        groups.append(ManifestGroup(group_epoch, group_seq, member_count, state_byte))
    state.groups = groups

    reader.require_end()
    return state


def validate_geometry(
    state: ManifestState,
    *,
    extent_size: int,
    file_size: int,
    num_slots: Optional[int] = None,
    data_dev: Optional[int] = None,
    data_ino: Optional[int] = None,
) -> None:
    """Fail closed if persisted geometry disagrees with the runtime.

    Kept separate from :func:`deserialize_manifest` on purpose: parsing must not
    depend on caller geometry, and recovery calls this explicitly.  ``data_dev``
    / ``data_ino`` of 0 mean "unknown" and are not treated as a mismatch.
    """

    if state.extent_size != extent_size:
        raise MetadataError(
            f"extent_size mismatch: manifest={state.extent_size} runtime={extent_size}"
        )
    if state.file_size != file_size:
        raise MetadataError(
            f"file_size mismatch: manifest={state.file_size} runtime={file_size}"
        )
    if num_slots is None:
        if extent_size <= 0:
            raise MetadataError("cannot derive num_slots from non-positive extent_size")
        num_slots = file_size // extent_size
    if state.num_slots != num_slots:
        raise MetadataError(
            f"num_slots mismatch: manifest={state.num_slots} runtime={num_slots}"
        )
    if data_dev is not None and state.data_dev != 0 and state.data_dev != data_dev:
        raise MetadataError(
            f"data_dev mismatch: manifest={state.data_dev} runtime={data_dev}"
        )
    if data_ino is not None and state.data_ino != 0 and state.data_ino != data_ino:
        raise MetadataError(
            f"data_ino mismatch: manifest={state.data_ino} runtime={data_ino}"
        )


# --------------------------------------------------------------------------- #
# WAL segment header (contract S4.1)
# --------------------------------------------------------------------------- #


@dataclass
class SegmentHeader:
    segment_generation: int = 0
    base_lsn: int = 0
    flags: int = 0


def serialize_segment_header(header: SegmentHeader) -> bytes:
    _check_uint(header.flags, 16, "flags")
    base = struct.pack(
        "<IHHQQ",
        SEGMENT_MAGIC,
        FORMAT_VERSION,
        header.flags,
        _check_uint(header.segment_generation, 64, "segment_generation"),
        _check_uint(header.base_lsn, 64, "base_lsn"),
    )
    return base + struct.pack("<I", crc32c(base))


def deserialize_segment_header(data: bytes) -> SegmentHeader:
    """Parse a segment header from the front of ``data``.

    Trailing bytes are allowed (frames follow the header in a segment); only
    the 28-byte header is consumed and validated.
    """

    if len(data) < SEGMENT_HEADER_SIZE:
        raise MetadataError(
            f"short segment header: {len(data)} < {SEGMENT_HEADER_SIZE}"
        )
    magic, version, flags, generation, base_lsn = struct.unpack_from("<IHHQQ", data, 0)
    if magic != SEGMENT_MAGIC:
        raise MetadataError(f"bad segment magic: 0x{magic:08X}")
    if version != FORMAT_VERSION:
        raise MetadataError(f"unsupported segment format_version: {version}")
    stored_crc = struct.unpack_from("<I", data, 24)[0]
    if crc32c(data[0:24]) != stored_crc:
        raise MetadataError("segment header crc32c mismatch")
    return SegmentHeader(generation, base_lsn, flags)


# --------------------------------------------------------------------------- #
# WAL frames (contract S4.2)
# --------------------------------------------------------------------------- #

FRAME_SET = 1
FRAME_BATCH_COMMIT = 2
FRAME_GROUP_COMMIT = 3
FRAME_DROP_SLOT = 4
FRAME_CLEAR = 5
FRAME_CHECKPOINT_MARK = 6
FRAME_TYPES = frozenset(
    {
        FRAME_SET,
        FRAME_BATCH_COMMIT,
        FRAME_GROUP_COMMIT,
        FRAME_DROP_SLOT,
        FRAME_CLEAR,
        FRAME_CHECKPOINT_MARK,
    }
)


def encode_frame(
    frame_type: int,
    seq: int,
    payload: bytes,
    *,
    wal_max_record_bytes: int = DEFAULT_WAL_MAX_RECORD_BYTES,
) -> bytes:
    """Encode one WAL frame.  ``CRC = crc32c(header[0..22) || payload)``."""

    if frame_type not in FRAME_TYPES:
        raise MetadataError(f"unknown frame type: {frame_type!r}")
    if not isinstance(payload, (bytes, bytearray)):
        raise MetadataError(f"payload must be bytes, got {type(payload).__name__}")
    payload = bytes(payload)
    if len(payload) > wal_max_record_bytes:
        raise MetadataError(
            f"payload_len {len(payload)} exceeds wal_max_record_bytes "
            f"{wal_max_record_bytes}"
        )
    _check_uint(seq, 64, "seq")
    header = struct.pack(
        "<IHBBHIQ", FRAME_MAGIC, FORMAT_VERSION, frame_type, 0, 0, len(payload), seq
    )
    checksum = crc32c(header + payload)
    return header + struct.pack("<I", checksum) + payload


def decode_frame(
    data: bytes,
    *,
    wal_max_record_bytes: int = DEFAULT_WAL_MAX_RECORD_BYTES,
) -> Tuple[int, int, bytes, int]:
    """Decode one frame from the front of ``data``.

    Returns ``(frame_type, seq, payload, consumed)``.  ``data`` may contain
    further frames after the decoded one; only ``consumed`` bytes are read.
    """

    if len(data) < FRAME_HEADER_SIZE:
        raise MetadataError(
            f"short frame header: {len(data)} < {FRAME_HEADER_SIZE}"
        )
    magic, version, frame_type, _flags, _reserved, payload_len, seq = (
        struct.unpack_from("<IHBBHIQ", data, 0)
    )
    if magic != FRAME_MAGIC:
        raise MetadataError(f"bad frame magic: 0x{magic:08X}")
    if version != FORMAT_VERSION:
        raise MetadataError(f"unsupported frame format_version: {version}")
    if frame_type not in FRAME_TYPES:
        raise MetadataError(f"unknown frame type: {frame_type!r}")
    if payload_len > wal_max_record_bytes:
        raise MetadataError(
            f"payload_len {payload_len} exceeds wal_max_record_bytes "
            f"{wal_max_record_bytes}"
        )
    total = FRAME_HEADER_SIZE + payload_len
    if len(data) < total:
        raise MetadataError(
            f"truncated frame payload: have {len(data) - FRAME_HEADER_SIZE}, "
            f"declared {payload_len}"
        )
    stored_crc = struct.unpack_from("<I", data, FRAME_CRC_OFFSET)[0]
    payload = data[FRAME_HEADER_SIZE:total]
    # CRC covers header[0..22) only, excluding the CRC field at bytes 22-25.
    if crc32c(data[0:FRAME_CRC_OFFSET] + payload) != stored_crc:
        raise MetadataError("frame crc32c mismatch")
    return frame_type, seq, payload, total


# --- payload codecs -------------------------------------------------------- #


@dataclass
class SetEntry:
    storage_key: bytes
    extent_id: int
    offset: int
    length: int
    generation: int
    component: int


@dataclass
class SetPayload:
    group_epoch: bytes
    group_seq: int
    batch_id: int
    member_index: int
    member_count: int
    entries: List[SetEntry] = field(default_factory=list)


@dataclass
class BatchCommitPayload:
    batch_id: int
    member_count: int


@dataclass
class GroupCommitPayload:
    group_epoch: bytes
    group_seq: int
    member_count: int


@dataclass
class DropSlotPayload:
    slot: int
    new_generation: int


def encode_set_payload(payload: SetPayload) -> bytes:
    out = bytearray()
    out += _check_fixed16(payload.group_epoch, "group_epoch")
    out += struct.pack(
        "<QQII",
        _check_uint(payload.group_seq, 64, "group_seq"),
        _check_uint(payload.batch_id, 64, "batch_id"),
        _check_uint(payload.member_index, 32, "member_index"),
        _check_uint(payload.member_count, 32, "member_count"),
    )
    for entry in payload.entries:
        out += _encode_index_entry(
            entry.storage_key,
            entry.extent_id,
            entry.offset,
            entry.length,
            entry.generation,
            entry.component,
        )
    return bytes(out)


def decode_set_payload(payload: bytes) -> SetPayload:
    reader = _Reader(payload)
    group_epoch = reader.raw(16)
    group_seq = reader.u64()
    batch_id = reader.u64()
    member_index = reader.u32()
    member_count = reader.u32()
    entries: List[SetEntry] = []
    # Entries have no explicit count in the contract layout; the frame's
    # payload_len delimits them, so parse until the payload is exhausted.
    min_entry_bytes = 2 + 4 + 4 + 4 + 8 + 1
    while reader.remaining >= min_entry_bytes:
        entries.append(_decode_index_entry(reader))
    reader.require_end()
    return SetPayload(
        group_epoch, group_seq, batch_id, member_index, member_count, entries
    )


def encode_batch_commit_payload(payload: BatchCommitPayload) -> bytes:
    return struct.pack(
        "<QI",
        _check_uint(payload.batch_id, 64, "batch_id"),
        _check_uint(payload.member_count, 32, "member_count"),
    )


def decode_batch_commit_payload(payload: bytes) -> BatchCommitPayload:
    reader = _Reader(payload)
    batch_id = reader.u64()
    member_count = reader.u32()
    reader.require_end()
    return BatchCommitPayload(batch_id, member_count)


def encode_group_commit_payload(payload: GroupCommitPayload) -> bytes:
    out = bytearray()
    out += _check_fixed16(payload.group_epoch, "group_epoch")
    out += struct.pack(
        "<QI",
        _check_uint(payload.group_seq, 64, "group_seq"),
        _check_uint(payload.member_count, 32, "member_count"),
    )
    return bytes(out)


def decode_group_commit_payload(payload: bytes) -> GroupCommitPayload:
    reader = _Reader(payload)
    group_epoch = reader.raw(16)
    group_seq = reader.u64()
    member_count = reader.u32()
    reader.require_end()
    return GroupCommitPayload(group_epoch, group_seq, member_count)


def encode_drop_slot_payload(payload: DropSlotPayload) -> bytes:
    return struct.pack(
        "<IQ",
        _check_uint(payload.slot, 32, "slot"),
        _check_uint(payload.new_generation, 64, "new_generation"),
    )


def decode_drop_slot_payload(payload: bytes) -> DropSlotPayload:
    reader = _Reader(payload)
    slot = reader.u32()
    new_generation = reader.u64()
    reader.require_end()
    return DropSlotPayload(slot, new_generation)


def encode_clear_payload(epoch: int) -> bytes:
    return struct.pack("<Q", _check_uint(epoch, 64, "epoch"))


def decode_clear_payload(payload: bytes) -> int:
    reader = _Reader(payload)
    epoch = reader.u64()
    reader.require_end()
    return epoch


def encode_checkpoint_mark_payload(checkpoint_lsn: int) -> bytes:
    return struct.pack("<Q", _check_uint(checkpoint_lsn, 64, "checkpoint_lsn"))


def decode_checkpoint_mark_payload(payload: bytes) -> int:
    reader = _Reader(payload)
    checkpoint_lsn = reader.u64()
    reader.require_end()
    return checkpoint_lsn


# --------------------------------------------------------------------------- #
# Persistence and restart primitives (D-I1c)
# --------------------------------------------------------------------------- #


def _metadata_prefix(storage_dir: Union[str, os.PathLike], config_suffix: str) -> str:
    return os.path.join(os.fspath(storage_dir), f"hicache.meta{config_suffix}")


def manifest_path(storage_dir: Union[str, os.PathLike], config_suffix: str = "") -> str:
    return _metadata_prefix(storage_dir, config_suffix) + ".manifest"


def segment_path(
    storage_dir: Union[str, os.PathLike], config_suffix: str, generation: int
) -> str:
    _check_uint(generation, 64, "generation")
    return _metadata_prefix(storage_dir, config_suffix) + f".wal.{generation}"


def current_path(storage_dir: Union[str, os.PathLike], config_suffix: str = "") -> str:
    return _metadata_prefix(storage_dir, config_suffix) + ".wal.current"


def lock_path(storage_dir: Union[str, os.PathLike], config_suffix: str = "") -> str:
    return _metadata_prefix(storage_dir, config_suffix) + ".lock"


def _fsync_parent(path: Union[str, os.PathLike]) -> None:
    parent = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
    fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_manifest_atomic(path: Union[str, os.PathLike], state: ManifestState) -> None:
    """Publish a complete manifest, never exposing a partially written file."""

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp.", dir=directory
    )
    try:
        data = serialize_manifest(state)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except Exception:
        if fd != -1:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_manifest(path: Union[str, os.PathLike]) -> ManifestState:
    path = os.fspath(path)
    try:
        with open(path, "rb") as stream:
            data = stream.read()
    except OSError as exc:
        raise RecoveryError(
            "unable to read manifest", path=path, failure_class="manifest-io"
        ) from exc
    try:
        return deserialize_manifest(data)
    except MetadataError as exc:
        raise RecoveryError(
            str(exc), path=path, failure_class="manifest-corruption"
        ) from exc


def create_segment(
    path: Union[str, os.PathLike], generation: int, base_lsn: int
) -> SegmentHeader:
    path = os.fspath(path)
    header = SegmentHeader(generation, base_lsn)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, serialize_segment_header(header))
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_parent(path)
    except OSError as exc:
        raise RecoveryError(
            "unable to create WAL segment", path=path, failure_class="segment-io"
        ) from exc
    return header


def publish_current(path: Union[str, os.PathLike], generation: int) -> None:
    path = os.fspath(path)
    _check_uint(generation, 64, "generation")
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp.", dir=directory, text=True
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(f"{generation}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_current(path: Union[str, os.PathLike]) -> int:
    path = os.fspath(path)
    try:
        with open(path, "r", encoding="ascii") as stream:
            text = stream.read()
    except OSError as exc:
        raise RecoveryError(
            "unable to read current WAL pointer", path=path, failure_class="current-io"
        ) from exc
    if not text.endswith("\n"):
        raise RecoveryError(
            "current WAL pointer is not newline terminated",
            path=path,
            failure_class="current-corruption",
        )
    value = text[:-1]
    if not value or not value.isascii() or not value.isdecimal():
        raise RecoveryError(
            "current WAL pointer is not a strict integer",
            path=path,
            failure_class="current-corruption",
        )
    try:
        generation = int(value, 10)
        _check_uint(generation, 64, "generation")
    except (MetadataError, ValueError) as exc:
        raise RecoveryError(
            "current WAL pointer is outside u64",
            path=path,
            failure_class="current-corruption",
        ) from exc
    return generation


@dataclass(frozen=True)
class WalRange:
    start_generation: int
    end_generation: int
    pending_rotation: bool = False

    @property
    def generations(self) -> range:
        return range(self.start_generation, self.end_generation + 1)


def resolve_wal_range(
    manifest_state: ManifestState, active_generation: int
) -> WalRange:
    _check_uint(active_generation, 64, "active_generation")
    floor = manifest_state.wal_floor
    if floor <= active_generation:
        return WalRange(floor, active_generation)
    if floor == active_generation + 1:
        return WalRange(floor, floor, True)
    raise RecoveryError(
        "WAL floor is beyond the single pending-rotation step",
        lsn=manifest_state.checkpoint_lsn,
        failure_class="wal-range-gap",
    )


def resolve_manifest_absent(
    active_generation: Optional[int], *, prior_metadata_exists: bool
) -> WalRange:
    if active_generation is None:
        if prior_metadata_exists:
            raise RecoveryError(
                "current WAL pointer is absent with prior metadata",
                failure_class="manifest-absent",
            )
        raise RecoveryError(
            "metadata has no current WAL pointer", failure_class="manifest-absent"
        )
    if prior_metadata_exists:
        raise RecoveryError(
            "manifest is absent while prior metadata/WAL exists",
            failure_class="manifest-absent",
        )
    return WalRange(active_generation, active_generation)


@dataclass(frozen=True)
class WalFrame:
    frame_type: int
    seq: int
    payload: bytes
    offset: int
    consumed: int


@dataclass(frozen=True)
class TornTail:
    last_valid_offset: int
    offset: int
    reason: str


class SegmentFrames(list):
    """Materialized strict frame stream with an optional tolerated tail result."""

    def __init__(
        self, frames: Iterable[WalFrame] = (), torn_tail: Optional[TornTail] = None
    ):
        super().__init__(frames)
        self.torn_tail = torn_tail


def _recovery_frame_error(path, offset, seq, message, failure_class="wal-corruption"):
    return RecoveryError(
        message, path=path, offset=offset, lsn=seq, failure_class=failure_class
    )


def iter_segment_frames(
    fd_or_path: Union[int, str, os.PathLike],
    start_offset: int = SEGMENT_HEADER_SIZE,
    active: bool = False,
    *,
    wal_max_record_bytes: int = DEFAULT_WAL_MAX_RECORD_BYTES,
) -> SegmentFrames:
    """Read a segment without changing an fd's shared seek position."""

    owned = not isinstance(fd_or_path, int)
    path = os.fspath(fd_or_path) if owned else None
    fd = os.open(path, os.O_RDONLY) if owned else fd_or_path
    try:
        try:
            size = os.fstat(fd).st_size
            data = os.pread(fd, size, 0)
        except OSError as exc:
            raise RecoveryError(
                "unable to read WAL segment", path=path, failure_class="segment-io"
            ) from exc
    finally:
        if owned:
            os.close(fd)
    if start_offset < SEGMENT_HEADER_SIZE or start_offset > len(data):
        raise _recovery_frame_error(
            path, start_offset, None, "invalid WAL start offset"
        )
    try:
        header = deserialize_segment_header(data[:SEGMENT_HEADER_SIZE])
    except MetadataError as exc:
        raise _recovery_frame_error(path, 0, None, str(exc)) from exc
    del header
    frames = []
    offset = start_offset
    expected_seq = None
    while offset < len(data):
        remaining = data[offset:]
        if len(remaining) < FRAME_HEADER_SIZE:
            if active:
                return SegmentFrames(
                    frames, TornTail(offset, offset, "truncated frame header")
                )
            raise _recovery_frame_error(
                path, offset, expected_seq, "truncated frame header"
            )
        payload_len = struct.unpack_from("<I", remaining, 10)[0]
        total = FRAME_HEADER_SIZE + payload_len
        if len(remaining) < total:
            if active:
                return SegmentFrames(
                    frames, TornTail(offset, offset, "truncated frame")
                )
            raise _recovery_frame_error(path, offset, expected_seq, "truncated frame")
        try:
            frame_type, seq, payload, consumed = decode_frame(
                remaining, wal_max_record_bytes=wal_max_record_bytes
            )
        except MetadataError as exc:
            # A bad checksum in a complete final frame is a torn active tail;
            # malformed framing (magic/version/type/length) is corruption.
            if active and "crc32c mismatch" in str(exc) and offset + total == len(data):
                return SegmentFrames(
                    frames, TornTail(offset, offset, "bad final frame crc")
                )
            raise _recovery_frame_error(path, offset, expected_seq, str(exc)) from exc
        if expected_seq is not None and seq != expected_seq:
            raise _recovery_frame_error(
                path, offset, seq,
                f"WAL sequence discontinuity: expected {expected_seq}, got {seq}",
                "wal-sequence",
            )
        frames.append(WalFrame(frame_type, seq, payload, offset, consumed))
        expected_seq = seq + 1
        offset += consumed
    return SegmentFrames(frames)


def repair_active_tail(path: Union[str, os.PathLike], last_valid_offset: int) -> None:
    path = os.fspath(path)
    try:
        fd = os.open(path, os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            if last_valid_offset < SEGMENT_HEADER_SIZE or last_valid_offset > size:
                raise RecoveryError(
                    "invalid active-tail repair offset",
                    path=path,
                    offset=last_valid_offset,
                    failure_class="wal-tail-repair",
                )
            os.ftruncate(fd, last_valid_offset)
            os.fdatasync(fd)
        finally:
            os.close(fd)
    except RecoveryError:
        raise
    except OSError as exc:
        raise RecoveryError(
            "unable to repair active WAL tail", path=path, offset=last_valid_offset,
            failure_class="wal-tail-repair",
        ) from exc


@dataclass
class ReplayResult:
    entries: List[ManifestEntry]
    generations: List[int]
    cursors: List[int]
    slot_bytes: List[int]
    complete_batches: Tuple[int, ...]
    complete_groups: Tuple[Tuple[bytes, int], ...]
    expected_fold_count: int
    seen_max_seq: int
    clear_epochs: Tuple[int, ...] = ()
    dropped_tail: Optional[TornTail] = None


def _state_value(state, name, default):
    if state is None:
        return default
    if isinstance(state, ManifestState):
        return getattr(state, name)
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _frame_parts(record):
    if isinstance(record, WalFrame):
        return record.frame_type, record.seq, record.payload
    if isinstance(record, tuple) and len(record) >= 3:
        return record[0], record[1], record[2]
    raise RecoveryError(
        "replay record is not a WAL frame", failure_class="wal-replay-input"
    )


def replay_records(
    records: Iterable[Union[WalFrame, Tuple[int, int, bytes]]],
    checkpoint_state: Optional[Union[ManifestState, Dict]] = None,
    *,
    num_slots: Optional[int] = None,
) -> ReplayResult:
    """Replay a validated frame stream with all-or-nothing batch/group rules.

    The function deliberately returns allocator-shaped plain state and imports no
    allocator/backend code.  ``checkpoint_state`` may be a ``ManifestState`` or
    a dict containing its corresponding arrays.
    """

    generations = list(_state_value(checkpoint_state, "generations", []))
    if num_slots is None:
        num_slots = _state_value(checkpoint_state, "num_slots", len(generations))
    if not generations:
        generations = [0] * num_slots
    if len(generations) != num_slots:
        raise RecoveryError(
            "checkpoint generations length mismatch", failure_class="wal-replay-state"
        )
    cursors = list(_state_value(checkpoint_state, "cursors", [0] * num_slots))
    slot_bytes = list(_state_value(checkpoint_state, "slot_bytes", [0] * num_slots))
    if len(cursors) != num_slots or len(slot_bytes) != num_slots:
        raise RecoveryError(
            "checkpoint accounting length mismatch", failure_class="wal-replay-state"
        )

    installed: Dict[bytes, ManifestEntry] = {}
    for entry in _state_value(checkpoint_state, "entries", []):
        installed[entry.storage_key] = entry
    staged: Dict[int, List[Tuple[SetPayload, SetEntry]]] = {}
    batch_complete = set()
    # A commit closes a batch.  SET frames arriving after that point must not
    # make the batch complete retroactively during the final fold.
    batch_commit_seen = set()
    batch_invalid = set()
    group_complete = set()
    # ``None`` is a permanently rejected group; a list is the exact member
    # snapshot present when GROUP_COMMIT was replayed.  Keeping that snapshot
    # is the temporal barrier: later frames cannot repair an early marker.
    group_states: Dict[
        Tuple[bytes, int], Optional[List[Tuple[SetPayload, SetEntry]]]
    ] = {}
    clear_epochs = []
    expected_seq = _state_value(checkpoint_state, "checkpoint_lsn", 0) + 1
    max_seq = expected_seq - 1

    def remove_slot(slot: int) -> None:
        for key, entry in list(installed.items()):
            if entry.extent_id == slot:
                del installed[key]
        for batch, values in list(staged.items()):
            kept = [
                (payload, entry)
                for payload, entry in values
                if entry.extent_id != slot
            ]
            if kept:
                staged[batch] = kept
            else:
                del staged[batch]
        slot_bytes[slot] = 0
        cursors[slot] = 0

    def clear_index() -> None:
        installed.clear()
        staged.clear()
        for index in range(num_slots):
            slot_bytes[index] = 0
            cursors[index] = 0

    for record in records:
        if isinstance(record, SegmentFrames):
            if record.torn_tail is not None:
                # The caller must repair before appending; replay still returns
                # the valid prefix and retains explicit tail evidence.
                torn_tail = record.torn_tail
            else:
                torn_tail = None
            iterable = record
        else:
            iterable = (record,)
            torn_tail = None
        for item in iterable:
            frame_type, seq, payload = _frame_parts(item)
            if expected_seq is not None and seq != expected_seq:
                raise RecoveryError(
                    f"WAL sequence discontinuity: expected {expected_seq}, got {seq}",
                    lsn=seq,
                    failure_class="wal-sequence",
                )
            expected_seq = seq + 1
            max_seq = max(max_seq, seq)
            try:
                if frame_type == FRAME_SET:
                    decoded = decode_set_payload(payload)
                    if decoded.batch_id in batch_commit_seen:
                        batch_complete.discard(decoded.batch_id)
                        batch_invalid.add(decoded.batch_id)
                    grouped = (decoded.group_epoch, decoded.group_seq)
                    if grouped != (b"\x00" * 16, 0) and grouped in group_states:
                        # A member after GROUP_COMMIT is reordered data, not a
                        # late contribution that can extend the group.
                        group_states[grouped] = None
                        group_complete.discard(grouped)
                    values = staged.setdefault(decoded.batch_id, [])
                    if decoded.member_index >= decoded.member_count:
                        raise MetadataError("SET member_index outside member_count")
                    for entry in decoded.entries:
                        if entry.extent_id >= num_slots:
                            raise MetadataError("SET slot outside checkpoint geometry")
                        if entry.generation < generations[entry.extent_id]:
                            raise MetadataError(
                                "SET generation is older than restored generation"
                            )
                        values.append((decoded, entry))
                elif frame_type == FRAME_BATCH_COMMIT:
                    decoded = decode_batch_commit_payload(payload)
                    batch_commit_seen.add(decoded.batch_id)
                    values = staged.get(decoded.batch_id, [])
                    grouped = {
                        (set_payload.group_epoch, set_payload.group_seq)
                        for set_payload, _ in values
                        if (set_payload.group_epoch, set_payload.group_seq)
                        != (b"\x00" * 16, 0)
                    }
                    if any(group in group_states for group in grouped):
                        # BATCH_COMMIT after GROUP_COMMIT is also a reordered
                        # member and invalidates that group permanently.
                        for group in grouped:
                            group_states[group] = None
                            group_complete.discard(group)
                    if decoded.batch_id not in batch_invalid:
                        member_indexes = {
                            set_payload.member_index for set_payload, _ in values
                        }
                        declared_counts = {
                            set_payload.member_count
                            for set_payload, _ in values
                        }
                        if (
                            decoded.member_count > 0
                            and len(values) > 0
                            and declared_counts == {decoded.member_count}
                            and member_indexes == set(range(decoded.member_count))
                        ):
                            batch_complete.add(decoded.batch_id)
                elif frame_type == FRAME_GROUP_COMMIT:
                    decoded = decode_group_commit_payload(payload)
                    group = (decoded.group_epoch, decoded.group_seq)
                    if group in group_states:
                        # Preserve the existing idempotent duplicate behavior
                        # for an identical valid marker.  Any differing or
                        # otherwise invalid duplicate fails closed.
                        previous = group_states[group]
                        if previous is None:
                            continue
                        current = [
                            (set_payload, entry)
                            for batch_id, values in staged.items()
                            if batch_id in batch_complete
                            for set_payload, entry in values
                            if (set_payload.group_epoch, set_payload.group_seq) == group
                        ]
                        if len(current) != decoded.member_count or current != previous:
                            group_states[group] = None
                            group_complete.discard(group)
                        continue

                    group_batches = {
                        batch_id
                        for batch_id, values in staged.items()
                        if any(
                            (set_payload.group_epoch, set_payload.group_seq) == group
                            for set_payload, _ in values
                        )
                    }
                    group_values = [
                        (set_payload, entry)
                        for batch_id, values in staged.items()
                        if batch_id in batch_complete
                        for set_payload, entry in values
                        if (set_payload.group_epoch, set_payload.group_seq) == group
                    ]
                    has_uncommitted_member = any(
                        (set_payload.group_epoch, set_payload.group_seq) == group
                        and batch_id not in batch_complete
                        for batch_id, values in staged.items()
                        for set_payload, _ in values
                    )
                    if (
                        not group_values
                        or not group_batches
                        or has_uncommitted_member
                        or len(group_values) != decoded.member_count
                    ):
                        # The group is rejected at the marker's position.  It
                        # must never become recoverable from later cardinality.
                        group_states[group] = None
                    else:
                        group_states[group] = group_values
                        group_complete.add(group)
                elif frame_type == FRAME_DROP_SLOT:
                    decoded = decode_drop_slot_payload(payload)
                    if (
                        decoded.slot >= num_slots
                        or decoded.new_generation <= generations[decoded.slot]
                    ):
                        raise RecoveryError(
                            "DROP_SLOT generation is not strictly increasing",
                            lsn=seq,
                            failure_class="generation-regression",
                        )
                    generations[decoded.slot] = decoded.new_generation
                    remove_slot(decoded.slot)
                    for group, values in list(group_states.items()):
                        if values is not None and any(
                            entry.extent_id == decoded.slot for _, entry in values
                        ):
                            group_states[group] = None
                            group_complete.discard(group)
                elif frame_type == FRAME_CLEAR:
                    clear_epochs.append(decode_clear_payload(payload))
                    clear_index()
                    for group in list(group_states):
                        group_states[group] = None
                    group_complete.clear()
                elif frame_type == FRAME_CHECKPOINT_MARK:
                    decode_checkpoint_mark_payload(payload)
                else:
                    raise MetadataError(f"unknown frame type: {frame_type}")
            except RecoveryError:
                raise
            except MetadataError as exc:
                raise RecoveryError(
                    str(exc), lsn=seq, failure_class="wal-replay-corruption"
                ) from exc

    complete_entries: List[ManifestEntry] = []
    for batch_id, values in staged.items():
        if batch_id not in batch_complete:
            continue
        for payload, entry in values:
            grouped = (payload.group_epoch, payload.group_seq)
            if grouped != (b"\x00" * 16, 0):
                # Grouped entries are folded from the commit-time snapshot
                # below, never from a later final-state cardinality match.
                continue
            complete_entries.append(
                ManifestEntry(
                    storage_key=entry.storage_key,
                    extent_id=entry.extent_id,
                    offset=entry.offset,
                    length=entry.length,
                    generation=entry.generation,
                    component=entry.component,
                    group_epoch=payload.group_epoch,
                    group_seq=payload.group_seq,
                )
            )
    for group, values in group_states.items():
        if values is None:
            continue
        for payload, entry in values:
            complete_entries.append(
                ManifestEntry(
                    storage_key=entry.storage_key,
                    extent_id=entry.extent_id,
                    offset=entry.offset,
                    length=entry.length,
                    generation=entry.generation,
                    component=entry.component,
                    group_epoch=payload.group_epoch,
                    group_seq=payload.group_seq,
                )
            )
    # A later SET for a key is the deterministic WAL-order winner.
    for entry in complete_entries:
        installed[entry.storage_key] = entry
    for index in range(num_slots):
        slot_bytes[index] = 0
        cursors[index] = 0
    for entry in installed.values():
        slot_bytes[entry.extent_id] += entry.length
        cursors[entry.extent_id] = max(
            cursors[entry.extent_id], entry.offset + entry.length
        )
    return ReplayResult(
        entries=list(installed.values()),
        generations=generations,
        cursors=cursors,
        slot_bytes=slot_bytes,
        complete_batches=tuple(sorted(batch_complete)),
        complete_groups=tuple(sorted(group_complete)),
        expected_fold_count=len(complete_entries),
        seen_max_seq=max_seq,
        clear_epochs=tuple(clear_epochs),
        dropped_tail=locals().get("torn_tail"),
    )


def checkpoint_state_from_allocator(
    *,
    checkpoint_lsn: int,
    wal_floor: int,
    wal_active_gen: int,
    extent_size: int,
    file_size: int,
    num_slots: int,
    generations: Iterable[int],
    cursors: Iterable[int],
    slot_bytes: Iterable[int],
    entries: Iterable[ManifestEntry] = (),
    groups: Iterable[ManifestGroup] = (),
    **kwargs,
) -> ManifestState:
    """Pure adapter for allocator snapshots; no allocator import is needed."""

    return ManifestState(
        checkpoint_lsn=checkpoint_lsn,
        wal_floor=wal_floor,
        wal_active_gen=wal_active_gen,
        extent_size=extent_size,
        file_size=file_size,
        num_slots=num_slots,
        generations=list(generations),
        cursors=list(cursors),
        slot_bytes=list(slot_bytes),
        entries=list(entries),
        groups=list(groups),
        **kwargs,
    )


def acquire_meta_lock(path: Union[str, os.PathLike], *, nonblocking: bool = False):
    """Acquire the persistent per-namespace metadata lock."""

    path = os.fspath(path)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(fd, flags)
        except OSError as exc:
            os.close(fd)
            if nonblocking and exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RecoveryError(
                    "metadata lock is already held",
                    path=path,
                    failure_class="lock-busy",
                ) from exc
            raise
        return fd
    except RecoveryError:
        raise
    except OSError as exc:
        raise RecoveryError(
            "unable to acquire metadata lock", path=path, failure_class="lock-io"
        ) from exc


def promote_pending_rotation(
    storage_dir: Union[str, os.PathLike],
    config_suffix: str,
    active_generation: int,
    checkpoint_lsn: int,
) -> int:
    """Repair/publish the one allowed manifest-published pending rotation."""

    generation = active_generation + 1
    path = segment_path(storage_dir, config_suffix, generation)
    current = current_path(storage_dir, config_suffix)
    try:
        if os.path.exists(current):
            named = read_current(current)
            if named not in (active_generation, generation):
                raise RecoveryError(
                    "current WAL pointer disagrees with pending rotation",
                    path=current,
                    failure_class="current-corruption",
                )
            current_segment = segment_path(storage_dir, config_suffix, named)
            try:
                with open(current_segment, "rb") as stream:
                    deserialize_segment_header(stream.read(SEGMENT_HEADER_SIZE))
            except (OSError, MetadataError) as exc:
                raise RecoveryError(
                    "current WAL segment has an invalid header",
                    path=current_segment,
                    failure_class="segment-corruption",
                ) from exc
        exists = os.path.exists(path)
        if exists:
            with open(path, "r+b") as stream:
                raw = stream.read(SEGMENT_HEADER_SIZE)
            try:
                header = deserialize_segment_header(raw)
                if header.segment_generation != generation:
                    raise MetadataError("segment generation does not match its name")
            except MetadataError as exc:
                # The promoted generation was never active.  Recreate only a
                # short/partial header; a full bad header is committed damage.
                if len(raw) >= SEGMENT_HEADER_SIZE:
                    raise RecoveryError(
                        str(exc), path=path, failure_class="segment-corruption"
                    ) from exc
                fd = os.open(path, os.O_RDWR)
                try:
                    os.ftruncate(fd, 0)
                    os.write(
                        fd,
                        serialize_segment_header(
                            SegmentHeader(generation, checkpoint_lsn)
                        ),
                    )
                    os.fdatasync(fd)
                finally:
                    os.close(fd)
        else:
            create_segment(path, generation, checkpoint_lsn)
    except RecoveryError:
        raise
    except OSError as exc:
        raise RecoveryError(
            "unable to promote WAL rotation", path=path, failure_class="segment-io"
        ) from exc
    publish_current(current, generation)
    return generation
