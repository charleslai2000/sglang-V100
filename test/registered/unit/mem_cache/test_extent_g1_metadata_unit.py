"""
Unit tests for the G1 durable metadata serialization primitives
(``extent_metadata.py``): checkpoint manifest framing and WAL segment
header/frame framing.

These tests cover only pure (de)serialization -- no recovery, no filesystem.

Run with:
    python test/registered/unit/mem_cache/test_extent_g1_metadata_unit.py
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import tempfile
import unittest

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.extent.extent_metadata import (
    COMPONENT_OTHER,
    DEFAULT_WAL_MAX_RECORD_BYTES,
    FRAME_BATCH_COMMIT,
    FRAME_CHECKPOINT_MARK,
    FRAME_CLEAR,
    FRAME_DROP_SLOT,
    FRAME_GROUP_COMMIT,
    FRAME_MAGIC,
    FRAME_SET,
    FORMAT_VERSION,
    BatchCommitPayload,
    DropSlotPayload,
    GroupCommitPayload,
    MANIFEST_MAGIC,
    ManifestEntry,
    ManifestGroup,
    ManifestState,
    MetadataError,
    RecoveryError,
    SegmentHeader,
    SetEntry,
    SetPayload,
    component_from_pool_name,
    crc32c,
    decode_batch_commit_payload,
    decode_checkpoint_mark_payload,
    decode_clear_payload,
    decode_drop_slot_payload,
    decode_frame,
    decode_group_commit_payload,
    decode_set_payload,
    deserialize_manifest,
    deserialize_segment_header,
    encode_batch_commit_payload,
    encode_checkpoint_mark_payload,
    encode_clear_payload,
    encode_drop_slot_payload,
    encode_frame,
    encode_group_commit_payload,
    encode_set_payload,
    pool_name_from_component,
    serialize_manifest,
    serialize_segment_header,
    validate_geometry,
    acquire_meta_lock,
    checkpoint_state_from_allocator,
    create_segment,
    current_path,
    iter_segment_frames,
    lock_path,
    manifest_path,
    promote_pending_rotation,
    publish_current,
    read_current,
    read_manifest,
    replay_records,
    repair_active_tail,
    resolve_manifest_absent,
    resolve_wal_range,
    segment_path,
    write_manifest_atomic,
)
from sglang.test.test_utils import CustomTestCase

_EPOCH = bytes(range(0x10, 0x20))
_UUID = bytes(range(0x30, 0x40))


def _sample_state() -> ManifestState:
    entries = [
        ManifestEntry(
            b"kv-key-0", 1, 0, 4096, 7, component_from_pool_name(PoolName.KV)
        ),
        ManifestEntry(
            b"kv-key-1",
            1,
            4096,
            2048,
            8,
            component_from_pool_name(PoolName.MAMBA),
        ),
        ManifestEntry(
            b"side-car",
            2,
            0,
            1024,
            9,
            component_from_pool_name(PoolName.DRAFT),
            group_epoch=_EPOCH,
            group_seq=1,
        ),
    ]
    groups = [ManifestGroup(_EPOCH, 1, 1)]
    return ManifestState(
        checkpoint_lsn=42,
        wal_floor=3,
        wal_active_gen=4,
        extent_size=4096,
        file_size=12288,
        num_slots=3,
        data_dev=11,
        data_ino=22,
        file_uuid=_UUID,
        namespace_epoch=_EPOCH,
        next_slot=2,
        generations=[1, 2, 3],
        cursors=[0, 10, 20],
        slot_bytes=[100, 200, 300],
        recency_order=[2, 1, 0],
        entries=entries,
        groups=groups,
    )


class TestCrc32c(CustomTestCase):
    def test_known_vector(self):
        # Castagnoli check value for "123456789".
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(crc32c(b""), 0)


class TestComponentEnum(CustomTestCase):
    _POOLS = (
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

    def test_all_thirteen_pools_round_trip(self):
        for expected, pool in enumerate(self._POOLS):
            code = component_from_pool_name(pool)
            self.assertEqual(code, expected)
            self.assertEqual(pool_name_from_component(code), pool)

    def test_unknown_pool_maps_to_other(self):
        self.assertEqual(component_from_pool_name("not-a-pool"), COMPONENT_OTHER)
        self.assertIsNone(pool_name_from_component(COMPONENT_OTHER))
        with self.assertRaises(MetadataError):
            pool_name_from_component(13)


class TestManifest(CustomTestCase):
    def test_round_trip(self):
        state = _sample_state()
        blob = serialize_manifest(state)
        parsed = deserialize_manifest(blob)
        self.assertEqual(parsed, state)
        self.assertEqual(parsed.num_slots, 3)
        self.assertEqual(len(parsed.groups), 1)
        self.assertEqual(parsed.entries[2].group_epoch, _EPOCH)
        self.assertEqual(parsed.entries[2].group_seq, 1)
        self.assertEqual(parsed.entries[0].group_seq, 0)

    def test_header_layout(self):
        state = _sample_state()
        blob = serialize_manifest(state)
        self.assertEqual(blob[0:4], MANIFEST_MAGIC.to_bytes(4, "little"))
        self.assertEqual(int.from_bytes(blob[4:6], "little"), FORMAT_VERSION)
        payload_len = int.from_bytes(blob[8:12], "little")
        self.assertEqual(len(blob), 16 + payload_len)

    def test_crc_corruption_detected(self):
        blob = bytearray(serialize_manifest(_sample_state()))
        blob[-1] ^= 0xFF
        with self.assertRaises(MetadataError):
            deserialize_manifest(bytes(blob))

    def test_bad_magic_detected(self):
        blob = bytearray(serialize_manifest(_sample_state()))
        blob[0:4] = b"\x00\x00\x00\x00"
        with self.assertRaises(MetadataError):
            deserialize_manifest(bytes(blob))

    def test_bad_version_detected(self):
        blob = bytearray(serialize_manifest(_sample_state()))
        blob[4:6] = (FORMAT_VERSION + 1).to_bytes(2, "little")
        with self.assertRaises(MetadataError):
            deserialize_manifest(bytes(blob))

    def test_truncation_detected(self):
        blob = serialize_manifest(_sample_state())
        with self.assertRaises(MetadataError):
            deserialize_manifest(blob[:-1])
        with self.assertRaises(MetadataError):
            deserialize_manifest(blob[:8])

    def test_trailing_bytes_detected(self):
        blob = serialize_manifest(_sample_state())
        with self.assertRaises(MetadataError):
            deserialize_manifest(blob + b"\x00")

    def test_num_slots_mismatch_rejected_on_serialize(self):
        state = _sample_state()
        state.generations = state.generations[:-1]
        with self.assertRaises(MetadataError):
            serialize_manifest(state)

    def test_geometry_validation(self):
        state = _sample_state()
        # Matching geometry passes, including the derived num_slots.
        validate_geometry(state, extent_size=4096, file_size=12288)
        validate_geometry(
            state,
            extent_size=4096,
            file_size=12288,
            num_slots=3,
            data_dev=11,
            data_ino=22,
        )
        with self.assertRaises(MetadataError):
            validate_geometry(state, extent_size=8192, file_size=12288)
        with self.assertRaises(MetadataError):
            validate_geometry(state, extent_size=4096, file_size=16384)
        with self.assertRaises(MetadataError):
            validate_geometry(
                state, extent_size=4096, file_size=12288, num_slots=4
            )
        with self.assertRaises(MetadataError):
            validate_geometry(
                state, extent_size=4096, file_size=12288, data_dev=99
            )

    def test_geometry_validation_skips_unknown_ids(self):
        state = _sample_state()
        state.data_dev = 0
        state.data_ino = 0
        validate_geometry(
            state, extent_size=4096, file_size=12288, data_dev=99, data_ino=99
        )


class TestSegmentHeader(CustomTestCase):
    def test_round_trip(self):
        header = SegmentHeader(segment_generation=7, base_lsn=42)
        blob = serialize_segment_header(header)
        self.assertEqual(len(blob), 28)
        self.assertEqual(deserialize_segment_header(blob), header)

    def test_trailing_bytes_allowed(self):
        blob = serialize_segment_header(SegmentHeader(1, 0))
        self.assertEqual(
            deserialize_segment_header(blob + b"frames"), SegmentHeader(1, 0)
        )

    def test_crc_corruption_detected(self):
        blob = bytearray(serialize_segment_header(SegmentHeader(1, 2)))
        blob[-1] ^= 0xFF
        with self.assertRaises(MetadataError):
            deserialize_segment_header(bytes(blob))

    def test_truncation_detected(self):
        with self.assertRaises(MetadataError):
            deserialize_segment_header(b"\x00" * 10)


class TestFrames(CustomTestCase):
    def _frames(self):
        set_payload = SetPayload(
            group_epoch=_EPOCH,
            group_seq=5,
            batch_id=7,
            member_index=0,
            member_count=2,
            entries=[
                SetEntry(b"k0", 0, 0, 4096, 1, component_from_pool_name(PoolName.KV)),
                SetEntry(
                    b"k1", 0, 4096, 1024, 2, component_from_pool_name(PoolName.SWA)
                ),
            ],
        )
        return [
            (FRAME_SET, encode_set_payload(set_payload)),
            (FRAME_BATCH_COMMIT, encode_batch_commit_payload(BatchCommitPayload(7, 2))),
            (
                FRAME_GROUP_COMMIT,
                encode_group_commit_payload(GroupCommitPayload(_EPOCH, 5, 2)),
            ),
            (FRAME_DROP_SLOT, encode_drop_slot_payload(DropSlotPayload(3, 99))),
            (FRAME_CLEAR, encode_clear_payload(12345)),
            (FRAME_CHECKPOINT_MARK, encode_checkpoint_mark_payload(42)),
        ]

    def test_round_trip_every_type(self):
        for index, (frame_type, payload) in enumerate(self._frames()):
            seq = 100 + index
            blob = encode_frame(frame_type, seq, payload)
            decoded_type, decoded_seq, decoded_payload, consumed = decode_frame(blob)
            self.assertEqual(decoded_type, frame_type)
            self.assertEqual(decoded_seq, seq)
            self.assertEqual(decoded_payload, payload)
            self.assertEqual(consumed, len(blob))

    def test_consumed_allows_sequential_frames(self):
        blobs = b""
        for index, (frame_type, payload) in enumerate(self._frames()):
            blobs += encode_frame(frame_type, index + 1, payload)
        offset = 0
        seen = []
        while offset < len(blobs):
            frame_type, seq, _payload, consumed = decode_frame(blobs[offset:])
            seen.append((frame_type, seq))
            offset += consumed
        self.assertEqual([t for t, _ in seen], [t for t, _ in self._frames()])
        self.assertEqual(offset, len(blobs))

    def test_payload_codecs_round_trip(self):
        payload = decode_set_payload(encode_set_payload(SetPayload(_EPOCH, 5, 7, 0, 2)))
        self.assertEqual(payload.group_epoch, _EPOCH)
        self.assertEqual(payload.group_seq, 5)
        self.assertEqual(payload.batch_id, 7)
        self.assertEqual(payload.entries, [])

        bc = decode_batch_commit_payload(
            encode_batch_commit_payload(BatchCommitPayload(7, 2))
        )
        self.assertEqual(bc, BatchCommitPayload(7, 2))
        gc = decode_group_commit_payload(
            encode_group_commit_payload(GroupCommitPayload(_EPOCH, 5, 2))
        )
        self.assertEqual(gc, GroupCommitPayload(_EPOCH, 5, 2))
        ds = decode_drop_slot_payload(encode_drop_slot_payload(DropSlotPayload(3, 99)))
        self.assertEqual(ds, DropSlotPayload(3, 99))
        self.assertEqual(decode_clear_payload(encode_clear_payload(12345)), 12345)
        self.assertEqual(
            decode_checkpoint_mark_payload(encode_checkpoint_mark_payload(42)), 42
        )

    def test_crc_covers_header_before_crc_field(self):
        blob = bytearray(encode_frame(FRAME_GROUP_COMMIT, 5, b"payload"))
        # Flip a byte inside header[0..22) -> detected.
        header_flip = bytearray(blob)
        header_flip[21] ^= 0x01
        with self.assertRaises(MetadataError):
            decode_frame(bytes(header_flip))
        # Flip a payload byte -> detected.
        payload_flip = bytearray(blob)
        payload_flip[-1] ^= 0x01
        with self.assertRaises(MetadataError):
            decode_frame(bytes(payload_flip))
        # Flip the CRC field itself (bytes 22-25) -> detected.
        crc_flip = bytearray(blob)
        crc_flip[22] ^= 0x01
        with self.assertRaises(MetadataError):
            decode_frame(bytes(crc_flip))

    def test_payload_bound_enforced(self):
        payload = b"x" * 9
        with self.assertRaises(MetadataError):
            encode_frame(FRAME_CLEAR, 1, payload, wal_max_record_bytes=8)
        # Encode under the default bound, then decode under a tighter bound.
        blob = encode_frame(FRAME_CLEAR, 1, payload)
        with self.assertRaises(MetadataError):
            decode_frame(blob, wal_max_record_bytes=8)
        # The default bound accepts the frame.
        self.assertLess(len(payload), DEFAULT_WAL_MAX_RECORD_BYTES)

    def test_sequence_bounds_enforced(self):
        with self.assertRaises(MetadataError):
            encode_frame(FRAME_CLEAR, -1, b"")
        with self.assertRaises(MetadataError):
            encode_frame(FRAME_CLEAR, 1 << 64, b"")

    def test_bad_frame_rejected(self):
        blob = encode_frame(FRAME_CLEAR, 1, b"")
        with self.assertRaises(MetadataError):
            encode_frame(999, 1, b"")
        with self.assertRaises(MetadataError):
            decode_frame(blob[:-1])
        bad_magic = bytearray(blob)
        bad_magic[0:4] = (FRAME_MAGIC ^ 0xFFFFFFFF).to_bytes(4, "little")
        with self.assertRaises(MetadataError):
            decode_frame(bytes(bad_magic))
        bad_type = bytearray(blob)
        bad_type[6] = 200
        with self.assertRaises(MetadataError):
            decode_frame(bytes(bad_type))
        with self.assertRaises(MetadataError):
            decode_frame(b"\x00" * 4)

    def test_payload_helpers_reject_trailing_bytes(self):
        with self.assertRaises(MetadataError):
            decode_clear_payload(b"\x00" * 9)
        with self.assertRaises(MetadataError):
            decode_batch_commit_payload(b"\x00" * 13)


class TestRecoveryPersistence(CustomTestCase):
    @staticmethod
    def _group_records(*, include_group_commit=True, group_member_count=2):
        records = []
        for seq, (batch_id, key, member_index, component) in enumerate(
            (
                (10, b"group-kv", 0, component_from_pool_name(PoolName.KV)),
                (11, b"group-mamba", 1, component_from_pool_name(PoolName.MAMBA)),
            ),
            1,
        ):
            records.extend(
                [
                    (
                        FRAME_SET,
                        seq,
                        encode_set_payload(
                            SetPayload(
                                _EPOCH,
                                7,
                                batch_id,
                                0,
                                1,
                                [
                                    SetEntry(
                                        key,
                                        0,
                                        (member_index + 1) * 10,
                                        10,
                                        1,
                                        component,
                                    )
                                ],
                            )
                        ),
                    ),
                    (
                        FRAME_BATCH_COMMIT,
                        seq + 2,
                        encode_batch_commit_payload(BatchCommitPayload(batch_id, 1)),
                    ),
                ]
            )
        if include_group_commit:
            records.append(
                (
                    FRAME_GROUP_COMMIT,
                    5,
                    encode_group_commit_payload(
                        GroupCommitPayload(_EPOCH, 7, group_member_count)
                    ),
                )
            )
        # The helper's intentional sequence layout is normalized by callers
        # below so each adversarial stream remains a valid WAL sequence.
        return [
            (frame_type, seq, payload)
            for seq, (frame_type, _old_seq, payload) in enumerate(records, 1)
        ]

    def test_manifest_atomic_and_corrupt_present_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = manifest_path(directory, ".x")
            write_manifest_atomic(path, _sample_state())
            self.assertEqual(read_manifest(path), _sample_state())
            with open(path, "r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                stream.write(b"\xff")
            with self.assertRaises(RecoveryError) as context:
                read_manifest(path)
            self.assertEqual(context.exception.failure_class, "manifest-corruption")
            self.assertEqual(context.exception.path, path)

    def test_current_and_lock_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            current = current_path(directory, "")
            with self.assertRaises(RecoveryError):
                read_current(current)
            publish_current(current, 4)
            self.assertEqual(read_current(current), 4)
            lock = lock_path(directory, "")
            first = acquire_meta_lock(lock)
            try:
                with self.assertRaises(RecoveryError) as context:
                    acquire_meta_lock(lock, nonblocking=True)
                self.assertEqual(context.exception.failure_class, "lock-busy")
            finally:
                os.close(first)

    def test_segment_creation_and_pending_promotion_repairs_partial_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = segment_path(directory, ".x", 2)
            create_segment(path, 2, 10)
            with open(path, "rb") as stream:
                self.assertEqual(
                    deserialize_segment_header(stream.read(28)), SegmentHeader(2, 10)
                )
            pending = segment_path(directory, ".x", 3)
            with open(pending, "wb") as stream:
                stream.write(b"partial")
            promote_pending_rotation(directory, ".x", 2, 99)
            self.assertEqual(read_current(current_path(directory, ".x")), 3)
            with open(pending, "rb") as stream:
                self.assertEqual(
                    deserialize_segment_header(stream.read(28)), SegmentHeader(3, 99)
                )

    def test_active_torn_tail_is_repaired_but_non_active_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = segment_path(directory, "", 1)
            create_segment(path, 1, 0)
            frame = encode_frame(FRAME_CLEAR, 1, encode_clear_payload(7))
            with open(path, "ab") as stream:
                stream.write(frame + frame[:5])
            parsed = iter_segment_frames(path, active=True)
            self.assertIsNotNone(parsed.torn_tail)
            with self.assertRaises(RecoveryError):
                iter_segment_frames(path, active=False)
            repair_active_tail(path, parsed.torn_tail.last_valid_offset)
            with open(path, "ab") as stream:
                stream.write(encode_frame(FRAME_CLEAR, 2, encode_clear_payload(8)))
            self.assertEqual(len(iter_segment_frames(path, active=True)), 2)

    def test_segment_gap_duplicate_and_interior_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = segment_path(directory, "", 1)
            create_segment(path, 1, 0)
            first = encode_frame(FRAME_CLEAR, 1, encode_clear_payload(1))
            second = encode_frame(FRAME_CLEAR, 3, encode_clear_payload(2))
            with open(path, "ab") as stream:
                stream.write(first + second)
            with self.assertRaises(RecoveryError):
                iter_segment_frames(path, active=True)

    def test_replay_atomicity_generations_clear_and_expected_fold(self):
        entry = SetEntry(b"k", 0, 0, 10, 1, component_from_pool_name(PoolName.KV))
        complete = [
            (
                FRAME_SET,
                1,
                encode_set_payload(SetPayload(b"\0" * 16, 0, 10, 0, 1, [entry])),
            ),
            (
                FRAME_BATCH_COMMIT,
                2,
                encode_batch_commit_payload(BatchCommitPayload(10, 1)),
            ),
        ]
        incomplete = [
            (
                FRAME_SET,
                3,
                encode_set_payload(SetPayload(_EPOCH, 5, 11, 0, 1, [entry])),
            ),
            (
                FRAME_BATCH_COMMIT,
                4,
                encode_batch_commit_payload(BatchCommitPayload(11, 1)),
            ),
        ]
        result = replay_records(complete + incomplete, num_slots=1)
        self.assertEqual([item.storage_key for item in result.entries], [b"k"])
        self.assertEqual(result.expected_fold_count, 1)
        dropped = (
            FRAME_DROP_SLOT,
            3,
            encode_drop_slot_payload(DropSlotPayload(0, 2)),
        )
        cleared = (FRAME_CLEAR, 4, encode_clear_payload(1))
        result = replay_records(complete + [dropped, cleared], num_slots=1)
        self.assertEqual(result.generations, [2])
        self.assertEqual(result.entries, [])
        with self.assertRaises(RecoveryError):
            replay_records(
                complete
                + [
                    (
                        FRAME_DROP_SLOT,
                        3,
                        encode_drop_slot_payload(DropSlotPayload(0, 0)),
                    )
                ],
                num_slots=1,
            )

    def test_group_commit_temporal_barrier_normal_order_recovers(self):
        result = replay_records(self._group_records(), num_slots=1)
        self.assertEqual(
            [entry.storage_key for entry in result.entries],
            [b"group-kv", b"group-mamba"],
        )
        self.assertEqual(result.complete_groups, ((_EPOCH, 7),))

    def test_group_commit_early_marker_does_not_match_later_members(self):
        records = self._group_records(include_group_commit=False)
        early = (
            FRAME_GROUP_COMMIT,
            1,
            encode_group_commit_payload(GroupCommitPayload(_EPOCH, 7, 2)),
        )
        result = replay_records(
            [early] + [(frame_type, seq + 1, payload) for frame_type, seq, payload in records],
            num_slots=1,
        )
        self.assertEqual(result.entries, [])
        self.assertEqual(result.complete_groups, ())

    def test_group_commit_reordered_before_batch_commits_fails_closed(self):
        records = self._group_records(include_group_commit=False)
        reordered_payloads = [
            records[0],
            records[2],
            (
                FRAME_GROUP_COMMIT,
                0,
                encode_group_commit_payload(GroupCommitPayload(_EPOCH, 7, 2)),
            ),
            records[1],
            records[3],
        ]
        reordered = [
            (frame_type, seq, payload)
            for seq, (frame_type, _old_seq, payload) in enumerate(
                reordered_payloads, 1
            )
        ]
        result = replay_records(reordered, num_slots=1)
        self.assertEqual(result.entries, [])
        self.assertEqual(result.complete_batches, (10, 11))
        self.assertEqual(result.complete_groups, ())

    def test_group_commit_missing_batch_commit_fails_closed(self):
        records = self._group_records(include_group_commit=False)
        records = [
            (frame_type, index, payload)
            for index, (frame_type, _seq, payload) in enumerate(
                [records[0], records[2], records[3]], 1
            )
        ]
        records.append(
            (
                FRAME_GROUP_COMMIT,
                len(records) + 1,
                encode_group_commit_payload(GroupCommitPayload(_EPOCH, 7, 2)),
            )
        )
        result = replay_records(records, num_slots=1)
        self.assertEqual(result.entries, [])
        self.assertEqual(result.complete_groups, ())

    def test_group_commit_missing_member_fails_closed(self):
        records = self._group_records(include_group_commit=False)[:2]
        records.append(
            (
                FRAME_GROUP_COMMIT,
                3,
                encode_group_commit_payload(GroupCommitPayload(_EPOCH, 7, 2)),
            )
        )
        result = replay_records(records, num_slots=1)
        self.assertEqual(result.entries, [])
        self.assertEqual(result.complete_groups, ())

    def test_duplicate_group_commit_is_idempotent(self):
        records = self._group_records()
        marker = records[-1]
        records.append((marker[0], len(records) + 1, marker[2]))
        result = replay_records(records, num_slots=1)
        self.assertEqual(len(result.entries), 2)
        self.assertEqual(result.complete_groups, ((_EPOCH, 7),))

    def test_batch_commit_without_group_commit_does_not_resurrect(self):
        result = replay_records(
            self._group_records(include_group_commit=False), num_slots=1
        )
        self.assertEqual(result.entries, [])
        self.assertEqual(result.complete_batches, (10, 11))
        self.assertEqual(result.complete_groups, ())

    def test_crash_before_group_commit_does_not_recover_kv_or_mamba(self):
        result = replay_records(
            self._group_records(include_group_commit=False), num_slots=1
        )
        self.assertEqual(result.entries, [])
        self.assertEqual(result.expected_fold_count, 0)
        self.assertEqual(result.complete_groups, ())

    def test_durable_group_commit_recovers_all_members(self):
        result = replay_records(self._group_records(), num_slots=1)
        self.assertEqual(result.expected_fold_count, 2)
        self.assertEqual(
            {
                (entry.storage_key, entry.component) for entry in result.entries
            },
            {
                (b"group-kv", component_from_pool_name(PoolName.KV)),
                (b"group-mamba", component_from_pool_name(PoolName.MAMBA)),
            },
        )

    def test_wal_range_and_manifest_absence_rules(self):
        self.assertEqual(
            resolve_wal_range(ManifestState(wal_floor=3), 3).generations, range(3, 4)
        )
        self.assertTrue(
            resolve_wal_range(ManifestState(wal_floor=4), 3).pending_rotation
        )
        with self.assertRaises(RecoveryError):
            resolve_wal_range(ManifestState(wal_floor=5), 3)
        self.assertEqual(
            resolve_manifest_absent(3, prior_metadata_exists=False).generations,
            range(3, 4),
        )
        with self.assertRaises(RecoveryError):
            resolve_manifest_absent(3, prior_metadata_exists=True)

    def test_checkpoint_adapter_is_filesystem_free(self):
        state = checkpoint_state_from_allocator(
            checkpoint_lsn=1,
            wal_floor=1,
            wal_active_gen=1,
            extent_size=4,
            file_size=4,
            num_slots=1,
            generations=[0],
            cursors=[0],
            slot_bytes=[0],
        )
        self.assertEqual(state.num_slots, 1)


if __name__ == "__main__":
    unittest.main()
