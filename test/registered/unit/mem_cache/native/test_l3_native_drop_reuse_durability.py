# SPDX-License-Identifier: Apache-2.0
"""Arm-B behavioural oracle: DROP_SLOT durable before the reuse data pwrite.

The frozen arm-A test ``test_drop_slot_is_durable_before_reuse_pwrite``
(``test_extent_file_backend_unit.py``) proves the ordering by monkeypatching
the Python ``ExtentFileBackend._pwritev``.  Arm B (``SGLANG_L3_NATIVE_WRITE_PATH=1``)
performs the syscall in the Rust worker, so that mock never fires -- the
ordering was previously only established structurally.

This file closes that gap **behaviourally on arm B**, using the real Rust
worker, the real data/WAL fds, a real hard crash (``os._exit``) and a real
recovery.  It is the silent-corruption boundary: if a slot's data ``pwrite``
were issued before the victim's ``DROP_SLOT`` became durable, a crash in that
window would let recovery resurrect the *old* storage identity pointing at the
slot now holding the *new* key's bytes.

Positive case (``ordered``):
  * ``set("old", A)`` commits (slot S, generation g).
  * ``set("new", B)`` selects the GC victim ``old``; the backend appends and
    ``fdatasync``s ``FRAME_DROP_SLOT(S, g+1)`` **before** the arm-B write branch.
  * the WAL fd is then made read-only so the arm-B job lands in the window
    "reuse data pwrite done, no FRAME_SET/BATCH_COMMIT"; the child hard-crashes.
  * recovery must show ``old`` MISS (its generation is stale by exactly the
    durable DROP_SLOT), ``new`` MISS (never committed), and a stale
    pre-crash ref must fail generation re-verification.

Negative control (``inverted``):
  * identical, except ``_append_drops`` is suppressed so the DROP_SLOT is not
    persisted before the reuse pwrite -- i.e. the ordering the invariant
    forbids.  Recovery then resurrects ``old`` and returns B (corruption),
    which is exactly the failure the positive assertion is sensitive to.

CPU-only.  ``CUDA_VISIBLE_DEVICES`` must be a non-GPU ordinal (e.g. 999) for
the frozen ``sglang.test.test_utils`` import path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.extent.extent_file_backend import (
    ExtentFileBackend,
)
from sglang.srt.mem_cache.storage.extent.extent_metadata import (
    FRAME_BATCH_COMMIT,
    FRAME_DROP_SLOT,
    FRAME_SET,
    iter_segment_frames,
    segment_path,
)
from sglang.test.test_utils import CustomTestCase

_NATIVE_ENV = "SGLANG_L3_NATIVE_WRITE_PATH"
_PAGE_BYTES = 1000
_EXTENT_SIZE = 16 * 1024


def _make_config(storage_dir: str) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        extra_config={
            "storage_dir": storage_dir,
            "extent_size": "16Ki",
            "file_size": "64Ki",
            # Force a GC victim on the second 1000-byte write: used (1000) is
            # above max_size * eviction_ratio (512), so "old" is dropped and
            # its slot is reused by "new".
            "max_size": "1Ki",
            "eviction_ratio": 0.5,
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
        model_name="drop-reuse",
    )


# The child runs arm B against the real Rust worker.  ``L3DR_MODE`` selects the
# ordering under test; the crash is a hard ``os._exit`` with no close, no
# checkpoint and no manifest update.
_CHILD = r'''
import json, os, sys
from pathlib import Path
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.extent.extent_file_backend import ExtentFileBackend
from sglang.srt.mem_cache.storage.extent.extent_metadata import iter_segment_frames

storage = os.environ["L3DR_STORAGE"]
mode = os.environ["L3DR_MODE"]
result_path = os.environ["L3DR_RESULT"]

A = bytes([0xA1]) * 1000
B = bytes([0xB2]) * 1000

cfg = HiCacheStorageConfig(
    extra_config={"storage_dir": storage, "extent_size": "16Ki",
                  "file_size": "64Ki", "max_size": "1Ki", "eviction_ratio": 0.5},
    tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
    attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
    enable_storage_metrics=False, is_page_first_layout=True,
    model_name="drop-reuse",
)
backend = ExtentFileBackend(cfg)
old_key = backend._storage_key("old")

out = {"mode": mode, "arm": int(os.environ.get("SGLANG_L3_NATIVE_WRITE_PATH", "0"))}
assert backend.set("old", bytearray(A)), "setup: set(old) must commit"
old_ref = backend._allocator.lookup(old_key)
out["old_slot"] = int(old_ref.extent_id)
out["old_offset"] = int(old_ref.offset)
out["old_length"] = int(old_ref.length)
out["old_generation"] = int(old_ref.generation)

seg = os.path.join(storage, "hicache.meta%s.wal.0" % backend.config_suffix)
data = os.path.join(storage, "hicache.data%s" % backend.config_suffix)

real_append_drops = backend._append_drops

def faulting_append_drops(drop_events):
    drop_events = list(drop_events)
    out["drop_events"] = [[int(s), int(g)] for s, g in drop_events]
    if mode == "ordered":
        # The invariant: the victim drop is written AND fdatasync'd before the
        # reuse data pwrite.  `_append_drops` does both; assert it did fire.
        real_append_drops(drop_events)
    # Make the arm-B WAL append fail so the crash lands after the data pwrite
    # and before any FRAME_SET/BATCH_COMMIT -- regardless of ordering mode.
    os.close(backend._wal_fd)
    backend._wal_fd = os.open(seg, os.O_RDONLY)

backend._append_drops = faulting_append_drops

out["set_new_ok"] = bool(backend.set("new", bytearray(B)))
assert out["set_new_ok"] is False, "fault injection failed: set(new) must not commit"

# Evidence that the window was actually reached: the reuse data pwrite landed
# in the old slot (arm B wrote it natively, before the broken WAL append).
with open(data, "rb") as fh:
    fh.seek(out["old_slot"] * 16384 + out["old_offset"])
    slot_after = fh.read(out["old_length"])
out["slot_bytes_are_new"] = slot_after == B
out["slot_bytes_are_old"] = slot_after == A

frames = list(iter_segment_frames(seg, active=True))
out["wal_frame_types"] = [int(f.frame_type) for f in frames]
out["wal_has_drop_slot"] = any(
    int(f.frame_type) == 4 for f in frames
)

Path(result_path).write_text(json.dumps(out))
os._exit(0)
'''


class NativeDropReuseDurabilityTests(CustomTestCase):
    """Arm-B crash/recovery oracle for the DROP_SLOT-before-reuse invariant."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="l3_drop_reuse_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        os.environ.pop(_NATIVE_ENV, None)

    def _run_crash_child(self, mode: str) -> dict:
        directory = self.root / mode
        directory.mkdir()
        result_file = self.root / f"{mode}.json"
        env = dict(os.environ)
        env[_NATIVE_ENV] = "1"
        env["L3DR_STORAGE"] = str(directory)
        env["L3DR_MODE"] = mode
        env["L3DR_RESULT"] = str(result_file)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        self.assertTrue(result_file.exists(), proc.stderr[-3000:])
        return json.loads(result_file.read_text())

    def _read_slot(self, directory: Path, slot: int, offset: int, length: int) -> bytes:
        data_path = next(directory.glob("hicache.data*"))
        with data_path.open("rb") as fh:
            fh.seek(slot * _EXTENT_SIZE + offset)
            return fh.read(length)

    def test_native_drop_slot_durable_before_reuse_pwrite(self):
        """Arm B: after a crash in the reuse window, recovery must NOT resurrect
        the dropped identity, and the stale generation must fail re-verify."""
        child = self._run_crash_child("ordered")
        directory = self.root / "ordered"

        self.assertEqual(child["arm"], 1, "child must run the native arm")
        self.assertFalse(child["set_new_ok"])
        self.assertIn(FRAME_DROP_SLOT, child["wal_frame_types"],
                      "the durable WAL must carry the victim DROP_SLOT")
        # The WAL legitimately holds the *old* batch's SET/BATCH_COMMIT before
        # the drop; what must be absent is any frame after the durable drop
        # (i.e. the reusing batch never reached FRAME_SET/BATCH_COMMIT).
        types = child["wal_frame_types"]
        self.assertEqual(types.count(FRAME_DROP_SLOT), 1)
        self.assertEqual(
            types[types.index(FRAME_DROP_SLOT) + 1:], [],
            "a frame followed the durable DROP_SLOT in the crash window",
        )
        self.assertIn(FRAME_SET, types)
        self.assertIn(FRAME_BATCH_COMMIT, types)
        self.assertTrue(child["slot_bytes_are_new"],
                        "the window was not reached: reuse data pwrite must "
                        "have overwritten the old slot")
        self.assertEqual(len(child["drop_events"]), 1)
        dropped_slot, dropped_gen = child["drop_events"][0]
        self.assertEqual(dropped_slot, child["old_slot"])
        self.assertEqual(dropped_gen, child["old_generation"] + 1)

        # Restart/recover (fresh arm-B instance, same storage).
        backend = ExtentFileBackend(_make_config(str(directory)))
        try:
            old_key = backend._storage_key("old")
            new_key = backend._storage_key("new")
            self.assertFalse(backend.exists("old"),
                             "old identity was resurrected after the crash")
            self.assertFalse(backend.exists("new"),
                             "uncommitted new identity became visible")
            # ``_read_pages`` is the raw read path; ``get`` needs a target.
            self.assertIsNone(backend._read_pages([old_key])[0])
            self.assertIsNone(backend._read_pages([new_key])[0])
            # The stale pre-crash identity is provably unusable.
            from sglang.srt.mem_cache.storage.extent.extent_allocator import (
                ExtentRef,
            )

            stale = ExtentRef(
                child["old_slot"], child["old_offset"],
                child["old_length"], child["old_generation"],
            )
            self.assertEqual(
                backend._allocator.verify([old_key], [stale]), [False],
                "stale generation passed read-side re-verification",
            )
            self.assertEqual(
                backend._allocator._generations[child["old_slot"]],
                child["old_generation"] + 1,
                "restored generation did not advance by the durable DROP_SLOT",
            )
        finally:
            backend.close()

        # The slot really does hold the new bytes -- so the miss above is a
        # correct refusal, not a vacuous absence.
        self.assertEqual(
            self._read_slot(directory, child["old_slot"],
                            child["old_offset"], child["old_length"]),
            bytes([0xB2]) * _PAGE_BYTES,
        )

    def test_inverted_control_resurrects_corruption(self):
        """Sensitivity control: with the drop NOT made durable before the reuse
        pwrite, recovery resurrects 'old' and returns the new key's bytes."""
        child = self._run_crash_child("inverted")
        directory = self.root / "inverted"

        self.assertEqual(child["arm"], 1)
        self.assertFalse(child["set_new_ok"])
        self.assertTrue(child["slot_bytes_are_new"])
        self.assertNotIn(FRAME_DROP_SLOT, child["wal_frame_types"],
                         "the inverted control must NOT persist the drop")

        backend = ExtentFileBackend(_make_config(str(directory)))
        try:
            # The exact assertion the positive test makes is now violated:
            # this is the corruption the ordering prevents.
            self.assertTrue(
                backend.exists("old"),
                "control invalid: inverted ordering did not resurrect old",
            )
            got = backend._read_pages([backend._storage_key("old")])[0]
            self.assertIsNotNone(got)
            self.assertEqual(bytes(got), bytes([0xB2]) * _PAGE_BYTES,
                             "control did not expose the new bytes under the "
                             "old identity")
            self.assertNotEqual(bytes(got), bytes([0xA1]) * _PAGE_BYTES)
            self.assertFalse(backend.exists("new"))
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
