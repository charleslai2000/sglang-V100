# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Pure-logic circular extent allocator for the ExtentFileBackend.

File layout contract (the backend's pwrite target):

    slot s occupies file bytes [s * extent_size, (s + 1) * extent_size);
    a key's bytes live at [slot_base + ref.offset, + ref.length).

The allocator owns all in-memory state: per-slot generations, the pack
cursor, the key -> ExtentRef index, per-slot bookkeeping, and the LRU
recency order. It never touches the filesystem and holds no lock -- the
backend serializes every call under a single index lock and performs the
page IO outside that lock.

Allocation is two-phase so a failed pwrite leaves no state behind:
``plan`` computes refs purely (zero mutation), ``commit`` applies them once
the write succeeded. Reusing a slot (the circular cursor wrapping into a
slot that cannot hold another page) or watermark GC eviction bumps the
slot's generation, so any index entry captured before the reuse fails the
read-side re-verification (generation mismatch = miss).

The G1 split exposes the same discipline with an explicit invalidation
step: ``plan_split`` is pure and returns ``drop_events``, ``apply_drops``
is the only new-path generation bump, and ``commit_install`` is
install-only. The legacy ``plan``/``commit`` pair remains byte-compatible
for the existing backend callers.
"""

from __future__ import annotations

import logging
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple

import msgspec

from sglang.srt.mem_cache.storage.extent.extent_trace import (
    attach_trace_context,
    marker_present,
)

logger = logging.getLogger(__name__)

_T38C_DURABLE_SEQ = {}
_T38C_DURABLE_SEQ_LOCK = threading.Lock()


def _t38c_durable_log(rec):
    if not marker_present("/tmp/t38c-durable-trace"):
        return
    try:
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
            f"{run_id}:{os.getpid()}:{thread_id}:allocator:{local_seq}",
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
            "slot_digest",
            "old_generation",
            "new_generation",
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
        with open("/tmp/t38c-probe.log", "a") as trace_file:
            trace_file.write(json.dumps(record, default=str) + "\n")
        return record["event_id"]
    except Exception:
        pass


class ExtentRef(msgspec.Struct, frozen=True):
    extent_id: int
    offset: int
    length: int
    generation: int


class ExtentAllocator:
    """Circular extent allocator with generation-based invalidation.

    All methods require the caller to hold the backend's index lock; there
    is no internal synchronization.
    """

    def __init__(self, extent_size: int, num_slots: int):
        if extent_size <= 0 or extent_size % 4096 != 0:
            # 4KiB sector alignment: the reserved O_DIRECT calibration tier
            # can only touch whole extents, so slot geometry must stay aligned.
            raise ValueError(
                f"extent_size must be a positive multiple of 4096, got {extent_size}"
            )
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self.extent_size = extent_size
        self.num_slots = num_slots
        self._generations = [0] * num_slots
        self._next_slot = 0
        self._cursors = [0] * num_slots
        self._slot_keys: List[set] = [set() for _ in range(num_slots)]
        self._slot_bytes = [0] * num_slots
        self._index: dict = {}
        self._recency: OrderedDict = OrderedDict()
        # All-or-nothing guard for the split install path (NEW-4).  The
        # backend still serializes logical calls under its own index lock;
        # this lock only makes apply_drops/commit_install internally atomic.
        self._lock = threading.RLock()

    # ---- read side ----

    @property
    def used_bytes(self) -> int:
        return sum(self._slot_bytes)

    def lookup(self, key: str) -> Optional[ExtentRef]:
        ref = self._index.get(key)
        if ref is None:
            return None
        if ref.generation != self._generations[ref.extent_id]:
            # Stale entry that survived a slot drop; treat as absent.
            self._index.pop(key, None)
            self._recency.pop(key, None)
            return None
        return ref

    def snapshot(self, keys: Sequence[str]) -> List[Optional[ExtentRef]]:
        return [self.lookup(k) for k in keys]

    def verify(
        self, keys: Sequence[str], refs: Sequence[Optional[ExtentRef]]
    ) -> List[bool]:
        """Re-validate that each key still maps to its exact snapshot ref.

        Slot reuse since the snapshot (eviction, cursor wrap, rewrite) changes
        the entry or its generation; the snapshot data must not be served.
        """
        ok: List[bool] = []
        for key, ref in zip(keys, refs):
            ok.append(ref is not None and self._index.get(key) == ref)
        return ok

    def touch(self, key: str) -> None:
        if key in self._index:
            self._recency.pop(key, None)
            self._recency[key] = None

    # ---- write side ----

    def plan(
        self,
        key_lengths: Sequence[Tuple[str, int]],
        pending_drops: Optional[Sequence[int]] = None,
    ) -> Tuple[List[Optional[ExtentRef]], List[int]]:
        """Backward-compatible 2-tuple wrapper around :meth:`plan_split`.

        Returns ``(refs, passed)`` where ``passed`` lists the slots the pack
        cursor enters that cannot fit the batch and are therefore dropped
        and reset (circular reuse; their old data is the oldest in the
        file); ``commit`` applies those drops plus the new index entries.

        The optional ``pending_drops`` set carries the GC-victim slots known
        to this plan.  Generation numbers use the unified NEW-1 form, which
        is identical to the legacy ``_generations[slot] + passed.count(slot)``
        whenever ``pending_drops`` is empty (the existing backend callers).
        """
        refs, passed, _drop_events = self._plan_pure(key_lengths, pending_drops)
        return refs, passed

    def plan_split(
        self,
        key_lengths: Sequence[Tuple[str, int]],
        pending_drops: Optional[Sequence[int]] = None,
    ) -> Tuple[
        List[Optional[ExtentRef]], List[int], List[Tuple[int, int]]
    ]:
        """Pure plan returning ``(refs, passed, drop_events)`` (NEW-1/NEW-4).

        ``pending_drops`` is the set of GC-victim slots selected purely for
        this plan; it may be None or empty.  The unified generation form
        bumps any slot in ``passed``, ``pending_drops``, or both exactly
        once::

            pending_gen[s] = _generations[s] + max(
                passed.count(s), 1 if s in pending_drops else 0
            )
            ref.gen = pending_gen[slot]

        ``drop_events`` lists ``(slot, pending_gen[slot])`` for
        ``passed`` U ``pending_drops`` exactly once per slot.  No state is
        mutated; ``apply_drops`` later consumes ``drop_events``.
        """
        return self._plan_pure(key_lengths, pending_drops)

    def _pending_generation(
        self,
        slot: int,
        passed: Sequence[int],
        pending_drops_slots: set,
    ) -> int:
        return self._generations[slot] + max(
            passed.count(slot), 1 if slot in pending_drops_slots else 0
        )

    def _plan_pure(
        self,
        key_lengths: Sequence[Tuple[str, int]],
        pending_drops: Optional[Sequence[int]] = None,
    ) -> Tuple[
        List[Optional[ExtentRef]], List[int], List[Tuple[int, int]]
    ]:
        """Shared pure planner (see :meth:`plan_split` for the contract).

        Placement is computed exactly as if ``_drop_slot`` had been applied
        once per ``pending_drops`` slot (cursor reset to 0, slot empty): a
        pending-drop slot is never treated as its pre-drop live cursor, and
        because it is empty it never contributes a ``passed`` entry -- it is
        already a drop_event, so overlap stays a single event / single bump.
        """
        pending_drops_slots = set(pending_drops) if pending_drops else set()
        refs: List[Optional[ExtentRef]] = []
        passed: List[int] = []
        packed: set = set()
        slot = self._next_slot
        cursor = 0 if slot in pending_drops_slots else self._cursors[slot]
        for _key, length in key_lengths:
            if length > self.extent_size:
                refs.append(None)
                continue
            while cursor + length > self.extent_size:
                slot = (slot + 1) % self.num_slots
                if slot in packed:
                    # Batch larger than the whole file: placement would
                    # overwrite this plan's own pages; refuse the batch.
                    return [None] * len(key_lengths), [], []
                if slot in pending_drops_slots:
                    # Victim dropped in this plan: empty with cursor 0, so
                    # place from the start of the slot and do NOT append it
                    # to ``passed`` (it is already a drop_event).
                    cursor = 0
                else:
                    cursor = self._cursors[slot]
                    if cursor + length > self.extent_size:
                        passed.append(slot)
                        cursor = 0
            packed.add(slot)
            refs.append(
                ExtentRef(
                    slot,
                    cursor,
                    length,
                    self._pending_generation(slot, passed, pending_drops_slots),
                )
            )
            cursor += length
        drop_slots = list(dict.fromkeys(passed))
        for dropped in sorted(pending_drops_slots):
            if dropped not in drop_slots:
                drop_slots.append(dropped)
        drop_events = [
            (
                dropped,
                self._pending_generation(dropped, passed, pending_drops_slots),
            )
            for dropped in drop_slots
        ]
        return refs, passed, drop_events

    def commit(
        self,
        keys: Sequence[str],
        refs: Sequence[ExtentRef],
        passed: List[int],
        trace_context: Optional[dict] = None,
    ) -> None:
        """Apply a plan after its pwrite succeeded (the backend's ACK point)."""
        for slot in passed:
            self._drop_slot(
                slot, reason="commit_reuse", trace_context=trace_context
            )
        for key, ref in zip(keys, refs):
            if ref is None:
                continue
            old = self._index.get(key)
            if old is not None:
                self._slot_keys[old.extent_id].discard(key)
                self._slot_bytes[old.extent_id] -= old.length
            self._index[key] = ref
            self._slot_keys[ref.extent_id].add(key)
            self._slot_bytes[ref.extent_id] += ref.length
            self._cursors[ref.extent_id] = ref.offset + ref.length
            self._recency.pop(key, None)
            self._recency[key] = None
            self._next_slot = ref.extent_id

    def apply_drops(
        self,
        drop_events: Sequence[Tuple[int, int]],
        trace_context: Optional[dict] = None,
    ) -> None:
        """Apply ``(slot, new_generation)`` drops (NEW-4).

        This is the only new-path method that removes keys, resets
        ``_cursors``/``_slot_bytes``, and bumps generations.  It is
        idempotent per ``(slot, new_generation)``: a slot already at
        ``new_generation`` is a no-op, and a request that would decrease a
        generation fails closed (NM-INV-06).
        """
        with self._lock:
            for slot, new_generation in drop_events:
                old_generation = self._generations[slot]
                if old_generation == new_generation:
                    continue
                if old_generation > new_generation:
                    raise ValueError(
                        "ExtentAllocator: generation decrease for slot "
                        f"{slot}: {old_generation} -> {new_generation}"
                    )
                freed = self._slot_bytes[slot]
                removed_keys = sorted(self._slot_keys[slot])
                for key in removed_keys:
                    self._index.pop(key, None)
                    self._recency.pop(key, None)
                self._slot_keys[slot].clear()
                self._slot_bytes[slot] = 0
                self._cursors[slot] = 0
                self._generations[slot] = new_generation
                for key in removed_keys:
                    _t38c_durable_log(
                        {
                            "evt": "storage_invalidate",
                            "trace_context": trace_context,
                            "operation": "allocator_apply_drops",
                            "pool": "storage",
                            "storage_key": key,
                            "extent_id": slot,
                            "extent_generation": new_generation,
                            "old_generation": old_generation,
                            "new_generation": new_generation,
                            "index_removed": True,
                            "reason": "apply_drops",
                            "result": True,
                        }
                    )
                if freed > 0:
                    logger.info(
                        "[P3] ts_ms=%d evt=extent_recycle slot=%d "
                        "generation=%d freed=%d",
                        int(time.time() * 1000),
                        slot,
                        new_generation,
                        freed,
                    )

    def commit_install(
        self,
        keys: Sequence[str],
        refs: Sequence[Optional[ExtentRef]],
    ) -> bool:
        """Install-only commit: never drops and never re-reads ``passed``.

        All-or-nothing under the allocator lock: every non-None ref is
        validated first (slot range, geometry, and ``ref.generation`` equal
        to the slot's current generation -- the value applied by
        :meth:`apply_drops`).  On any validation failure nothing is
        installed and ``False`` is returned.
        """
        with self._lock:
            if len(keys) != len(refs):
                return False
            for ref in refs:
                if ref is None:
                    continue
                if not 0 <= ref.extent_id < self.num_slots:
                    return False
                if (
                    ref.offset < 0
                    or ref.length < 0
                    or ref.offset + ref.length > self.extent_size
                ):
                    return False
                if ref.generation != self._generations[ref.extent_id]:
                    return False
            for key, ref in zip(keys, refs):
                if ref is None:
                    continue
                old = self._index.get(key)
                if old is not None:
                    self._slot_keys[old.extent_id].discard(key)
                    self._slot_bytes[old.extent_id] -= old.length
                self._index[key] = ref
                self._slot_keys[ref.extent_id].add(key)
                self._slot_bytes[ref.extent_id] += ref.length
                self._cursors[ref.extent_id] = ref.offset + ref.length
                self._recency.pop(key, None)
                self._recency[key] = None
                self._next_slot = ref.extent_id
            return True

    # ---- reclaim-time live-ref relocation ----

    def plan_relocation(
        self,
        victim_slots: Sequence[int],
        key_lengths: Sequence[Tuple[str, int]],
        excluded_slots: Sequence[int] = (),
        start_slot: Optional[int] = None,
    ) -> Optional[List[ExtentRef]]:
        """Pure placement of relocated refs outside the victim generation.

        Circular reclaim bumps a victim slot's generation and reuses its
        bytes, destroying every live index entry that still points into that
        generation.  Before any of that happens the backend enumerates the
        victim's live refs and asks for replacement space here.

        Relocated refs are appended to the live tail (``_cursors``) of slots
        that are neither victims nor ``excluded_slots`` (the in-flight plan's
        own refs and the staged reservations), scanning forward circularly
        from ``start_slot``.  The method is pure: it never passes, drops,
        resets or recurses into a slot, and it never bumps a generation -- the
        returned refs carry each destination slot's *current* generation.

        Returns a list aligned with ``key_lengths``, or ``None`` when the
        total free tail space is smaller than the requested bytes (fail
        closed: the caller must not touch the victim).  The caller performs
        the byte IO and the durable publish (SET -> BATCH_COMMIT) before
        applying the drops.
        """
        avoid = set(victim_slots) | set(excluded_slots)
        if not key_lengths:
            return []
        for _key, length in key_lengths:
            if length < 0 or length > self.extent_size:
                return None
        if start_slot is None:
            start_slot = self._next_slot
        order: List[int] = []
        for step in range(self.num_slots):
            slot = (start_slot + step) % self.num_slots
            if slot not in avoid:
                order.append(slot)
        refs: List[ExtentRef] = []
        floors = {slot: self._cursors[slot] for slot in order}
        index = 0
        for slot in order:
            room = self.extent_size - floors[slot]
            while index < len(key_lengths) and (room > 0 or key_lengths[index][1] == 0):
                length = key_lengths[index][1]
                if length > room:
                    break
                refs.append(
                    ExtentRef(slot, floors[slot], length, self._generations[slot])
                )
                floors[slot] += length
                room -= length
                index += 1
            if index >= len(key_lengths):
                break
        if index < len(key_lengths):
            return None
        return refs

    # ---- GC ----

    def evict_slot_lru(self, trace_context: Optional[dict] = None) -> int:
        """Drop the slot holding the least-recently-used key; returns the
        number of live bytes freed (0 when there is nothing to evict)."""
        if not self._recency:
            return 0
        oldest_key = next(iter(self._recency))
        oldest = self._index.get(oldest_key)
        if oldest is None:
            self._recency.pop(oldest_key, None)
            return 0
        return self._drop_slot(
            oldest.extent_id, reason="lru_gc", trace_context=trace_context
        )

    def clear(self, trace_context: Optional[dict] = None) -> None:
        for slot, keys in enumerate(self._slot_keys):
            old_generation = self._generations[slot]
            for key in sorted(keys):
                _t38c_durable_log(
                    {
                        "evt": "storage_invalidate",
                        "trace_context": trace_context,
                        "operation": "allocator_clear",
                        "pool": "storage",
                        "storage_key": key,
                        "extent_id": slot,
                        "extent_generation": old_generation,
                        "old_generation": old_generation,
                        "new_generation": old_generation,
                        "index_removed": True,
                        "reason": "backend_clear",
                        "result": True,
                    }
                )
        self._index.clear()
        self._recency.clear()
        for slot in range(self.num_slots):
            self._slot_keys[slot].clear()
            self._slot_bytes[slot] = 0
            self._cursors[slot] = 0
        # Generations stay monotonic; any slot reuse still invalidates.

    # ---- internals ----

    def _drop_slot(
        self,
        slot: int,
        reason: str = "unknown",
        trace_context: Optional[dict] = None,
    ) -> int:
        old_generation = self._generations[slot]
        freed = self._slot_bytes[slot]
        removed_keys = sorted(self._slot_keys[slot])
        for key in removed_keys:
            self._index.pop(key, None)
            self._recency.pop(key, None)
        self._slot_keys[slot].clear()
        self._slot_bytes[slot] = 0
        self._cursors[slot] = 0
        self._generations[slot] += 1
        for key in removed_keys:
            _t38c_durable_log(
                {
                    "evt": "storage_invalidate",
                    "trace_context": trace_context,
                    "operation": "allocator_drop_slot",
                    "pool": "storage",
                    "storage_key": key,
                    "extent_id": slot,
                    "extent_generation": self._generations[slot],
                    "old_generation": old_generation,
                    "new_generation": self._generations[slot],
                    "index_removed": True,
                    "reason": reason,
                    "result": True,
                }
            )
        if freed > 0:
            # [P3] extent_recycle: a slot's generation bumped (the index
            # entries captured before the reuse now fail the read-side
            # generation re-verification -- a storage miss). Emitted under
            # the allocator lock held by the backend's write path.
            logger.info(
                "[P3] ts_ms=%d evt=extent_recycle slot=%d generation=%d freed=%d",
                int(time.time() * 1000),
                slot,
                self._generations[slot],
                freed,
            )
        return freed
