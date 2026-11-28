"""Focused unit tests for the G1 split of the ``ExtentAllocator`` API.

These pin the NEW-1/NEW-4 contract only (pure ``plan_split`` returning
``drop_events``, ``apply_drops`` as the single generation bump, install-only
``commit_install``, and the byte-compatible legacy ``plan``/``commit``):

* a slot bumped as a ``passed`` reuse, a GC victim in ``pending_drops``, or
  both is bumped exactly once and its installed ref matches the post-drop
  ``_generations[slot]`` (never born stale);
* ``commit_install`` never bumps a generation;
* the legacy ``commit`` path is unchanged.

Run with:
    python test/registered/unit/mem_cache/test_extent_g1_allocator_unit.py
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.storage.extent.extent_allocator import (
    ExtentAllocator,
    ExtentRef,
)
from sglang.test.test_utils import CustomTestCase

_EXTENT = 4096


def _write_legacy(alloc: ExtentAllocator, key: str, length: int) -> ExtentRef:
    """Set up state through the legacy plan/commit pair (unchanged path)."""
    refs, passed = alloc.plan([(key, length)])
    assert refs[0] is not None, f"key {key} refused by allocator"
    alloc.commit([key], refs, passed)
    return refs[0]


class TestExtentG1AllocatorSplit(CustomTestCase):
    def _alloc(self, num_slots=4, extent_size=_EXTENT) -> ExtentAllocator:
        return ExtentAllocator(extent_size=extent_size, num_slots=num_slots)

    def _fill_two_full_slots(self, alloc: ExtentAllocator) -> None:
        """Fill both 4096-byte slots of a 2-slot allocator with 1000-byte keys."""
        for i in range(4):
            _write_legacy(alloc, f"s0-{i}", 1000)  # slot 0 full
        for i in range(4):
            _write_legacy(alloc, f"s1-{i}", 1000)  # slot 1 full

    # ---- plan_split is the pure split planner ----

    def test_plan_split_is_pure(self):
        alloc = self._alloc(num_slots=2)
        _write_legacy(alloc, "a", 1000)  # slot 0, cursor 1000
        gens_before = list(alloc._generations)
        used_before = alloc.used_bytes

        refs, passed, events = alloc.plan_split([("b", 1000)], pending_drops={1})

        # No mutation whatsoever.
        self.assertEqual(alloc._generations, gens_before)
        self.assertEqual(alloc.used_bytes, used_before)
        self.assertIsNone(alloc.lookup("b"))
        self.assertEqual(passed, [])
        # The GC victim slot is bumped once even though it is not passed.
        self.assertEqual(events, [(1, 1)])
        # The destination slot is untouched and keeps its generation.
        self.assertEqual((refs[0].extent_id, refs[0].offset), (0, 1000))
        self.assertEqual(refs[0].generation, 0)

    def test_plan_legacy_arity_and_split_equivalence(self):
        alloc = self._alloc(num_slots=2)
        out = alloc.plan([("a", 1000)])
        self.assertEqual(len(out), 2)  # legacy 2-tuple must stay unpackable
        refs2, passed2, events2 = alloc.plan_split([("a", 1000)])
        self.assertEqual(out[0], refs2)
        self.assertEqual(out[1], passed2)
        self.assertEqual(events2, [])
        # None and empty pending_drops are equivalent.
        self.assertEqual(
            alloc.plan_split([("a", 1000)], pending_drops=None)[2],
            alloc.plan_split([("a", 1000)], pending_drops=set())[2],
        )

    def test_batch_larger_than_whole_file_refused(self):
        alloc = self._alloc(num_slots=2, extent_size=_EXTENT)
        refs, passed, events = alloc.plan_split([(f"k{i}", 3000) for i in range(3)])
        self.assertTrue(all(r is None for r in refs))
        self.assertEqual(passed, [])
        self.assertEqual(events, [])
        self.assertEqual(alloc.used_bytes, 0)

    # ---- the four bump cases ----

    def test_case_destination_only(self):
        alloc = self._alloc(num_slots=2)
        refs, passed, events = alloc.plan_split([("a", 1000)], pending_drops=set())
        self.assertEqual(passed, [])
        self.assertEqual(events, [])
        alloc.apply_drops(events)
        self.assertTrue(alloc.commit_install(["a"], refs))
        self.assertEqual(refs[0].generation, alloc._generations[refs[0].extent_id])
        self.assertEqual(refs[0].generation, 0)  # fresh slot: no bump
        self.assertEqual(alloc.lookup("a"), refs[0])

    def test_case_in_passed(self):
        alloc = self._alloc(num_slots=2)
        self._fill_two_full_slots(alloc)

        refs, passed, events = alloc.plan_split([("x", 1000)], pending_drops=set())

        self.assertEqual(passed, [0])
        self.assertEqual(events, [(0, 1)])
        self.assertEqual(refs[0].extent_id, 0)
        self.assertEqual(refs[0].generation, 1)  # bumped exactly once
        alloc.apply_drops(events)
        self.assertEqual(alloc._generations[0], 1)
        self.assertTrue(alloc.commit_install(["x"], refs))
        self.assertEqual(refs[0].generation, alloc._generations[0])
        self.assertIsNone(alloc.lookup("s0-0"))  # dropped slot's keys gone
        self.assertEqual(alloc.lookup("x"), refs[0])
        self.assertIsNotNone(alloc.lookup("s1-0"))

    def test_case_gc_victim_in_pending_drops_not_passed(self):
        alloc = self._alloc(num_slots=1)
        _write_legacy(alloc, "old", 1000)  # slot 0, pre-drop cursor 1000

        refs, passed, events = alloc.plan_split([("new", 3000)], pending_drops={0})

        # Placement is computed as if the victim were dropped (cursor 0), so
        # it is reused at offset 0 and never counted as a circular pass.
        self.assertEqual(refs[0].offset, 0)
        self.assertEqual(passed, [])
        self.assertEqual(events, [(0, 1)])
        self.assertEqual(refs[0].extent_id, 0)
        # NM-F-07 repair: a victim not in passed is still born at the new
        # generation, never at the pre-drop one.
        self.assertEqual(refs[0].generation, 1)
        alloc.apply_drops(events)
        self.assertEqual(alloc._generations[0], 1)
        self.assertTrue(alloc.commit_install(["new"], refs))
        self.assertEqual(refs[0].generation, alloc._generations[refs[0].extent_id])
        self.assertIsNone(alloc.lookup("old"))
        self.assertEqual(alloc.lookup("new"), refs[0])

    def test_gc_victim_destination_ignores_pre_drop_cursor(self):
        # Fidelity case 1: a GC-victim destination is placed at offset 0, not
        # at its pre-drop cursor (1000), because the plan behaves as if
        # _drop_slot had reset the slot to empty.
        alloc = self._alloc(num_slots=1)
        _write_legacy(alloc, "old", 1000)  # slot 0, pre-drop cursor 1000
        self.assertEqual(alloc._cursors[0], 1000)

        refs, passed, events = alloc.plan_split([("new", 500)], pending_drops={0})

        self.assertEqual(refs[0].offset, 0)  # not 1000
        self.assertEqual(passed, [])
        self.assertEqual(events, [(0, 1)])

    def test_batch_fits_only_because_victim_space_freed(self):
        # Fidelity case 2: the live cursor cannot fit the key, so only the
        # victim's freed space (cursor 0) makes the batch placeable without a
        # circular pass.
        alloc = self._alloc(num_slots=1)
        for i in range(3):
            _write_legacy(alloc, f"v{i}", 1200)  # slot 0 -> live cursor 3600
        self.assertEqual(alloc._cursors[0], 3600)
        self.assertGreater(alloc._cursors[0] + 1000, _EXTENT)

        refs, passed, events = alloc.plan_split([("new", 1000)], pending_drops={0})

        self.assertEqual(passed, [])
        self.assertEqual(events, [(0, 1)])
        self.assertEqual((refs[0].extent_id, refs[0].offset), (0, 0))
        self.assertEqual(refs[0].generation, 1)
        alloc.apply_drops(events)
        self.assertTrue(alloc.commit_install(["new"], refs))
        self.assertEqual(refs[0].generation, alloc._generations[0])
        self.assertEqual(alloc.lookup("new"), refs[0])
        self.assertIsNone(alloc.lookup("v0"))

    def test_case_both_passed_and_pending_drop_bumped_once(self):
        alloc = self._alloc(num_slots=2)
        self._fill_two_full_slots(alloc)

        refs, passed, events = alloc.plan_split([("w", 1000)], pending_drops={0})

        # The victim is empty (cursor 0) for this plan, so it is placed at
        # offset 0 and is NOT in ``passed``; being already a drop_event the
        # would-be overlap still yields exactly one event and one bump.
        self.assertEqual(passed, [])
        self.assertEqual(events, [(0, 1)])  # one event, generation +1 (not +2)
        self.assertEqual((refs[0].extent_id, refs[0].offset), (0, 0))
        self.assertEqual(refs[0].generation, 1)
        alloc.apply_drops(events)
        self.assertEqual(alloc._generations[0], 1)  # no double bump
        self.assertTrue(alloc.commit_install(["w"], refs))
        self.assertEqual(refs[0].generation, alloc._generations[0])
        self.assertEqual(alloc.lookup("w"), refs[0])

    def test_unified_generation_overlap_bumps_once(self):
        # The unified NEW-1 form bumps a slot in passed AND pending_drops
        # exactly once (overlap tolerated, never double-counted).  The
        # placement fix makes that overlap unreachable from _plan_pure, so pin
        # the formula directly.
        alloc = self._alloc(num_slots=2)
        alloc._generations[0] = 4
        self.assertEqual(alloc._pending_generation(0, [0], {0}), 5)
        self.assertEqual(alloc._pending_generation(0, [0, 0], {0}), 6)

    # ---- apply_drops / commit_install atomicity ----

    def test_commit_install_does_not_bump_generations(self):
        alloc = self._alloc(num_slots=2)
        self._fill_two_full_slots(alloc)

        refs, _passed, events = alloc.plan_split([("x", 1000)], pending_drops=set())
        alloc.apply_drops(events)
        gens_after_drop = list(alloc._generations)
        self.assertTrue(alloc.commit_install(["x"], refs))
        self.assertEqual(list(alloc._generations), gens_after_drop)

    def test_apply_drops_bumps_exactly_once_and_is_idempotent(self):
        alloc = self._alloc(num_slots=2)
        _write_legacy(alloc, "a", 1000)

        alloc.apply_drops([(0, 3)])
        self.assertEqual(alloc._generations[0], 3)
        self.assertIsNone(alloc.lookup("a"))  # old keys removed
        # Replaying the same event is a no-op, not another bump.
        alloc.apply_drops([(0, 3)])
        self.assertEqual(alloc._generations[0], 3)

    def test_apply_drops_fails_closed_on_generation_decrease(self):
        alloc = self._alloc(num_slots=1)
        alloc.apply_drops([(0, 2)])
        with self.assertRaises(ValueError):
            alloc.apply_drops([(0, 1)])

    def test_commit_install_all_or_nothing_on_bad_generation(self):
        alloc = self._alloc(num_slots=2)
        refs, _passed, _events = alloc.plan_split([("a", 1000)], pending_drops=set())
        stale = ExtentRef(
            refs[0].extent_id,
            refs[0].offset,
            refs[0].length,
            refs[0].generation + 1,
        )
        self.assertFalse(alloc.commit_install(["a"], [stale]))
        self.assertIsNone(alloc.lookup("a"))
        self.assertEqual(alloc.used_bytes, 0)
        # The valid ref still installs.
        self.assertTrue(alloc.commit_install(["a"], refs))
        self.assertEqual(alloc.lookup("a"), refs[0])

    def test_commit_install_length_mismatch_installs_nothing(self):
        alloc = self._alloc(num_slots=2)
        refs, _passed, _events = alloc.plan_split([("a", 1000)], pending_drops=set())
        self.assertFalse(alloc.commit_install([], refs))
        self.assertIsNone(alloc.lookup("a"))
        self.assertEqual(alloc.used_bytes, 0)

    # ---- legacy commit behavior is preserved ----

    def test_legacy_commit_still_bumps_on_reuse(self):
        alloc = self._alloc(num_slots=1, extent_size=_EXTENT)
        first = _write_legacy(alloc, "k", 1000)
        for i in range(15):
            _write_legacy(alloc, f"f{i}", 1000)

        refs, passed = alloc.plan([("other", 1000)])
        self.assertIn(0, passed)
        alloc.commit(["other"], refs, passed)

        self.assertEqual(refs[0].generation, alloc._generations[0])
        self.assertIsNone(alloc.lookup("k"))
        self.assertEqual(alloc.lookup("other"), refs[0])
        self.assertNotEqual(first.generation, alloc._generations[0])

    def test_legacy_plan_accepts_pending_drops_on_destination(self):
        # The legacy 2-tuple return still honors the unified generation form.
        alloc = self._alloc(num_slots=1)
        _write_legacy(alloc, "old", 1000)
        refs, passed = alloc.plan([("new", 3000)], pending_drops={0})
        self.assertEqual(passed, [])
        self.assertEqual(refs[0].generation, 1)


if __name__ == "__main__":
    unittest.main()
