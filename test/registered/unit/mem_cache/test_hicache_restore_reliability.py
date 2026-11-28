"""HiCache restore reliability: staging crop, anchor identity, ack drain, liveness.

Three field defects from the E050/E051 campaigns are pinned here:

* **Capacity crop** -- a demand restore whose host staging allocation fails while
  the requested span is much larger than the free pool used to be revoked whole
  after the L3 read had already been performed. The rebuild stages the prefix
  that fits, but must never shorten a validated ``TRAILING_PAGES`` frontier
  (SWA / Mamba state windows are validated as tails, not as page-aligned spans).
* **Anchor identity** -- an async prefetch records its anchor's last page hash at
  issue; the hash chain covers every ancestor, so an equal value at commit proves
  the NodeId still resolves to a node holding that exact prefix. A deleted or
  reused anchor discards the fetch as a terminal miss without double-freeing.
* **``wait_complete`` liveness** -- the policy means "do not cut an active read",
  not "wait forever": a completion that stops arriving must be cut and retried
  instead of wedging admission, while a slow but progressing transfer is never cut.

Pure CPU, no CUDA, no server. Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_restore_reliability.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    staging_crop_length,
    storage_hit_crop_is_safe,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class _Transfer:
    def __init__(self, hit_policy=None):
        self.hit_policy = hit_policy


class _Operation:
    def __init__(self, storage_hit_count=0, pool_transfers=None, completed=0):
        self.storage_hit_count = storage_hit_count
        self.pool_transfers = pool_transfers
        self.completed_tokens = completed
        self.pool_transfers_done = False
        self.storage_start = 0
        self.stats_requested_tokens = storage_hit_count


class TestStagingCropPolicy(unittest.TestCase):
    def test_no_crop_without_free_units(self):
        self.assertEqual(staging_crop_length(_Operation(4864), 4864, 0, 1, 256), 0)

    def test_no_crop_when_the_crop_cannot_pay_for_itself(self):
        # 512 of 4864 is below the 1/8 floor.
        self.assertEqual(staging_crop_length(_Operation(4864), 4864, 512, 1, 256), 0)

    def test_min_tokens_is_a_floor_too(self):
        # 1024 clears 1/8 but not the configured prefetch threshold.
        self.assertEqual(
            staging_crop_length(_Operation(8192), 8192, 1024, 1, 2048), 0
        )

    def test_crops_to_the_free_prefix(self):
        self.assertEqual(
            staging_crop_length(_Operation(4864), 4864, 1024, 1, 256), 1024
        )

    def test_crop_is_page_aligned(self):
        self.assertEqual(
            staging_crop_length(_Operation(4864), 4864, 1000, 64, 256), 960
        )

    def test_full_grant_is_not_reported_as_a_crop(self):
        self.assertEqual(
            staging_crop_length(_Operation(4864), 4864, 8192, 1, 256), 4864
        )

    def test_trailing_pages_frontier_is_never_shortened(self):
        op = _Operation(4864, [_Transfer(PoolHitPolicy.TRAILING_PAGES)])
        self.assertEqual(staging_crop_length(op, 4864, 1024, 1, 256), 0)
        self.assertFalse(storage_hit_crop_is_safe(op, 1024))

    def test_all_pages_transfer_may_be_cropped(self):
        op = _Operation(4864, [_Transfer(PoolHitPolicy.ALL_PAGES)])
        self.assertEqual(staging_crop_length(op, 4864, 1024, 1, 256), 1024)
        self.assertTrue(storage_hit_crop_is_safe(op, 1024))

    def test_full_grant_is_always_safe(self):
        op = _Operation(4864, [_Transfer(PoolHitPolicy.TRAILING_PAGES)])
        self.assertTrue(storage_hit_crop_is_safe(op, 4864))

    def test_kv_derived_pool_blocks_the_crop(self):
        transfer = _Transfer(PoolHitPolicy.ALL_PAGES)
        transfer.indices_from_pool = "kv"
        op = _Operation(4864, [transfer])
        self.assertFalse(storage_hit_crop_is_safe(op, 1024))

    def test_trimmed_full_head_blocks_the_crop(self):
        op = _Operation(4864)
        op.sidecar_hash_values = ["h1", "h2", "h3"]
        self.assertFalse(storage_hit_crop_is_safe(op, 1024))


class TestStagingDeferral(unittest.TestCase):
    """A staging shortfall is re-measured on the next pass, then revoked."""

    def _cache(self, budget=2):
        import queue

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.prefetch_staging_deferrals = budget
        cache._prefetch_outcome_stats = {"staging_deferred": 0}
        cache.cache_controller = types.SimpleNamespace(
            prefetch_hit_queue=queue.Queue()
        )
        cache.defer = types.MethodType(
            UnifiedRadixCache._defer_staging_attempt, cache
        )
        return cache

    def test_requeues_within_the_budget_then_revokes(self):
        cache = self._cache(budget=2)
        op = _Operation(4096)
        self.assertTrue(cache.defer(op))
        self.assertTrue(cache.defer(op))
        self.assertFalse(cache.defer(op))
        self.assertEqual(op.staging_deferrals, 2)
        self.assertEqual(cache._prefetch_outcome_stats["staging_deferred"], 2)
        # The same operation is re-queued, not re-issued with fresh state.
        self.assertEqual(cache.cache_controller.prefetch_hit_queue.qsize(), 2)

    def test_zero_budget_revokes_immediately(self):
        cache = self._cache(budget=0)
        op = _Operation(4096)
        self.assertFalse(cache.defer(op))
        self.assertEqual(cache.cache_controller.prefetch_hit_queue.qsize(), 0)


# ---- anchor identity -------------------------------------------------------


class _FakeComponent:
    """The slice of a component the node-creation path touches."""

    def __init__(self, component_type):
        self.component_type = component_type

    def reclaimable_value_len(self, node):
        return 0

    def __getattr__(self, name):
        return mock.Mock()


def _make_tree():
    """A real UnifiedTreeCore with one FULL component, HiCache enabled."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    base = ComponentType.FULL
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_components=(base,),
    )
    tree = UnifiedTreeCore(params, {base: _FakeComponent(base)})
    tree.set_hicache_enabled()
    tree.enable_storage = True
    return tree, base


def _add_node(tree, parent, tokens, slots, hashes=None):
    from sglang.srt.mem_cache.radix_cache import RadixKey

    node = tree._add_new_node(
        parent, RadixKey(list(tokens)), torch.tensor(list(slots), dtype=torch.int64)
    )
    if hashes is not None:
        node.hash_value = list(hashes)
    return node


def _make_cache(tree):
    """A cache facade with only the state the identity/discard path owns."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = tree
    cache.page_size = tree.page_size
    cache.disable = False
    cache.host_memory_mode = "cache"
    cache.ongoing_prefetch = {}
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache._storage_prefetch_wait = {}
    cache.buffer_pipeline = None
    cache.cache_controller = types.SimpleNamespace(
        prefetch_tokens_occupied=0,
        append_host_mem_release=mock.Mock(),
    )
    cache.storage_prefetch_retries = mock.Mock()
    cache.storage_metrics_collector = mock.Mock()
    cache.enable_storage_metrics = False
    cache._init_params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=tree.page_size,
        tree_components=tuple(tree.components),
    )
    return cache


class TestPrefetchAnchorIdentity(unittest.TestCase):
    def setUp(self):
        self.tree, self.base = _make_tree()
        self.cache = _make_cache(self.tree)

    def test_evicted_anchor_is_detected(self):
        anchor = _add_node(
            self.tree, self.tree.root_node, [1, 2], [1, 2], hashes=["h1", "h2"]
        )
        seen = self.tree.get_last_hash_value(anchor.id)
        self.assertTrue(self.cache._prefetch_anchor_is_valid(anchor.id, seen))

        self.tree._remove_leaf_from_parent(anchor)

        self.assertFalse(self.tree.node_exists(anchor.id))
        self.assertFalse(self.cache._prefetch_anchor_is_valid(anchor.id, seen))

    def test_reused_anchor_id_is_not_the_same_prefix(self):
        anchor = _add_node(
            self.tree, self.tree.root_node, [1, 2], [1, 2], hashes=["h1", "h2"]
        )
        seen = self.tree.get_last_hash_value(anchor.id)
        anchor.hash_value = ["h1", "different"]
        self.assertFalse(self.cache._prefetch_anchor_is_valid(anchor.id, seen))

    def test_unhashed_anchor_falls_back_to_existence(self):
        # No chain at issue means no comparison is possible; the node still
        # resolving is the available proof, and the fetch must not be thrown
        # away on that account.
        anchor = _add_node(self.tree, self.tree.root_node, [1], [1])
        self.assertTrue(self.cache._prefetch_anchor_is_valid(anchor.id, None))
        self.tree._remove_leaf_from_parent(anchor)
        self.assertFalse(self.cache._prefetch_anchor_is_valid(anchor.id, None))

    def test_a_hashed_anchor_that_became_the_root_is_rejected(self):
        # Only the root is legitimately unhashed; a node that carried a chain
        # at issue cannot turn into the root and still be the same target.
        anchor = _add_node(self.tree, self.tree.root_node, [1], [1], hashes=["h1"])
        self.cache.tree_core.is_root = lambda node_id: True
        self.assertFalse(self.cache._prefetch_anchor_is_valid(anchor.id, "h1"))

    def test_a_split_keeps_identity(self):
        anchor = _add_node(
            self.tree,
            self.tree.root_node,
            [1, 2, 3],
            [1, 2, 3],
            hashes=["h1", "h2", "h3"],
        )
        seen = self.tree.get_last_hash_value(anchor.id)
        # A split moves the earlier pages to the new parent and leaves the
        # anchor holding only its last page, so the last hash is unchanged.
        anchor.hash_value = ["h3"]
        self.assertTrue(self.cache._prefetch_anchor_is_valid(anchor.id, seen))


class TestStaleAnchorDiscard(unittest.TestCase):
    def setUp(self):
        self.tree, self.base = _make_tree()
        self.cache = _make_cache(self.tree)

    def _entry(self, anchor_id, host_indices, op, lock_params, anchor_hash):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_radix_cache import _OngoingPrefetch

        return _OngoingPrefetch(
            anchor_id,
            RadixKey([1, 2]),
            host_indices,
            op,
            lock_params,
            {},
            anchor_hash,
        )

    def test_discard_releases_operation_staging_once(self):
        anchor = _add_node(
            self.tree, self.tree.root_node, [1, 2], [1, 2], hashes=["h1", "h2"]
        )
        host_indices = torch.tensor([900, 901], dtype=torch.int64)
        request = mock.Mock(rid="r1")
        op = _Operation(2, completed=2)
        op.host_indices = host_indices
        self.cache.ongoing_prefetch[request] = self._entry(
            anchor.id, host_indices, op, None, "h2"
        )
        self.cache.dec_host_lock_ref = mock.Mock()
        self.cache._retire_ongoing_prefetch = types.MethodType(
            UnifiedRadixCache._retire_ongoing_prefetch, self.cache
        )

        self.cache._discard_prefetch_stale_anchor(request, op, 2, host_indices, {})
        releases = self.cache.cache_controller.append_host_mem_release.call_args_list
        self.assertEqual(len(releases), 1)
        self.assertEqual(list(releases[0].kwargs["host_indices"]), [900, 901])
        # No anchor lock was held, so nothing is returned.
        self.cache.dec_host_lock_ref.assert_not_called()
        self.assertNotIn(request, self.cache.ongoing_prefetch)

        # A repeated terminal path must not release the same staging twice.
        self.cache._discard_prefetch_stale_anchor(request, op, 2, host_indices, {})
        self.assertEqual(
            len(self.cache.cache_controller.append_host_mem_release.call_args_list), 1
        )

    def test_discard_does_not_touch_a_deleted_anchor_lock(self):
        anchor = _add_node(
            self.tree, self.tree.root_node, [1, 2], [1, 2], hashes=["h1", "h2"]
        )
        host_indices = torch.tensor([900, 901], dtype=torch.int64)
        request = mock.Mock(rid="r2")
        op = _Operation(2, completed=2)
        op.host_indices = host_indices
        lock_params = self.cache.tree_core.inc_host_lock_ref(anchor.id).to_dec_params()
        self.cache.ongoing_prefetch[request] = self._entry(
            anchor.id, host_indices, op, lock_params, "h2"
        )
        self.cache._retire_ongoing_prefetch = types.MethodType(
            UnifiedRadixCache._retire_ongoing_prefetch, self.cache
        )
        self.tree._remove_leaf_from_parent(anchor)

        self.cache._discard_prefetch_stale_anchor(request, op, 2, host_indices, {})
        # No KeyError from a dead NodeId, and the staging still came back.
        self.assertEqual(
            len(self.cache.cache_controller.append_host_mem_release.call_args_list), 1
        )


# ---- extent reclaim -> existence beliefs ---------------------------------


class TestExtentReclaimInvalidation(unittest.TestCase):
    """A backend-side reclaim must clear the beliefs that would skip a rewrite."""

    drain = UnifiedRadixCache._drain_reclaimed_storage_keys

    def _cache(self, backend, stats=None):
        from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
            StorageExistenceCache,
        )

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = types.SimpleNamespace(storage_backend=backend)
        cache.storage_existence_cache = StorageExistenceCache()
        cache._prefetch_outcome_stats = stats if stats is not None else {
            "storage_reclaim_invalidated": 0
        }
        cache.drain = types.MethodType(
            UnifiedRadixCache._drain_reclaimed_storage_keys, cache
        )
        return cache

    def test_reclaimed_keys_drop_beliefs_in_every_pool(self):
        beliefs_cache = self._cache(None)
        beliefs_cache.storage_existence_cache.pool("kv").add(["h1", "h2"])
        beliefs_cache.storage_existence_cache.pool("swa").add(["h2"])

        backend = types.SimpleNamespace(
            config_suffix="_m",
            drain_evicted_storage_keys=lambda: ["h1_m", "h2.swa_m"],
        )
        cache = self._cache(
            backend, {"storage_reclaim_invalidated": 0}
        )
        cache.storage_existence_cache = beliefs_cache.storage_existence_cache
        cache.drain()

        # h1 was KV-only, h2 was KV+SWA: both beliefs are gone everywhere.
        self.assertFalse(cache.storage_existence_cache.pool("kv").contains("h1"))
        self.assertFalse(cache.storage_existence_cache.pool("kv").contains("h2"))
        self.assertFalse(cache.storage_existence_cache.pool("swa").contains("h2"))
        self.assertEqual(cache._prefetch_outcome_stats["storage_reclaim_invalidated"], 3)

    def test_backend_without_the_hook_is_a_noop(self):
        cache = self._cache(types.SimpleNamespace())
        cache.drain()
        self.assertEqual(cache._prefetch_outcome_stats["storage_reclaim_invalidated"], 0)

    def test_empty_drain_does_nothing(self):
        backend = types.SimpleNamespace(
            config_suffix="", drain_evicted_storage_keys=lambda: []
        )
        cache = self._cache(backend)
        cache.storage_existence_cache.pool("kv").add(["h1"])
        cache.drain()
        self.assertTrue(cache.storage_existence_cache.pool("kv").contains("h1"))


# ---- wait_complete liveness ------------------------------------------------


class _LivenessShim:
    """Drives the real ``wait_complete`` stall detector without a process group."""

    pp_rank = 0
    prefetch_stop_policy = "wait_complete"
    prefetch_timeout_base = 0.05
    prefetch_timeout_per_page = 0.0
    page_size = 16

    _start_storage_prefetch_wait = UnifiedRadixCache._start_storage_prefetch_wait
    _prefetch_progress_watermark = staticmethod(
        UnifiedRadixCache._prefetch_progress_watermark
    )
    _wait_complete_stalled = UnifiedRadixCache._wait_complete_stalled
    _can_terminate_prefetch = UnifiedRadixCache._can_terminate_prefetch
    _mark_prefetch_stalled = UnifiedRadixCache._mark_prefetch_stalled

    def __init__(self):
        self._storage_prefetch_wait = {}
        self._prefetch_outcome_stats = {"wait_complete_stalled": 0}

    def _all_reduce(self, tensor, op):
        return None

    @staticmethod
    def _logger():
        return mock.patch("sglang.srt.mem_cache.unified_radix_cache.logger")


class TestWaitCompleteLiveness(unittest.TestCase):
    def _op(self, completed=0):
        op = _Operation(4096, completed=completed)
        op.handle = mock.Mock(rid="lc")
        op.stats_requested_tokens = 4096
        op.deadline_expired = False
        return op

    def test_a_progressing_transfer_is_never_cut(self):
        shim = _LivenessShim()
        op = self._op()
        self.assertFalse(shim._can_terminate_prefetch(op))
        for completed in (16, 32, 48, 64):
            op.completed_tokens = completed
            shim._start_storage_prefetch_wait(op)
            self.assertFalse(shim._can_terminate_prefetch(op))

    def test_a_stalled_completion_is_cut(self):
        shim = _LivenessShim()
        op = self._op()
        self.assertFalse(shim._can_terminate_prefetch(op))
        op.handle = op.handle
        shim._start_storage_prefetch_wait(op)
        # No ACK advances the watermark: the budget expires.
        deadline = 2.0
        import time

        started = time.monotonic()
        while time.monotonic() - started < deadline:
            if shim._can_terminate_prefetch(op):
                break
            time.sleep(0.01)
        self.assertTrue(shim._can_terminate_prefetch(op))
        with _LivenessShim._logger():
            shim._mark_prefetch_stalled(op)
        self.assertTrue(op.prefetch_stalled)
        self.assertEqual(shim._prefetch_outcome_stats["wait_complete_stalled"], 1)

    def test_a_finished_sidecar_read_registers_progress(self):
        shim = _LivenessShim()
        op = self._op()
        shim._start_storage_prefetch_wait(op)
        op.pool_transfers_done = True
        self.assertFalse(shim._can_terminate_prefetch(op))


if __name__ == "__main__":
    unittest.main()
