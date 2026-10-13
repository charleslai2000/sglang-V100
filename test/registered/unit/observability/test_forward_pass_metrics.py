from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

import msgspec

from sglang.srt.observability.forward_pass_metrics import (
    ForwardPassMetrics,
    decode,
    encode,
)

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)


def _make_ps(**overrides) -> ParallelState:
    """Build a ParallelState with reasonable defaults for tests; override fields via kwargs."""
    defaults = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_dp_rank=0,
        attn_dp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=None,
        moe_dp_size=1,
        gpu_id=0,
    )
    defaults.update(overrides)
    return ParallelState(**defaults)


class _FakeReq:
    def __init__(
        self,
        prompt_len: int,
        output_len: int = 0,
        prefix_len: int = 0,
    ):
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = list(range(output_len))
        self.prefix_indices = list(range(prefix_len))
        self.seqlen = prompt_len + output_len


class _FakeForwardMode:
    def __init__(self, *, is_mixed: bool = False, is_extend: bool = False):
        self._is_mixed = is_mixed
        self._is_extend = is_extend
        self.name = "MIXED" if is_mixed else ("EXTEND" if is_extend else "DECODE")

    def is_mixed(self):
        return self._is_mixed

    def is_extend(self, include_draft_extend_v2: bool = False):
        return self._is_extend

    def is_decode(self):
        return not self._is_mixed and not self._is_extend

    def is_prebuilt(self):
        return False


class _FakeDeviceTimer:
    def __init__(self, reporter):
        self._reporters = [reporter]
        self._completions = []
        self._next_key = 0
        self.shutdown_dropped = 0
        self._accepting = True
        self._owners = set()
        self._intervals = []

    def register_owner(self, owner):
        self._owners.add(id(owner))

    def release_owner(self, owner, timeout=1.0):
        self._owners.discard(id(owner))
        return self.shutdown(timeout) if not self._owners else None

    def shutdown(self, timeout=1.0):
        self._accepting = False
        return self.shutdown_dropped

    def add_reporter(self, reporter):
        self._reporters.append(reporter)

    def begin(self, metadata):
        if not self._accepting:
            return None
        return dict(metadata)

    def end(self, token):
        if token is not None and self._accepting:
            self._completions.append((0.001, token))

    def add_reporter(self, reporter):
        self._reporters.append(reporter)

    def complete(self, duration, **metadata):
        metadata.setdefault("timer_key", len(self._completions) + self._next_key)
        metadata.setdefault("timer_start_host_ns", self._next_key * 1_000_000)
        metadata.setdefault("timer_end_host_ns", self._next_key * 1_000_000 + int(duration * 1e9))
        self._next_key += 1
        for reporter in self._reporters:
            reporter(t=duration, **metadata)

    def _report(self):
        completions, self._completions = self._completions, []
        for duration, metadata in completions:
            self.complete(duration, **metadata)


class _CollectingPublisher:
    def __init__(self):
        self.metrics = []
        self.dropped = 0

    def publish(self, metrics):
        self.metrics.append(metrics)
        return True

    def shutdown(self):
        self.shutdown_called = True

    def stats_snapshot(self):
        return {"queue_dropped": self.dropped, "send_dropped": 0}


class _DummyPublisherThread:
    def __init__(self, endpoint: str, worker_id: str, dp_rank: int, **_: object):
        self.endpoint = endpoint
        self.worker_id = worker_id
        self.dp_rank = dp_rank

    def shutdown(self):
        pass


def _make_reporter(scheduler) -> SchedulerMetricsReporter:
    if not hasattr(scheduler, "server_args"):
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            kv_events_config=None,
            enable_mfu_metrics=False,
            enable_forward_pass_metrics=False,
        )
    if not hasattr(scheduler, "ps"):
        scheduler.ps = types.SimpleNamespace(attn_tp_rank=0, attn_cp_rank=0)
    if not hasattr(scheduler, "kv_events_publisher"):
        scheduler.kv_events_publisher = types.SimpleNamespace(
            init_kv_events=lambda *a, **kw: None,
        )
    if not hasattr(scheduler, "tp_workers"):
        scheduler.tp_workers = []
    if not hasattr(scheduler, "tp_worker"):
        scheduler.tp_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(),
        )
    if not hasattr(scheduler, "draft_worker"):
        scheduler.draft_worker = None
    context = types.SimpleNamespace(
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    return SchedulerMetricsReporter(
        scheduler=scheduler,
        tp_rank=0,
        pp_rank=0,
        dp_rank=0,
        metrics_collector_context=context,
        metrics_collector=None,
    )


class TestForwardPassMetrics(unittest.TestCase):
    def setUp(self):
        self.scheduler = types.SimpleNamespace()
        self.scheduler._fpm_worker_id = "worker-7"
        self.scheduler._fpm_dp_rank = 0
        self.scheduler._fpm_publisher = _CollectingPublisher()
        self.scheduler._fpm_uses_device_timer = False
        self.scheduler._fpm_gpu_time_acc = 0.0
        self.scheduler._fpm_completed_gpu_times = {}
        self.scheduler._fpm_pending_batches = {}
        self.scheduler._fpm_closed_batches = set()
        self.scheduler._fpm_completed_status = {}
        self.scheduler._fpm_shutting_down = False
        self.scheduler.waiting_queue = []
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.reporter = _make_reporter(self.scheduler)
        self.scheduler.enable_fpm = True

    def _make_batch(self, **overrides):
        defaults = dict(
            forward_mode=_FakeForwardMode(),
            reqs=[],
            decoding_reqs=[],
            prefill_stats=None,
            seq_lens_cpu=[],
            fpm_start_time=100.0,
            forward_iter=1,
            batch_size=lambda: len(overrides.get("reqs", [])),
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_emit_mixed_batch_separates_prefill_and_decode(self):
        self.scheduler._fpm_dp_rank = 3
        self.scheduler.waiting_queue = [_FakeReq(6), _FakeReq(4, output_len=2)]

        prefill_a = _FakeReq(10, prefix_len=2)
        prefill_b = _FakeReq(14, prefix_len=3)
        decode_req = _FakeReq(8, output_len=3)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_mixed=True, is_extend=True),
            reqs=[prefill_a, prefill_b, decode_req],
            decoding_reqs=[decode_req],
            prefill_stats=PrefillStats(
                log_input_tokens=12,
                log_hit_tokens=5,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=2,
            ),
            seq_lens_cpu=[decode_req.seqlen],
        )

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=104.5,
        ):
            self.reporter._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.worker_id, "worker-7")
        self.assertEqual(metrics.dp_rank, 3)
        self.assertEqual(metrics.wall_time, 4.5)
        self.assertEqual(metrics.scheduled_requests.num_prefill_requests, 2)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_tokens, 12)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_kv_tokens, 5)
        self.assertEqual(metrics.scheduled_requests.num_decode_requests, 1)
        self.assertEqual(
            metrics.scheduled_requests.sum_decode_kv_tokens, decode_req.seqlen
        )
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 1)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 1)

    def test_emit_uses_device_timer_gpu_time(self):
        self.scheduler._fpm_uses_device_timer = True
        self.scheduler._fpm_completed_gpu_times = {1: [{"duration_s": 0.042, "metadata": {"forward_iter": 1, "timer_key": 1, "timer_start_host_ns": 1000000000, "timer_end_host_ns": 1042000000}, "timer_key": 1}]}
        self.reporter.forward_pass_device_timer = _FakeDeviceTimer(lambda **_: None)
        batch = self._make_batch()
        self.scheduler._fpm_pending_batches[1] = {
            "batch_id": id(batch),
            "mode": "DECODE",
            "batch_size": 0,
            "scheduled": self.reporter._build_scheduled_request_metrics(batch),
            "queued": self.reporter._build_queued_request_metrics(),
            "gpu_intervals": [],
            "audit_events": [],
            "timer_keys": set(),
            "start_time": 99.9,
            "scheduler_start_time": 99.9,
            "forward_start_time": 99.9,
            "scheduler_end_time": 100.0,
        }

        self.reporter._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertGreaterEqual(self.scheduler._fpm_publisher.metrics[0].wall_time, 0.0)
        self.assertAlmostEqual(
            self.scheduler._fpm_pending_batches.get(1, {}).get("gpu_interval_sum_s", 0.042),
            0.042,
        )

    def test_emit_defers_device_timer_batch_until_its_identity_completes(self):
        self.scheduler._fpm_uses_device_timer = True
        self.scheduler._fpm_completed_gpu_times = {}
        self.scheduler._fpm_pending_batches = {}
        def completion_reporter(t, **metadata):
            self.scheduler._fpm_completed_gpu_times.setdefault(
                metadata["forward_iter"], []
            ).append({"duration_s": t, "metadata": dict(metadata), "timer_key": metadata.get("timer_key", t)})

        self.reporter.forward_pass_device_timer = _FakeDeviceTimer(completion_reporter)
        req = _FakeReq(77)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_extend=True),
            reqs=[req],
            prefill_stats=PrefillStats(
                log_input_tokens=77,
                log_hit_tokens=0,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=1,
            ),
        )

        self.reporter._emit_forward_pass_metrics(batch)
        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 0)
        self.assertIn(1, self.scheduler._fpm_pending_batches)

        # A completion for a later ID must not be assigned to this batch.
        later = self._make_batch(forward_iter=2)
        self.reporter.forward_pass_device_timer._completions.append(
            (0.025, {"forward_iter": 2, "category": "decode"})
        )
        self.reporter._emit_forward_pass_metrics(later)
        # ID 2's event completed before its snapshot was registered. It must
        # join only ID 2's current snapshot; ID 1 remains pending.
        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertIn(1, self.scheduler._fpm_pending_batches)
        self.assertEqual(len(self.scheduler._fpm_completed_gpu_times.get(1, [])), 0)
        self.assertEqual(len(self.scheduler._fpm_pending_batches[1]["audit_events"]), 0)

        self.reporter.forward_pass_device_timer._completions.append(
            (0.041, {"forward_iter": 1, "category": "extend"})
        )
        self.reporter._emit_forward_pass_metrics(batch)
        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 2)
        metrics = self.scheduler._fpm_publisher.metrics[1]
        self.assertGreaterEqual(metrics.wall_time, 0.0)
        self.assertEqual(metrics.scheduled_requests.num_prefill_requests, 1)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_tokens, 77)
        self.assertNotIn(2, self.scheduler._fpm_pending_batches)
        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 2)

    def test_emit_uses_monotonic_without_device_timer(self):
        batch = self._make_batch()

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=100.035,
        ):
            self.reporter._emit_forward_pass_metrics(batch, result=None)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertAlmostEqual(
            self.scheduler._fpm_publisher.metrics[0].wall_time, 0.035, places=4
        )

    def test_timer_completion_callbacks_are_joined_once_and_shutdown_cleans_pending(self):
        self.scheduler._fpm_uses_device_timer = True
        req = _FakeReq(41)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_extend=True),
            reqs=[req],
            prefill_stats=PrefillStats(
                log_input_tokens=41,
                log_hit_tokens=0,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=1,
            ),
        )
        def completion_reporter(t, **metadata):
            self.scheduler._fpm_completed_gpu_times.setdefault(
                metadata["forward_iter"], []
            ).append({"duration_s": t, "metadata": dict(metadata), "timer_key": metadata.get("timer_key", t)})

        self.reporter.forward_pass_device_timer = _FakeDeviceTimer(completion_reporter)
        self.reporter._emit_forward_pass_metrics(batch)
        pending = self.scheduler._fpm_pending_batches[1]
        timer = self.reporter.forward_pass_device_timer
        timer._completions.extend(
            [
                (0.0, {"forward_iter": 1, "category": "extend"}),
                (0.01, {"forward_iter": 1, "category": "extend"}),
                (0.01, {"forward_iter": 1, "category": "extend"}),
            ]
        )
        self.reporter._emit_forward_pass_metrics(batch)
        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 0)
        self.assertEqual(self.scheduler._fpm_completed_status[1]["status"], "INVALID")
        self.assertEqual(
            self.scheduler._fpm_completed_status[1]["reason"],
            "multiple_gpu_intervals_without_device_elapsed",
        )
        self.assertAlmostEqual(
            self.scheduler._fpm_completed_status[1]["gpu_interval_sum_s"], 0.02
        )
        self.assertNotIn(1, self.scheduler._fpm_pending_batches)

        self.scheduler._fpm_pending_batches[2] = pending
        self.scheduler._fpm_completed_gpu_times[2] = [{"duration_s": 0.03, "metadata": {"forward_iter": 2, "timer_key": 2}, "timer_key": 2}]
        self.reporter.forward_pass_device_timer.shutdown_dropped = 2
        self.reporter._shutdown_fpm()
        self.assertEqual(self.scheduler._fpm_pending_batches, {})
        self.assertEqual(self.scheduler._fpm_completed_gpu_times, {})
        self.assertTrue(self.scheduler._fpm_shutting_down)
        self.assertFalse(self.reporter.forward_pass_device_timer._accepting)
        self.assertTrue(self.scheduler._fpm_publisher.shutdown_called)
        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 0)

    def test_disagg_prefill_queued_metrics(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        self.scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
            queue=[_FakeReq(100), _FakeReq(200), _FakeReq(50)],
        )
        batch = self._make_batch()

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=101.0,
        ):
            self.reporter._emit_forward_pass_metrics(batch)

        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_prefill_tokens, 350)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 0)

    def test_disagg_decode_queued_metrics(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.DECODE
        self.scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[_FakeReq(10, output_len=5), _FakeReq(20, output_len=10)],
        )
        self.scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[_FakeReq(30, output_len=15)],
        )
        batch = self._make_batch()

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=101.0,
        ):
            self.reporter._emit_forward_pass_metrics(batch)

        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 0)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_decode_kv_tokens, 15 + 30 + 45)

    def test_init_metrics_uses_server_worker_id(self):
        scheduler = types.SimpleNamespace()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name=None,
            kv_events_config=None,
        )
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=2, pp_rank=0, pp_size=1)
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            reporter = _make_reporter(scheduler)

        self.assertTrue(scheduler.enable_fpm)
        self.assertEqual(scheduler._fpm_worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_dp_rank, 2)
        self.assertEqual(scheduler._fpm_publisher.worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_publisher.dp_rank, 2)
        self.assertTrue(scheduler._fpm_publisher.endpoint.startswith("ipc://"))
        self.assertIsNotNone(scheduler.server_args.forward_pass_metrics_ipc_name)

    def test_forward_iter_is_assigned_once_by_run_batch(self):
        scheduler = types.SimpleNamespace(
            forward_ct=7,
            enable_fpm=False,
            metrics_reporter=None,
        )
        batch = self._make_batch()
        # Mirror Scheduler.run_batch's allocation at its single authority.
        scheduler.forward_ct += 1
        batch.forward_iter = scheduler.forward_ct
        scheduler.forward_ct += 1
        later = self._make_batch()
        later.forward_iter = scheduler.forward_ct
        self.assertEqual(batch.forward_iter, 8)
        self.assertEqual(later.forward_iter, 9)
        self.assertNotEqual(batch.forward_iter, later.forward_iter)

    def test_publisher_queue_drop_is_counted_and_reported(self):
        from sglang.srt.observability.forward_pass_metrics import _FpmPublisherThread

        publisher = object.__new__(_FpmPublisherThread)
        publisher._queue = __import__("queue").Queue(maxsize=1)
        publisher._queue.put_nowait(None)
        publisher._running = True
        publisher._stats_lock = __import__("threading").Lock()
        publisher._stats = {
            "published_enqueued": 0,
            "queue_dropped": 0,
            "sent": 0,
            "send_dropped": 0,
            "send_failed": 0,
        }
        metrics = ForwardPassMetrics(worker_id="test")
        self.assertFalse(publisher.publish(metrics))
        self.assertEqual(publisher.stats_snapshot()["queue_dropped"], 1)
        self.assertEqual(publisher.stats_snapshot()["send_failed"], 0)

    def test_fpm_schema_round_trip_and_rank_endpoint_suffix(self):
        metrics = ForwardPassMetrics(
            worker_id="worker-7",
            dp_rank=3,
            wall_time=0.125,
        )
        decoded = decode(encode(metrics))
        self.assertEqual(decoded, metrics)
        self.assertEqual(metrics.version, 1)
        self.assertEqual(
            list(msgspec.structs.fields(ForwardPassMetrics)[0].name for _ in [0]),
            ["version"],
        )

        scheduler = types.SimpleNamespace()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name="ipc:///tmp/fpm-test",
            kv_events_config=None,
        )
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=3, pp_rank=0, pp_size=1)
        scheduler.enable_kv_cache_events = False
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                False,
            ),
            patch(
                "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
                _DummyPublisherThread,
            ),
        ):
            _make_reporter(scheduler)
        self.assertEqual(scheduler._fpm_publisher.endpoint, "ipc:///tmp/fpm-test.3")

    def test_init_fallback_timer_is_installed_on_runner(self):
        scheduler = types.SimpleNamespace()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name=None,
            kv_events_config=None,
        )
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=0, pp_rank=0, pp_size=1)
        scheduler.enable_kv_cache_events = False
        scheduler.tp_workers = []
        scheduler.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace())
        scheduler.draft_worker = None

        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                False,
            ),
            patch(
                "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
                _DummyPublisherThread,
            ),
        ):
            reporter = _make_reporter(scheduler)

        self.assertTrue(scheduler.enable_fpm)
        self.assertIs(scheduler.tp_worker.model_runner.device_timer, reporter.forward_pass_device_timer)

    def test_init_fpm_disabled_on_non_last_pp_rank(self):
        scheduler = types.SimpleNamespace()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name=None,
            kv_events_config=None,
        )
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=0, pp_rank=0, pp_size=2)
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            reporter = _make_reporter(scheduler)

        self.assertFalse(scheduler.enable_fpm)


if __name__ == "__main__":
    unittest.main()
