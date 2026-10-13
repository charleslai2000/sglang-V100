from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
)

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
    SchedulerStats,
    compute_routing_key_stats,
)
from sglang.srt.utils.device_timer import DeviceTimer
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import EmbeddingBatchResult


logger = logging.getLogger(__name__)


RECORD_STEP_TIME = envs.SGLANG_RECORD_STEP_TIME.get()
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()
ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


def _decode_total_seq_lens(batch: ScheduleBatch) -> int:
    """Sync-free sum of seq_lens for decode metrics."""
    if batch.seq_lens_cpu is not None:
        return int(batch.seq_lens_cpu.sum().item())
    return sum(req.seqlen for req in batch.reqs)


@dataclasses.dataclass
class PrefillStats:
    """Stats for logging prefill batch metrics."""

    log_input_tokens: int
    log_hit_tokens: int
    new_token_ratio: float
    num_running_reqs: QueueCount
    num_new_seqs: int  # len(can_run_list)
    num_pending_tokens: int = 0

    @classmethod
    def from_adder(
        cls,
        adder: PrefillAdder,
        running_reqs: List[Req],
        enable_priority_scheduling: bool = False,
        num_pending_tokens: int = 0,
    ):
        return cls(
            log_input_tokens=adder.log_input_tokens,
            log_hit_tokens=adder.log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            num_running_reqs=QueueCount.from_reqs(
                running_reqs, enable_priority_scheduling
            ),
            num_new_seqs=len(adder.can_run_list),
            num_pending_tokens=num_pending_tokens,
        )


@dataclass(kw_only=True)
class SchedulerMetricsReporter:
    scheduler: "Scheduler"
    tp_rank: int
    pp_rank: int
    dp_rank: Optional[int]
    metrics_collector_context: SchedulerMetricsCollectorContext
    metrics_collector: Optional[SchedulerMetricsCollector]
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0

    def __post_init__(self) -> None:
        self.enable_metrics = self.metrics_collector_context.enable_metrics
        self.is_stats_logging_rank = (
            self.metrics_collector_context.is_stats_logging_rank
        )
        self.current_scheduler_metrics_enabled = (
            self.metrics_collector_context.current_scheduler_metrics_enabled
        )
        self.enable_kv_cache_events = (
            self.metrics_collector_context.enable_kv_cache_events
        )
        self._init_metrics(self.tp_rank, self.pp_rank, self.dp_rank)
        self._install_device_timer_on_runners()

    def _init_metrics(
        self,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
    ):
        # Basic stats
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.stats = SchedulerStats()
        self._graph_backend_label = {
            "cpu": "cpu graph",
            "npu": "npu graph",
            "musa": "musa graph",
        }.get(getattr(self.scheduler, "device", ""), "cuda graph")

        # Cumulative spec-decoding counters (reset every decode_log_interval).
        # Each update adds (num_correct_drafts + bs, bs).
        # `*_accept_tokens` = drafts + bonus; `*_correct_drafts` = drafts-only.
        self.spec_num_accept_tokens = 0  # per-log-interval
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0  # lifetime
        self.spec_total_num_forward_ct = 0

        # For PD disaggregation
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        self.enable_mfu_metrics = False

        if self.enable_metrics:
            self.enable_mfu_metrics = self.scheduler.server_args.enable_mfu_metrics
            if self.enable_mfu_metrics:
                self._init_estimated_perf_constants()
                self._mfu_log_flops = 0.0
                self._mfu_log_read_bytes = 0.0
                self._mfu_log_write_bytes = 0.0

        self.fwd_occupancy = float("nan")

        self.forward_pass_device_timer: Optional[DeviceTimer] = None
        self._metrics_timer_owner = None
        self._fpm_timer_owner = None

        if ENABLE_METRICS_DEVICE_TIMER:
            self._device_timer_window_batch_count = 0
            self._device_timer_window_gpu_time = 0.0
            self._device_timer_window_start = None

            def _wrap_execution_reporter(**kwargs):
                self._device_timer_window_gpu_time += kwargs["t"]
                if self.enable_metrics:
                    self.metrics_collector.increment_forward_execution_seconds(**kwargs)

            self.forward_pass_device_timer = DeviceTimer(
                reporter=_wrap_execution_reporter,
            )
            self._metrics_timer_owner = object()
            self.forward_pass_device_timer.register_owner(self._metrics_timer_owner)

        self._init_fpm()

        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create(
            enable_metrics=self.enable_metrics
        )

    def _install_device_timer_on_runners(self):
        if self.forward_pass_device_timer is None:
            return
        timer = self.forward_pass_device_timer
        self.scheduler.tp_worker.model_runner.device_timer = timer
        if self.scheduler.draft_worker is not None:
            dw = getattr(self.scheduler.draft_worker, "draft_worker", None)
            if dw is not None:
                if hasattr(dw, "draft_runner"):
                    dw.draft_runner.device_timer = timer
                for r in getattr(dw, "draft_runner_list", []):
                    r.device_timer = timer

    def _init_fpm(self):
        """Initialize Forward Pass Metrics (FPM) publisher if configured."""
        self.scheduler.enable_fpm = False
        if (
            self.scheduler.server_args.enable_forward_pass_metrics
            and self.scheduler.ps.attn_tp_rank == 0
            and self.scheduler.ps.pp_rank == self.scheduler.ps.pp_size - 1
        ):
            from sglang.srt.observability.forward_pass_metrics import (
                _FpmPublisherThread,
            )

            self.scheduler._fpm_dp_rank = (
                self.scheduler.ps.dp_rank
                if self.scheduler.ps.dp_rank is not None
                else 0
            )
            self.scheduler._fpm_worker_id = (
                self.scheduler.server_args.forward_pass_metrics_worker_id
            )
            base_endpoint = self.scheduler.server_args.forward_pass_metrics_ipc_name
            if base_endpoint is None:
                ipc_path = tempfile.NamedTemporaryFile(delete=False).name
                base_endpoint = f"ipc://{ipc_path}"
                self.scheduler.server_args.forward_pass_metrics_ipc_name = base_endpoint
            endpoint = f"{base_endpoint}.{self.scheduler._fpm_dp_rank}"
            self.scheduler._fpm_publisher = _FpmPublisherThread(
                endpoint,
                worker_id=self.scheduler._fpm_worker_id,
                dp_rank=self.scheduler._fpm_dp_rank,
            )
            self.scheduler._fpm_gpu_time_acc = 0.0
            self.scheduler._fpm_completed_gpu_times = {}
            self.scheduler._fpm_pending_batches = {}
            self.scheduler._fpm_closed_batches = set()
            self.scheduler._fpm_completed_status = {}
            self.scheduler._fpm_completed_timer_keys = set()
            self.scheduler._fpm_shutting_down = False
            self.scheduler._fpm_audit_counters = {
                "forward_completed": 0,
                "scheduler_forward_completed": 0,
                "timer_completed": 0,
                "active_emit_attempted": 0,
                "queue_accepted": 0,
                "queue_dropped": 0,
                "timer_duplicate": 0,
                "unmatched_timer": 0,
                "pending_timeout": 0,
            }
            # Optional timeline sidecar state is independent of FPM correctness.
            self.scheduler._fpm_timeline_pending = {}

            def _fpm_device_timer_reporter(t, **_kwargs):
                forward_iter = _kwargs.get("forward_iter")
                if forward_iter is None or self.scheduler._fpm_shutting_down:
                    return
                self.scheduler._fpm_audit_counters["timer_completed"] += 1
                timer_key = _kwargs.get("timer_key")
                if timer_key is None:
                    logger.error("FPM timer completion missing timer_key: %s", _kwargs)
                    return
                if timer_key in self.scheduler._fpm_completed_timer_keys:
                    self.scheduler._fpm_audit_counters["timer_duplicate"] += 1
                    logger.error("Duplicate FPM timer callback key=%s", timer_key)
                    return
                self.scheduler._fpm_completed_timer_keys.add(timer_key)
                pending = self.scheduler._fpm_pending_batches.get(forward_iter)
                if pending is not None:
                    interval_for_join = {"duration_s": t, "metadata": dict(_kwargs), "timer_key": timer_key}
                    pending["gpu_intervals"].append(interval_for_join)
                    pending["audit_events"].append(
                        {
                            "forward_iter": forward_iter,
                            "batch_id": pending["batch_id"],
                            "category": _kwargs.get("category", "unknown"),
                            "graph_bs": _kwargs.get("graph_bs"),
                            "graph_key": _kwargs.get("graph_key"),
                            "timer_key": timer_key,
                            "event_duration_s": t,
                            "timer_start_host_ns": _kwargs.get("timer_start_host_ns"),
                            "timer_end_host_ns": _kwargs.get("timer_end_host_ns"),
                            "timer_id": id(self.forward_pass_device_timer),
                        }
                    )
                else:
                    self.scheduler._fpm_audit_counters["unmatched_timer"] += 1
                    self.scheduler._fpm_completed_gpu_times.setdefault(forward_iter, []).append(
                        {"duration_s": t, "metadata": dict(_kwargs), "timer_key": timer_key}
                    )
                pending = self.scheduler._fpm_timeline_pending.get(forward_iter)
                if pending is not None:
                    interval = {
                        "timer_key": _kwargs.get("timer_key"),
                        "category": _kwargs.get("category"),
                        "graph_bs": _kwargs.get("graph_bs"),
                        "graph_key": _kwargs.get("graph_key"),
                        "cuda_event_elapsed_s": t,
                        "timer_start_host_ns": _kwargs.get("timer_start_host_ns"),
                        "timer_end_host_ns": _kwargs.get("timer_end_host_ns"),
                    }
                    pending.setdefault("gpu_event_intervals", []).append(interval)
                    event_row = {
                        "record_type": "gpu_event_completion",
                        "forward_iter": forward_iter,
                        "batch_id": pending["batch_id"],
                        **interval,
                    }
                    self._write_timeline_sidecar(forward_iter, event_row)

            if self.forward_pass_device_timer is not None:
                self.forward_pass_device_timer.add_reporter(_fpm_device_timer_reporter)
            else:
                self.forward_pass_device_timer = DeviceTimer(
                    reporter=_fpm_device_timer_reporter,
                )
            self._install_device_timer_on_runners()
            self._fpm_timer_owner = object()
            self.forward_pass_device_timer.register_owner(self._fpm_timer_owner)
            self.scheduler._fpm_uses_device_timer = True
            self.scheduler.enable_fpm = True
            logger.info(
                "FPM: ZMQ PUB bound on %s (dp_rank=%d, device_timer=%s)",
                endpoint,
                self.scheduler._fpm_dp_rank,
                self.scheduler._fpm_uses_device_timer,
            )

    def _snapshot_forward_pass_metrics(self, batch: ScheduleBatch):
        """Snapshot forward identity and request ownership before GPU execution."""
        sidecar_enabled = bool(os.environ.get("SGLANG_G021_FPM_SIDECAR"))
        if not self.scheduler._fpm_uses_device_timer:
            return
        forward_iter = batch.forward_iter
        if forward_iter is None:
            raise RuntimeError("FPM requires forward_iter")
        pending = self.scheduler._fpm_pending_batches.get(forward_iter)
        if pending is not None:
            if pending["batch_id"] != id(batch):
                raise RuntimeError(f"Duplicate FPM forward identity {forward_iter}")
            return
        scheduled = self._build_scheduled_request_metrics(batch)
        queued = self._build_queued_request_metrics()
        graph_replay_expected = batch.forward_mode.is_decode()
        clock_sync_start_ns = time.monotonic_ns()
        clock_sync_name = f"G022_CLOCK_SYNC_START forward_iter={forward_iter}"
        clock_sync_end_ns = time.monotonic_ns()
        if sidecar_enabled:
            try:
                import torch

                with torch.profiler.record_function(clock_sync_name):
                    clock_sync_end_ns = time.monotonic_ns()
            except RuntimeError:
                clock_sync_end_ns = time.monotonic_ns()
        request_owners = [
            {
                "rid": getattr(req, "rid", None),
                "input_tokens": len(
                    getattr(req, "origin_input_ids_unpadded", req.origin_input_ids)
                ),
                "output_tokens_before_forward": len(req.output_ids),
                "first_token_time_perf_counter": getattr(
                    getattr(req, "time_stats", None), "first_token_time", None
                ),
                "prefill_finished_time_perf_counter": getattr(
                    getattr(req, "time_stats", None), "prefill_finished_time", None
                ),
            }
            for req in batch.reqs
        ]
        pending = {
            "forward_iter": forward_iter,
            "batch_id": id(batch),
            "mode": batch.forward_mode.name,
            "scheduled_batch_size": batch.batch_size(),
            "batch_size": batch.batch_size(),
            "actual_tokens": getattr(batch, "actual_input_token_count", batch.batch_size()),
            "actual_batch_size": batch.batch_size(),
            "actual_input_token_count": getattr(batch, "actual_input_token_count", batch.batch_size()),
            "scheduled_prefill_tokens": scheduled.sum_prefill_tokens,
            "scheduled_decode_requests": scheduled.num_decode_requests,
            "scheduled_decode_kv_tokens": scheduled.sum_decode_kv_tokens,
            "requests": request_owners,
            "scheduled": scheduled,
            "queued": queued,
            "running": len(batch.reqs),
            "prefill_tokens": scheduled.sum_prefill_tokens,
            "decode_tokens": batch.batch_size() if batch.forward_mode.is_decode() else 0,
            "queued_prefill_reqs": queued.num_prefill_requests,
            "queued_prefill_tokens": queued.sum_prefill_tokens,
            "queued_decode_reqs": queued.num_decode_requests,
            "queued_decode_tokens": queued.sum_decode_kv_tokens,
            "start_time": getattr(batch, "fpm_start_time", time.monotonic()),
            "forward_start_time": time.monotonic(),
            "scheduler_start_time": time.monotonic(),
            "scheduler_end_time": None,
            "gpu_intervals": [],
            "audit_events": [],
            "timer_keys": set(),
            "published": False,
            "scheduler_start_monotonic_ns": time.monotonic_ns(),
            "batch_result_start_monotonic_ns": None,
            "scheduler_end_monotonic_ns": None,
            "trace_capture_active": getattr(batch, "profiling_trace_active", False),
            "clock_sync_name": clock_sync_name,
            "clock_sync_monotonic_start_ns": clock_sync_start_ns,
            "clock_sync_monotonic_end_ns": clock_sync_end_ns,
            "first_token_ready_perf_counter": max(
                (r["first_token_time_perf_counter"] or 0.0 for r in request_owners),
                default=0.0,
            ) or None,
            "graph_replay_expected": graph_replay_expected,
            "graph_key": None,
            "gpu_event_intervals": [],
            "sidecar_status": "SCHEDULED",
        }
        self.scheduler._fpm_pending_batches[forward_iter] = pending
        if sidecar_enabled:
            self.scheduler._fpm_timeline_pending[forward_iter] = pending
            self._write_timeline_sidecar(
                forward_iter, {**pending, "record_type": "forward_snapshot"}
            )

    def _write_timeline_sidecar(self, forward_iter: int, item: dict):
        """Append one bounded research record; consumers join rows by forward_iter."""
        path = os.environ.get("SGLANG_G021_FPM_SIDECAR")
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as dst:
            dst.write(
                json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n"
            )

    def _complete_forward_timeline(self, batch: ScheduleBatch):
        if not os.environ.get("SGLANG_G021_FPM_SIDECAR"):
            return
        forward_iter = batch.forward_iter
        pending = self.scheduler._fpm_timeline_pending.get(forward_iter)
        if pending is None:
            return
        pending["batch_result_start_monotonic_ns"] = time.monotonic_ns()
        clock_sync_start_ns = time.monotonic_ns()
        clock_sync_name = f"G022_CLOCK_SYNC_END forward_iter={forward_iter}"
        try:
            import torch

            with torch.profiler.record_function(clock_sync_name):
                clock_sync_end_ns = time.monotonic_ns()
        except RuntimeError:
            clock_sync_end_ns = time.monotonic_ns()
        pending["scheduler_end_monotonic_ns"] = time.monotonic_ns()
        pending["completion_clock_sync_name"] = clock_sync_name
        pending["completion_clock_sync_start_ns"] = clock_sync_start_ns
        pending["completion_clock_sync_end_ns"] = clock_sync_end_ns
        pending["request_completion_snapshot"] = [
            {
                "rid": getattr(req, "rid", None),
                "output_tokens_after_forward": len(req.output_ids),
                "first_token_time_perf_counter": getattr(
                    getattr(req, "time_stats", None), "first_token_time", None
                ),
                "prefill_finished_time_perf_counter": getattr(
                    getattr(req, "time_stats", None), "prefill_finished_time", None
                ),
                "completion_time_perf_counter": getattr(
                    getattr(req, "time_stats", None), "completion_time", None
                ),
            }
            for req in batch.reqs
        ]
        pending["sidecar_status"] = "SCHEDULER_COMPLETE"
        completion = {
            "record_type": "scheduler_completion",
            "forward_iter": forward_iter,
            "batch_id": pending["batch_id"],
            "batch_result_start_monotonic_ns": pending["batch_result_start_monotonic_ns"],
            "scheduler_end_monotonic_ns": pending["scheduler_end_monotonic_ns"],
            "sidecar_status": pending["sidecar_status"],
        }
        self._write_timeline_sidecar(forward_iter, completion)

    def _write_fpm_research_sidecar(
        self, forward_iter: int, item: dict, *, status: str, invalid_reason=None,
        scheduler_wall_s=None, gpu_elapsed_s=None, queue_accepted=None,
    ):
        """Append bounded, opt-in FPM/GPU-event audit rows to a JSONL sidecar."""
        path = os.environ.get("SGLANG_G022_FPM_GPU_SIDECAR")
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        row = {
            "record_type": "fpm_forward", "forward_iter": forward_iter,
            "status": status, "invalid_reason": invalid_reason,
            "mode": item.get("mode"), "batch_size": item.get("actual_batch_size"),
            "scheduled_decode_requests": item.get("scheduled_decode_requests"),
            "scheduled_decode_kv_tokens": item.get("scheduled_decode_kv_tokens"),
            "scheduled_prefill_tokens": item.get("scheduled_prefill_tokens"),
            "graph_replay_expected": item.get("graph_replay_expected"),
            "graph_key": item.get("graph_key"), "trace_capture_active": item.get("trace_capture_active"),
            "gpu_interval_count": len(item.get("gpu_intervals", [])),
            "gpu_event_intervals": item.get("gpu_event_intervals", []),
            "scheduler_wall_s": scheduler_wall_s,
            "gpu_interval_sum_s": item.get("gpu_interval_sum_s"),
            "gpu_elapsed_s": gpu_elapsed_s, "queue_accepted": queue_accepted,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    def _build_scheduled_request_metrics(self, batch: ScheduleBatch):
        from sglang.srt.observability.forward_pass_metrics import (
            ScheduledRequestMetrics,
            WelfordAccumulator,
        )

        num_prefill_requests = 0
        sum_prefill_tokens = 0
        sum_prefill_kv_tokens = 0
        prefill_lengths = WelfordAccumulator()

        if batch.forward_mode.is_mixed():
            decode_req_ids = {id(req) for req in batch.decoding_reqs or []}
            prefill_reqs = [req for req in batch.reqs if id(req) not in decode_req_ids]
        elif batch.forward_mode.is_extend():
            prefill_reqs = batch.reqs
        else:
            prefill_reqs = []

        if prefill_reqs:
            stats = batch.prefill_stats
            for req in prefill_reqs:
                prefill_lengths.add(len(req.origin_input_ids))
            num_prefill_requests = stats.num_new_seqs if stats else len(prefill_reqs)
            sum_prefill_tokens = stats.log_input_tokens if stats else 0
            sum_prefill_kv_tokens = sum(len(req.prefix_indices) for req in prefill_reqs)

        decode_kv = WelfordAccumulator()
        if batch.forward_mode.is_mixed():
            for req in batch.decoding_reqs or []:
                decode_kv.add(req.seqlen)
        elif batch.forward_mode.is_decode():
            for sl in batch.seq_lens_cpu:
                decode_kv.add(int(sl))

        return ScheduledRequestMetrics(
            num_prefill_requests=num_prefill_requests,
            sum_prefill_tokens=sum_prefill_tokens,
            var_prefill_length=prefill_lengths.variance(),
            sum_prefill_kv_tokens=sum_prefill_kv_tokens,
            num_decode_requests=decode_kv.count,
            sum_decode_kv_tokens=decode_kv.total,
            var_decode_kv_tokens=decode_kv.variance(),
        )

    def _build_queued_request_metrics(self):
        from sglang.srt.observability.forward_pass_metrics import (
            QueuedRequestMetrics,
            WelfordAccumulator,
        )

        prefill_q = WelfordAccumulator()
        decode_q = WelfordAccumulator()
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in self.scheduler.disagg_prefill_bootstrap_queue.queue:
                prefill_q.add(len(req.origin_input_ids))
        elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            for req in self.scheduler.disagg_decode_prealloc_queue.queue:
                decode_q.add(req.seqlen)
            for req in self.scheduler.disagg_decode_transfer_queue.queue:
                decode_q.add(req.seqlen)
        else:
            for req in self.scheduler.waiting_queue:
                if len(req.output_ids) > 0:
                    decode_q.add(req.seqlen)
                else:
                    prefill_q.add(len(req.origin_input_ids))

        return QueuedRequestMetrics(
            num_prefill_requests=prefill_q.count,
            sum_prefill_tokens=prefill_q.total,
            var_prefill_length=prefill_q.variance(),
            num_decode_requests=decode_q.count,
            sum_decode_kv_tokens=decode_q.total,
            var_decode_kv_tokens=decode_q.variance(),
        )

    def update_spec_metrics(self, bs: int, num_correct_drafts: int):
        self.spec_num_accept_tokens += num_correct_drafts + bs
        self.spec_num_forward_ct += bs

        # Bonus tokens updated elsewhere
        self.num_generated_tokens += num_correct_drafts

    def _init_estimated_perf_constants(self) -> None:
        model_config = self.scheduler.model_config
        hf_text_config = model_config.hf_text_config

        hidden_size = float(model_config.hidden_size)
        num_layers = float(getattr(model_config, "num_attention_layers", 0))
        head_dim = float(getattr(model_config, "head_dim", 0))
        num_attn_heads = float(
            model_config.get_num_attention_heads(self.scheduler.ps.tp_size)
        )
        num_kv_heads = float(model_config.get_num_kv_heads(self.scheduler.ps.tp_size))
        intermediate_size = getattr(hf_text_config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(hf_text_config, "ffn_hidden_size", 0)
        intermediate_size = float(intermediate_size)

        dtype_num_bytes = getattr(model_config.dtype, "itemsize", None)
        if dtype_num_bytes is None:
            dtype_num_bytes = 2
        # Keep this estimator lightweight and consistent with current server dtype.
        # KV cache quantization-aware bytes can be added in a follow-up.
        act_bytes = float(dtype_num_bytes)
        w_bytes = float(dtype_num_bytes)
        cache_bytes = float(dtype_num_bytes)

        # Linear-layer FLOPs per token on one GPU.
        attn_linear_flops = (
            2.0 * hidden_size * head_dim * (num_attn_heads + 2.0 * num_kv_heads)
            + 2.0 * hidden_size * head_dim * num_attn_heads
        )
        mlp_flops = (
            6.0 * hidden_size * intermediate_size if intermediate_size > 0 else 0.0
        )
        self._linear_flops_per_token = max(
            0.0, (attn_linear_flops + mlp_flops) * num_layers
        )

        # Attention dot-product FLOPs coefficient to multiply token-context product.
        # attn_qk + attn_av = 4 * q * TC * d * L
        self._attn_dot_flops_coeff = 4.0 * num_attn_heads * head_dim * num_layers

        # KV cache bytes (write one K and one V vector per generated token).
        self._kv_cache_bytes_per_token = (
            2.0 * num_layers * num_kv_heads * head_dim * cache_bytes
        )

        # Weight read bytes per token.
        self._weight_read_bytes_per_token = (
            hidden_size
            * head_dim
            * (num_attn_heads + 2.0 * num_kv_heads)
            * w_bytes
            * num_layers
            + hidden_size * head_dim * num_attn_heads * w_bytes * num_layers
            + (
                3.0 * hidden_size * intermediate_size * w_bytes * num_layers
                if intermediate_size > 0
                else 0.0
            )
        )

        # Activation movement bytes per token (coarse approximation).
        self._qkv_act_bytes_per_token = (
            hidden_size * act_bytes * num_layers
            + (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
            + head_dim * num_attn_heads * act_bytes * num_layers
            + hidden_size * act_bytes * num_layers
        )
        self._ffn_act_bytes_per_token = (
            3.0 * intermediate_size * act_bytes * num_layers
            if intermediate_size > 0
            else 0.0
        )

        # Prefill reads Q/K/V activations from on-device memory.
        self._prefill_attn_act_read_per_token = (
            (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
        )

        # Decode reads Q from activation memory; K/V reads are from KV cache.
        self._decode_q_read_bytes_per_token = (
            num_attn_heads * head_dim * act_bytes * num_layers
        )

    def _estimate_prefill_perf(self, num_tokens: int) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        # Causal prefill token-context product.
        context_product = tokens * (tokens + 1) / 2.0
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * context_product
        )

        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._prefill_attn_act_read_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def _estimate_decode_perf(
        self, batch: ScheduleBatch, num_tokens: int
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        total_context = float(_decode_total_seq_lens(batch))
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * total_context
        )
        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._decode_q_read_bytes_per_token
            + total_context * self._kv_cache_bytes_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def reset_metrics(self):
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_accept_tokens = 0
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0
        self.spec_total_num_forward_ct = 0

    def report_prefill_stats(
        self,
        batch: Optional[ScheduleBatch],
        prefill_stats: PrefillStats,
        can_run_cuda_graph: bool,
        dp_cooperation_info: Optional[DPCooperationInfo] = None,
    ):
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        now = time.perf_counter()
        gap_latency = now - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = now
        self.last_input_throughput = (
            prefill_stats.log_input_tokens / gap_latency if gap_latency > 0 else 0.0
        )

        pool_stats = self.scheduler.pool_stats_observer.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_prefill_usage_msg_parts()) + ", "

        self.stats.new_token_ratio = prefill_stats.new_token_ratio
        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.scheduler.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""

        msg = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {prefill_stats.num_running_reqs.total}, "
            f"#queue-req: {len(self.scheduler.waiting_queue)}, "
            f"#pending-token: {prefill_stats.num_pending_tokens}, "
        )

        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            msg += f"#bootstrap-req: {len(self.scheduler.disagg_prefill_bootstrap_queue.queue)}, "
            msg += (
                f"#inflight-req: {len(self.scheduler.disagg_prefill_inflight_queue)}, "
            )

        if (
            self.scheduler.server_args.language_only
            and self.scheduler.server_args.encoder_transfer_backend
            == "zmq_to_scheduler"
        ):
            msg += (
                f"waiting-image-req: {len(self.scheduler.mm_receiver.waiting_list)}, "
            )

        msg += f"{self._graph_backend_label}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"

        if self.enable_mfu_metrics and gap_latency > 0:
            flops, _, _ = self._estimate_prefill_perf(prefill_stats.log_input_tokens)
            tflops_per_s = flops / gap_latency / 1e12
            msg += f", est. prefill TFLOPS/s (per GPU): {tflops_per_s:.2f}"

        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            self.metrics_collector.increment_prefill_cuda_graph_pass(
                value=can_run_cuda_graph
            )
            self.metrics_collector.increment_realtime_tokens(
                prefill_compute_tokens=prefill_stats.log_input_tokens,
                prefill_cache_tokens=prefill_stats.log_hit_tokens,
                dp_cooperation_info=dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_prefill_perf(
                    prefill_stats.log_input_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )

            priority_enabled = self.scheduler.enable_priority_scheduling
            total_tokens = prefill_stats.log_input_tokens + prefill_stats.log_hit_tokens
            cache_hit_rate = (
                prefill_stats.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            # Basics
            self.stats.num_running_reqs = prefill_stats.num_running_reqs
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.scheduler.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            # Memory pool usage ratios / Absolute token counts
            pool_stats.update_scheduler_stats(self.stats)

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_bootstrap_queue.queue,
                    priority_enabled,
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_inflight_queue, priority_enabled
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
            elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Utilization / LoRA / HiCache
            self._calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self._update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self.scheduler.kv_events_publisher.emit_kv_metrics()
        self.scheduler.kv_events_publisher.publish_kv_events()

    def report_decode_stats(
        self,
        can_run_cuda_graph: bool,
        running_batch: ScheduleBatch = None,
        num_correct_drafts: int = 0,
    ):
        batch = running_batch or self.scheduler.running_batch

        # Every-iteration work: realtime token counting + status logger
        if self.current_scheduler_metrics_enabled:
            decode_tokens = batch.batch_size() + num_correct_drafts
            self.metrics_collector.increment_realtime_tokens(
                # TODO unify this w/ the bumping logic in `Scheduler.num_generated_tokens` accumulator
                decode_tokens=decode_tokens,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_decode_perf(
                    batch, decode_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )
                self._mfu_log_flops += flops
                self._mfu_log_read_bytes += read_bytes
                self._mfu_log_write_bytes += write_bytes

            if x := self.scheduler_status_logger:
                x.maybe_dump(batch, self.scheduler.waiting_queue)

        # Periodic work: log + heavy metrics at decode_log_interval
        if self.forward_ct_decode % self.scheduler.server_args.decode_log_interval != 0:
            return
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)

        pool_stats = self.scheduler.pool_stats_observer.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_decode_usage_msg_parts()) + ", "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.scheduler.server_args.decode_log_interval
            )

        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.scheduler.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""
        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"

        if self.scheduler.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            spec_accept_length = self.spec_num_accept_tokens / self.spec_num_forward_ct
            num_correct_drafts = self.spec_num_accept_tokens - self.spec_num_forward_ct
            if self.scheduler.server_args.speculative_num_draft_tokens:
                draft_per_round = (
                    self.scheduler.server_args.speculative_num_draft_tokens - 1
                )
            else:
                draft_per_round = self.scheduler.server_args.speculative_num_steps or 0
            total_draft_tokens = self.spec_num_forward_ct * draft_per_round
            spec_accept_rate = (
                num_correct_drafts / total_draft_tokens if total_draft_tokens > 0 else 0
            )
            self.spec_total_num_accept_tokens += self.spec_num_accept_tokens
            self.spec_total_num_forward_ct += self.spec_num_forward_ct
            self.spec_num_accept_tokens = self.spec_num_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "
        cache_hit_rate = 0.0

        if self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.scheduler.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.scheduler.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.scheduler.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.scheduler.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.scheduler.disagg_decode_prealloc_queue.retracted_queue)}, "

        if (
            self.scheduler.server_args.language_only
            and self.scheduler.server_args.encoder_transfer_backend
            == "zmq_to_scheduler"
        ):
            msg += (
                f"waiting-image-req: {len(self.scheduler.mm_receiver.waiting_list)}, "
            )

        msg += (
            f"{self._graph_backend_label}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.scheduler.waiting_queue)}"
        )

        if self.enable_mfu_metrics and gap_latency > 0:
            flops_per_s = self._mfu_log_flops / gap_latency
            read_bytes_per_s = self._mfu_log_read_bytes / gap_latency
            write_bytes_per_s = self._mfu_log_write_bytes / gap_latency
            tflops_per_s = flops_per_s / 1e12
            read_gb_per_s = read_bytes_per_s / 1e9
            write_gb_per_s = write_bytes_per_s / 1e9
            msg += (
                f", est. decode TFLOPS/s (per GPU): {tflops_per_s:.2f}, "
                f"est. read BW (GB/s per GPU): {read_gb_per_s:.2f}, "
                f"est. write BW (GB/s per GPU): {write_gb_per_s:.2f}"
            )
            self._mfu_log_flops = 0.0
            self._mfu_log_read_bytes = 0.0
            self._mfu_log_write_bytes = 0.0

        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            priority_enabled = self.scheduler.enable_priority_scheduling

            # Basics
            self.stats.num_running_reqs = QueueCount.from_reqs(
                batch.reqs, priority_enabled
            )
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.scheduler.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.cache_hit_rate = cache_hit_rate
            self.stats.decode_sum_seq_lens = _decode_total_seq_lens(batch)

            # Memory pool usage ratios / Absolute token counts
            pool_stats.update_scheduler_stats(self.stats)

            # Speculative decoding
            self.stats.spec_accept_length = spec_accept_length
            self.stats.spec_accept_rate = spec_accept_rate

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_bootstrap_queue.queue,
                    priority_enabled,
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_inflight_queue, priority_enabled
                )
            elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Streaming session metrics
            self.stats.num_streaming_sessions = (
                self.scheduler.pool_stats_observer.streaming_session_count()
            )
            self.stats.streaming_session_held_tokens = (
                self.scheduler.pool_stats_observer.session_held_tokens()
            )

            # Routing key metrics
            # (to reduce the overhead, we only compute this when all requests have routing_key)
            if all(r.routing_key is not None for r in batch.reqs):
                running_routing_keys = [r.routing_key for r in batch.reqs]
                waiting_routing_keys = [
                    r.routing_key for r in self.scheduler.waiting_queue
                ]
                (
                    self.stats.num_unique_running_routing_keys,
                    self.stats.routing_key_running_req_counts,
                ) = compute_routing_key_stats(running_routing_keys)
                _, self.stats.routing_key_all_req_counts = compute_routing_key_stats(
                    running_routing_keys + waiting_routing_keys
                )

            # Utilization / LoRA / HiCache
            self._calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self._update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self.scheduler.kv_events_publisher.emit_kv_metrics()
        self.scheduler.kv_events_publisher.publish_kv_events()

    def log_batch_result_stats(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if not self.enable_metrics:
            return
        if not isinstance(result, GenerationBatchResult):
            return

        if (m := result.expert_distribution_metrics) is not None:
            self.metrics_collector.increment_eplb_balancedness(
                forward_mode=batch.forward_mode.name.lower(),
                balancedness=m.eplb_balancedness.item(),
            )

    def _emit_forward_pass_metrics(
        self,
        batch: ScheduleBatch,
        result=None,
    ):
        """Emit per-iteration ForwardPassMetrics over ZMQ PUB.

        Prefers GPU-accurate timing from DeviceTimer (which wraps
        model_runner.forward / cuda_graph.replay via PR #24197).
        Falls back to monotonic clock when DeviceTimer is not enabled.
        """
        if not self.scheduler.enable_fpm:
            return

        from sglang.srt.observability.forward_pass_metrics import (
            ForwardPassMetrics,
        )

        if self.scheduler._fpm_uses_device_timer:
            if not hasattr(self.scheduler, "_fpm_audit_counters"):
                self.scheduler._fpm_audit_counters = defaultdict(int)
            # Snapshot immediately on the scheduler thread. The ScheduleBatch
            # is mutable/reused by scheduling logic; do not retain it by reference.
            forward_iter = batch.forward_iter
            if forward_iter is None:
                raise RuntimeError("FPM requires a scheduler forward_iter identity")
            pending = self.scheduler._fpm_pending_batches.get(forward_iter)
            if forward_iter in self.scheduler._fpm_closed_batches:
                logger.error("Late FPM timer completion for closed forward=%s", forward_iter)
                return
            if pending is None:
                # Compatibility for direct reporter callers. Production scheduler
                # execution snapshots before model forward begins.
                self._snapshot_forward_pass_metrics(batch)
                pending = self.scheduler._fpm_pending_batches[forward_iter]
            pending["scheduler_end_time"] = time.monotonic()
            for completed_iter in list(self.scheduler._fpm_completed_gpu_times):
                item = self.scheduler._fpm_pending_batches.get(completed_iter)
                if item is None:
                    continue
                intervals = self.scheduler._fpm_completed_gpu_times.pop(completed_iter)
                for interval in intervals:
                    if interval["timer_key"] in item["timer_keys"]:
                        logger.error("Duplicate FPM timer key while joining: %s", interval["timer_key"])
                        continue
                    item["timer_keys"].add(interval["timer_key"])
                    item["gpu_intervals"].append(interval)
                    item["audit_events"].append(
                        {
                            "forward_iter": completed_iter,
                            "batch_id": item["batch_id"],
                            "category": interval["metadata"].get("category", "unknown"),
                            "graph_bs": interval["metadata"].get("graph_bs"),
                            "graph_key": interval["metadata"].get("graph_key"),
                            "timer_key": interval["timer_key"],
                            "event_duration_s": interval["duration_s"],
                            "timer_start_host_ns": interval["metadata"].get("timer_start_host_ns"),
                            "timer_end_host_ns": interval["metadata"].get("timer_end_host_ns"),
                            "timer_id": id(self.forward_pass_device_timer),
                        }
                    )
            if batch.forward_mode.is_prebuilt():
                pending["timer_status"] = "UNSUPPORTED"
                pending["scheduler_wall_s"] = max(
                    0.0, pending["scheduler_end_time"] - pending["forward_start_time"]
                )
                self.scheduler._fpm_pending_batches.pop(forward_iter)
                self.scheduler._fpm_closed_batches.add(forward_iter)
                logger.info(
                    "FPM_FORWARD_STATUS forward_iter=%s status=UNSUPPORTED reason=prebuilt_no_model_forward",
                    forward_iter,
                )
                return
            # Event.query remains nonblocking. Join completions to the immutable
            # snapshot by forward identity, even if query succeeds immediately.
            self.forward_pass_device_timer._report()
            for completed_iter in list(self.scheduler._fpm_completed_gpu_times):
                item = self.scheduler._fpm_pending_batches.get(completed_iter)
                if item is not None:
                    intervals = self.scheduler._fpm_completed_gpu_times.pop(
                        completed_iter
                    )
                    for interval in intervals:
                        if interval["timer_key"] in item["timer_keys"]:
                            logger.error(
                                "Duplicate FPM timer key while joining: %s",
                                interval["timer_key"],
                            )
                            continue
                        item["timer_keys"].add(interval["timer_key"])
                        item["gpu_intervals"].append(interval)
                        item["audit_events"].append(
                            {
                                "forward_iter": completed_iter,
                                "batch_id": item["batch_id"],
                                "category": interval["metadata"].get("category", "unknown"),
                                "graph_bs": interval["metadata"].get("graph_bs"),
                                "graph_key": interval["metadata"].get("graph_key"),
                                "timer_key": interval["timer_key"],
                                "event_duration_s": interval["duration_s"],
                                "timer_start_host_ns": interval["metadata"].get("timer_start_host_ns"),
                                "timer_end_host_ns": interval["metadata"].get("timer_end_host_ns"),
                                "timer_id": id(self.forward_pass_device_timer),
                            }
                        )
            if pending["scheduler_end_time"] is None:
                if not pending.get("scheduler_completion_counted"):
                    pending["scheduler_completion_counted"] = True
                    self.scheduler._fpm_audit_counters["scheduler_forward_completed"] += 1
                return
            if not pending.get("scheduler_completion_counted"):
                pending["scheduler_completion_counted"] = True
                self.scheduler._fpm_audit_counters["scheduler_forward_completed"] += 1
            for completed_iter in sorted(self.scheduler._fpm_pending_batches):
                item = self.scheduler._fpm_pending_batches[completed_iter]
                if not item["gpu_intervals"] or item["scheduler_end_time"] is None:
                    continue
                if len(item["gpu_intervals"]) != 1:
                    item["timer_status"] = "INVALID"
                    item["invalid_reason"] = "multiple_gpu_intervals_without_device_elapsed"
                    logger.error(
                        "FPM forward_iter=%s INVALID: %d GPU intervals but v1 needs "
                        "one complete Forward elapsed; duration sum is diagnostic only",
                        completed_iter,
                        len(item["gpu_intervals"]),
                    )
                    self.scheduler._fpm_completed_status[completed_iter] = {
                        "status": "INVALID",
                        "reason": item["invalid_reason"],
                        "gpu_interval_sum_s": sum(
                            interval["duration_s"] for interval in item["gpu_intervals"]
                        ),
                    }
                    self._write_fpm_research_sidecar(
                        completed_iter, item, status="INVALID",
                        invalid_reason=item["invalid_reason"],
                        scheduler_wall_s=max(0.0, item["scheduler_end_time"] - item["start_time"]),
                        gpu_elapsed_s=None,
                    )
                    self.scheduler._fpm_pending_batches.pop(completed_iter)
                    self.scheduler._fpm_closed_batches.add(completed_iter)
                    continue
                counters = getattr(self.scheduler, "_fpm_audit_counters", None)
                if counters is None:
                    self.scheduler._fpm_audit_counters = counters = defaultdict(int)
                counters["forward_completed"] += 1
                gpu_elapsed = sum(
                    interval["duration_s"] for interval in item["gpu_intervals"]
                )
                scheduler_wall_duration = max(
                    0.0,
                    item["scheduler_end_time"] - item["start_time"],
                )
                if os.environ.get("SGLANG_G021_FPM_DIAG") == "1":
                    logger.info(
                        "G021_FPM_FORWARD_COMPLETE forward_iter=%s batch_id=%x "
                        "mode=%s batch_size=%d scheduler_wall_s=%.9f "
                        "gpu_event_count=%d gpu_interval_sum_s=%.9f "
                        "graph_bs=%s graph_key=%s tokens=%d timer_id=%x",
                        completed_iter,
                        item["batch_id"],
                        item["mode"],
                        item["batch_size"],
                        scheduler_wall_duration,
                        len(item["gpu_intervals"]),
                        gpu_elapsed,
                        item["gpu_intervals"][0]["metadata"].get("graph_bs"),
                        item["gpu_intervals"][0]["metadata"].get("graph_key"),
                        item["scheduled"].sum_prefill_tokens
                        + item["scheduled"].num_decode_requests,
                        id(self.forward_pass_device_timer),
                    )
                # FPM v1 wall_time is scheduler wall duration. A single CUDA
                # event interval is the complete ModelRunner forward boundary
                # in this serving path; multiple intervals are not combined into
                # elapsed without device-clock endpoints, and host record bounds
                # are not GPU-active timestamps.
                scheduler_wall_duration = max(
                    0.0,
                    item["scheduler_end_time"] - item["start_time"],
                )
                gpu_forward_elapsed = (
                    gpu_elapsed if len(item["gpu_intervals"]) == 1 else None
                )
                if os.environ.get("SGLANG_G021_FPM_DIAG") == "1":
                    logger.info(
                        "G021_FPM_TIME_AUDIT forward_iter=%s scheduler_wall_s=%.9f "
                        "gpu_interval_sum_s=%.9f gpu_forward_elapsed_s=%s "
                        "gpu_active_union=UNSUPPORTED status=%s",
                        completed_iter,
                        scheduler_wall_duration,
                        gpu_elapsed,
                        "unsupported" if gpu_forward_elapsed is None else f"{gpu_forward_elapsed:.9f}",
                        "MEASURED" if gpu_forward_elapsed is not None else "PARTIAL",
                    )
                item["gpu_interval_sum_s"] = gpu_elapsed
                item["gpu_elapsed_s"] = gpu_forward_elapsed
                item["gpu_union_s"] = None
                self._complete_forward_timeline(batch)
                timeline_pending = getattr(self.scheduler, "_fpm_timeline_pending", None)
                if timeline_pending is not None:
                    timeline_pending.pop(completed_iter, None)
                fpm = ForwardPassMetrics(
                    worker_id=self.scheduler._fpm_worker_id,
                    dp_rank=self.scheduler._fpm_dp_rank,
                    wall_time=scheduler_wall_duration,
                    scheduled_requests=item["scheduled"],
                    queued_requests=item["queued"],
                )
                self.scheduler._fpm_audit_counters["active_emit_attempted"] += 1
                queue_accepted = self.scheduler._fpm_publisher.publish(fpm)
                if queue_accepted:
                    self.scheduler._fpm_audit_counters["queue_accepted"] += 1
                else:
                    self.scheduler._fpm_audit_counters["queue_dropped"] += 1
                if not queue_accepted:
                    logger.error(
                        "FPM publish enqueue dropped forward_iter=%s batch_id=%x",
                        completed_iter,
                        item["batch_id"],
                    )
                self._write_fpm_research_sidecar(
                    completed_iter, item, status="MEASURED",
                    invalid_reason=None, scheduler_wall_s=scheduler_wall_duration,
                    gpu_elapsed_s=gpu_forward_elapsed, queue_accepted=queue_accepted,
                )
                self.scheduler._fpm_pending_batches.pop(completed_iter)
                self.scheduler._fpm_closed_batches.add(completed_iter)
                if os.environ.get("SGLANG_G021_FPM_DIAG") == "1":
                    logger.info(
                        "G021_FPM_EMIT_ACTIVE batch_id=%x forward_iter=%s "
                        "mode=%s batch_size=%d wall_time_s=%.9f timer_id=%x",
                        item["batch_id"],
                        completed_iter,
                        item["mode"],
                        item["batch_size"],
                        scheduler_wall_duration,
                        id(self.forward_pass_device_timer),
                    )
        else:
            wall_time = max(0.0, time.monotonic() - batch.fpm_start_time)
            fpm = ForwardPassMetrics(
                worker_id=self.scheduler._fpm_worker_id,
                dp_rank=self.scheduler._fpm_dp_rank,
                wall_time=wall_time,
                scheduled_requests=self._build_scheduled_request_metrics(batch),
                queued_requests=self._build_queued_request_metrics(),
            )
            if not self.scheduler._fpm_publisher.publish(fpm):
                logger.error(
                    "FPM publish enqueue dropped scheduler-only batch_id=%x",
                    id(batch),
                )

    def _shutdown_fpm(self):
        """Drain timer callbacks, release timer ownership, and stop FPM publishing."""
        if not self.scheduler.enable_fpm:
            return
        self.scheduler._fpm_shutting_down = True
        timer = self.forward_pass_device_timer
        dropped = 0
        if timer is not None:
            release = getattr(timer, "release_owner", None)
            fpm_owner = getattr(self, "_fpm_timer_owner", None)
            if release is not None and fpm_owner is not None:
                dropped = release(fpm_owner, timeout=1.0)
                if dropped:
                    for interval in list(timer._intervals):
                        key = interval.metadata.get("timer_key") if interval.metadata else None
                        self.scheduler._fpm_audit_counters["timer_completed"] += 1
                        if key is not None:
                            self.scheduler._fpm_audit_counters["timer_duplicate"] += 1
                        else:
                            self.scheduler._fpm_audit_counters["unmatched_timer"] += 1
                        logger.warning(
                            "FPM shutdown discarded incomplete event key=%s metadata=%s",
                            key,
                            interval.metadata,
                        )
            else:
                dropped = timer.shutdown()
        self.scheduler._fpm_audit_counters["pending_timeout"] = (
            dropped if dropped else 0
        )
        self.scheduler._fpm_pending_batches.clear()
        self.scheduler._fpm_completed_gpu_times.clear()
        self.scheduler._fpm_closed_batches.clear()
        if timer is not None and self._metrics_timer_owner is not None:
            metrics_dropped = timer.release_owner(
                self._metrics_timer_owner, timeout=1.0
            )
            if metrics_dropped:
                logger.warning(
                    "Metrics DeviceTimer shutdown dropped %d intervals",
                    metrics_dropped,
                )
        self.scheduler._fpm_publisher.shutdown()

    def _log_hicache_stats(self):
        """Populate HiCache host-tier stats on self.stats.

        These are pushed to Prometheus by SchedulerMetricsCollector.log_stats().
        """
        if not self.scheduler.enable_hierarchical_cache:
            return

        host_pool = getattr(
            self.scheduler.tree_cache, "token_to_kv_pool_host", None
        ) or getattr(self.scheduler.tree_cache, "full_kv_pool_host", None)
        assert host_pool is not None, "Host pool not found"
        self.stats.hicache_host_used_tokens = (
            host_pool.size - host_pool.available_size()
        )
        self.stats.hicache_host_total_tokens = host_pool.size

    def _update_lora_metrics(self):
        """Update LoRA pool metrics for monitoring and autoscaling."""
        if not self.scheduler.enable_lora:
            return

        try:
            # Get LoRA memory pool stats
            lora_manager = self.scheduler.tp_worker.model_runner.lora_manager
            if lora_manager is None or lora_manager.memory_pool is None:
                return

            mem_pool = lora_manager.memory_pool
            slots_total = mem_pool.max_loras_per_batch

            # Calculate active adapters from running batch
            # This gives a true measure of current load for autoscaling purposes
            active_lora_ids = set()

            # For PP mode, check all running micro batches
            if self.scheduler.server_args.pp_size > 1:
                for batch in self.scheduler.running_mbs:
                    if batch and hasattr(batch, "reqs"):
                        for req in batch.reqs:
                            if hasattr(req, "lora_id") and req.lora_id is not None:
                                active_lora_ids.add(req.lora_id)
            # For normal mode, check running_batch
            elif self.scheduler.running_batch:
                if hasattr(self.scheduler.running_batch, "reqs"):
                    for req in self.scheduler.running_batch.reqs:
                        if hasattr(req, "lora_id") and req.lora_id is not None:
                            active_lora_ids.add(req.lora_id)

            # Count active adapters (excluding None for base model)
            slots_used = len(active_lora_ids)
            utilization = slots_used / slots_total if slots_total > 0 else 0.0

            # Update stats
            self.stats.lora_pool_slots_used = slots_used
            self.stats.lora_pool_slots_total = slots_total
            self.stats.lora_pool_utilization = utilization

        except Exception as e:
            logger.warning(f"Failed to update LoRA metrics: {e}")

    def _calculate_utilization(self):
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            # TODO: max_running_requests_under_SLO has no setter — sglang:utilization stuck at 0 (regressed #22713).
            max_under_slo = getattr(
                self.scheduler, "max_running_requests_under_SLO", None
            )
            if max_under_slo is not None and max_under_slo > 0:
                self.stats.utilization = max(
                    self.stats.num_running_reqs.total / max_under_slo,
                    self.stats.token_usage / 0.9,
                )

    def update_device_timer(self):
        if not ENABLE_METRICS_DEVICE_TIMER:
            return
        self.forward_pass_device_timer._report()
        now = time.perf_counter()
        if self._device_timer_window_batch_count == 0:
            self._device_timer_window_start = now
            self._device_timer_window_gpu_time = 0.0
            cpu_time = 0
            self.fwd_occupancy = float("nan")
        else:
            cpu_time = now - self._device_timer_window_start
            self.fwd_occupancy = min(
                self._device_timer_window_gpu_time / cpu_time * 100, 100
            )
        # ratio = self._device_timer_window_gpu_time / cpu_time if cpu_time > 0 else float("nan")
        # print(f"{self._device_timer_window_batch_count=} {self.fwd_occupancy=}, {self._device_timer_window_gpu_time=}, {cpu_time=}, {ratio=}")
        self._device_timer_window_batch_count += 1
        if (
            self._device_timer_window_batch_count
            >= self.scheduler.server_args.decode_log_interval
        ):
            self._device_timer_window_batch_count = 0

    def reset_device_timer_window(self):
        if ENABLE_METRICS_DEVICE_TIMER:
            self._device_timer_window_batch_count = 0
            self.fwd_occupancy = float("nan")

    def _maybe_log_idle_metrics(self):
        """Collect and log metrics every 30 seconds during idle."""
        if (
            not self.current_scheduler_metrics_enabled
            or time.perf_counter() <= self.metrics_collector.last_log_time + 30
        ):
            return

        self.scheduler.pool_stats_observer.get_pool_stats().update_scheduler_stats(
            self.stats
        )
        self.stats.num_streaming_sessions = (
            self.scheduler.pool_stats_observer.streaming_session_count()
        )
        self.stats.streaming_session_held_tokens = (
            self.scheduler.pool_stats_observer.session_held_tokens()
        )

        priority_enabled = self.scheduler.enable_priority_scheduling
        self.stats.num_running_reqs = QueueCount.from_reqs(
            self.scheduler.running_batch.reqs, priority_enabled
        )
        self.stats.gen_throughput = 0
        self.stats.num_queue_reqs = QueueCount.from_reqs(
            self.scheduler.waiting_queue, priority_enabled
        )
        self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_prefill_bootstrap_queue.queue, priority_enabled
            )
            self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_prefill_inflight_queue, priority_enabled
            )
        if self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
            )
            self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
            )
        self.metrics_collector.log_stats(self.stats)
