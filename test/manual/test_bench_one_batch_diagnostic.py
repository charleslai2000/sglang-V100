from types import SimpleNamespace
from unittest.mock import patch

from sglang.bench_one_batch import latency_test_run_once


class _Event:
    instances = []

    def __init__(self, enable_timing):
        assert enable_timing
        self.recorded = False
        self.__class__.instances.append(self)

    def record(self):
        assert not self.recorded
        self.recorded = True

    def synchronize(self):
        assert self.recorded

    def elapsed_time(self, other):
        assert self.recorded and other.recorded
        return 12.5


class TestBenchOneBatchDiagnostic:
    def test_marks_exactly_one_measured_forward_and_records_events_inside_range(self):
        _Event.instances = []
        events = []

        class _Nvtx:
            @staticmethod
            def range_push(label):
                events.append(("push", label))

            @staticmethod
            def range_pop():
                events.append(("pop",))

        class _Runner:
            def forward(self, value):
                events.append(("forward", value))
                return value + 1

        runner = _Runner()

        class _BenchRunner:
            torch_runner = runner

            def max_batch_size(self, input_len, output_len):
                return 1

            def clear(self):
                pass

            def extend(self, reqs):
                result = runner.forward(4096)
                return result, None, "batch"

            def synchronize(self):
                events.append(("sync",))

            def cleanup(self, batch):
                pass

        with (
            patch("sglang.bench_one_batch.torch.cuda.is_available", return_value=True),
            patch("sglang.bench_one_batch.torch.cuda.Event", _Event),
            patch("sglang.bench_one_batch.torch.cuda.nvtx", _Nvtx),
            patch("sglang.bench_one_batch.torch.cuda.synchronize"),
            patch("sglang.bench_one_batch.prepare_synthetic_inputs_for_latency_test", return_value=[]),
            patch("sglang.bench_one_batch.extend", side_effect=lambda reqs, mr: mr.extend(reqs)),
        ):
            # The helper dispatches through the module-level extend wrapper.
            with patch("sglang.bench_one_batch._TorchBenchRunner", return_value=_BenchRunner()):
                result = latency_test_run_once(
                    "test",
                    _BenchRunner(),
                    lambda *args, **kwargs: None,
                    [],
                    1,
                    4096,
                    1,
                    0,
                    False,
                    False,
                    ("CPU", "GPU"),
                    "",
                    "prefill",
                    0,
                    prefill_diagnostic_id="test-id",
                )

        assert result["prefill_latency"] >= 0
        assert events[0] == ("sync",)
        assert events[1][0] == "push"
        assert events[2] == ("forward", 4096)
        assert events[3][0] == "pop"
        assert len(_Event.instances) == 2
        assert all(event.recorded for event in _Event.instances)
