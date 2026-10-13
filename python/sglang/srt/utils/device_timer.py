from collections import deque
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import torch


logger = logging.getLogger(__name__)


class DeviceTimer:
    def __init__(self, reporter: Callable):
        self._intervals: Deque[_TimingInterval] = deque()
        self._reporters: List[Callable] = [reporter]
        self._next_timer_key = 0
        self._accepting = True
        self._closed = False
        self._owners = set()

    def add_reporter(self, reporter: Callable):
        if self._closed:
            raise RuntimeError("Cannot attach reporter to a closed DeviceTimer")
        if reporter not in self._reporters:
            self._reporters.append(reporter)

    def register_owner(self, owner: object) -> None:
        if self._closed:
            raise RuntimeError("Cannot register owner on a closed DeviceTimer")
        self._owners.add(id(owner))

    def release_owner(self, owner: object, timeout: float = 1.0) -> Optional[int]:
        self._owners.discard(id(owner))
        if self._owners:
            return None
        return self.shutdown(timeout=timeout)

    @contextmanager
    def wrap(self, metadata: Dict):
        if not self._accepting:
            yield
            return
        # Keep the CUDA event interval's host-clock brackets and a stable key so
        # the asynchronous completion can be joined to its scheduled forward.
        metadata = dict(metadata)
        metadata.setdefault("timer_key", self._next_timer_key)
        self._next_timer_key += 1
        metadata.setdefault("timer_start_host_ns", time.monotonic_ns())
        interval = _TimingInterval.create()
        self._intervals.append(interval)
        try:
            yield
        finally:
            metadata.setdefault("timer_end_host_ns", time.monotonic_ns())
            interval.end(metadata=metadata)
            self._report()

    def _report(self):
        if self._closed:
            return
        while len(self._intervals) > 0:
            interval = self._intervals[0]
            if not interval.end_event.query():
                break

            self._intervals.popleft()
            elapsed = interval.elapsed_time() / 1000.0
            for reporter in self._reporters:
                reporter(t=elapsed, **interval.metadata)

    def shutdown(self, timeout: float = 1.0) -> int:
        """Stop accepting intervals and nonblockingly drain completed events."""
        self._accepting = False
        deadline = time.monotonic() + max(0.0, timeout)
        while self._intervals and time.monotonic() < deadline:
            before = len(self._intervals)
            self._report()
            if len(self._intervals) == before:
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        dropped = len(self._intervals)
        if dropped:
            logger.warning(
                "DeviceTimer shutdown dropped %d incomplete CUDA event intervals",
                dropped,
            )
            self._intervals.clear()
        self._closed = True
        self._owners.clear()
        return dropped


class GapTimer(DeviceTimer):
    """Measures GPU idle gaps between consecutive uses of a stream.

    Where DeviceTimer.wrap() measures the duration *inside* a block,
    GapTimer.wrap() measures the time *between* consecutive blocks
    (gap = next_block_start - last_block_end).
    """

    def __init__(self, reporter: Callable):
        super().__init__(reporter)
        self._pending: Optional[_TimingInterval] = None

    @contextmanager
    def wrap(self, metadata: Dict):
        if not self._accepting:
            yield
            return
        if self._pending is not None:
            self._pending.end(metadata=metadata)
            self._intervals.append(self._pending)
            self._pending = None
            self._report()
        try:
            yield
        finally:
            self._pending = _TimingInterval.create()

    def shutdown(self, timeout: float = 1.0) -> int:
        self._accepting = False
        self._pending = None
        return super().shutdown(timeout)

    def cancel(self):
        """Discard a pending gap (e.g. server went idle)."""
        self._pending = None


@dataclass
class _TimingInterval:
    start_event: torch.cuda.Event
    end_event: Optional[torch.cuda.Event] = None
    metadata: Optional[Dict] = None

    @staticmethod
    def create():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return _TimingInterval(start_event=start_event)

    def end(self, metadata: Dict):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

        assert self.end_event is None
        self.end_event = end_event
        self.metadata = metadata

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)
