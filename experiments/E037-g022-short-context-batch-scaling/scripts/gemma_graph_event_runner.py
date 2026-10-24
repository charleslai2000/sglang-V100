#!/usr/bin/env python3
"""E037's isolated ModelRunner/Event harness adapted to Gemma4; no new bench framework."""
import argparse, json, logging, numpy as np, torch
from sglang.srt.server_args import ServerArgs
from sglang.bench_one_batch import BenchArgs
import sglang.bench_one_batch as bench
from sglang.srt.utils.device_timer import DeviceTimer

# Gemma model code has no Falcon Mamba selective-state initialization.
from sglang.srt.model_executor import model_runner as model_runner_module
_original_step_span = model_runner_module._build_step_span_name
def _safe_step_span(batch):
    if getattr(batch, "rids", None):
        batch.rids = [str(x) for x in batch.rids]
    return _original_step_span(batch)
model_runner_module._build_step_span_name = _safe_step_span

# Reproduce E037's standalone prefill input resolution and keep it scoped to
# this runner; Gemma4's multimodal wrapper requires a non-None input_ids.
_original_prepare = bench.ScheduleBatch.prepare_for_extend
def _prepare_resolve(self):
    _original_prepare(self)
    if self.input_ids is None and self.prefill_input_ids_cpu is not None:
        self.input_ids = self.prefill_input_ids_cpu.to(self.device, non_blocking=True)
        self.prefill_input_ids_cpu = None
bench.ScheduleBatch.prepare_for_extend = _prepare_resolve
bench.TreeCacheNamespace.available_and_evictable_str = lambda self: "standalone no-radix cache"
bench.TreeCacheNamespace.pretty_print = lambda self: None

records=[]
class DecodeTimer(DeviceTimer):
    def __init__(self): super().__init__(lambda **row: records.append(row))
    def wrap(self, metadata):
        if metadata.get("category") == "decode": return super().wrap(metadata)
        from contextlib import nullcontext
        return nullcontext()
_original_load=bench.load_model
def _load(*args, **kwargs):
    runner, tokenizer = _original_load(*args, **kwargs)
    raw = getattr(runner, "torch_runner", runner)
    raw.device_timer = DecodeTimer()
    return runner, tokenizer
bench.load_model=_load

# Fixed deterministic tokenizer-valid token IDs, passed through E037's existing
# synthetic-input seam; the actual prompt is token IDs as in ModelRunner bench.
def _inputs(bs, length, custom_inputs=None):
    if custom_inputs is None:
        rng=np.random.default_rng(37002+bs)
        custom_inputs=rng.integers(100, 10000, size=(bs,length), dtype=np.int32).tolist()
    return _original_inputs(bs,length,custom_inputs)
_original_inputs=bench.prepare_synthetic_inputs_for_latency_test
bench.prepare_synthetic_inputs_for_latency_test=_inputs

p=argparse.ArgumentParser(); ServerArgs.add_cli_args(p); BenchArgs.add_cli_args(p)
a=p.parse_args(); s=ServerArgs.from_cli_args(a); b=BenchArgs.from_cli_args(a)
logging.basicConfig(level=getattr(logging,s.log_level.upper()),format='%(message)s')
try:
    bench.main(s,b)
finally:
    print(json.dumps({'gemma_graph_decode_cuda_events':records}),flush=True)
