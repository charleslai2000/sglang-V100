import argparse, json, logging, numpy as np, torch
from sglang.srt.server_args import ServerArgs
from sglang.bench_one_batch import BenchArgs
import sglang.bench_one_batch as bench
from sglang.srt.layers.attention.mamba.ops import initialize_mamba_selective_state_update_backend
_orig_init=bench.initialize_moe_config
def _init(s): _orig_init(s); initialize_mamba_selective_state_update_backend(s)
bench.initialize_moe_config=_init
from sglang.srt.utils.device_timer import DeviceTimer
# Production scheduler RIDs are strings; standalone bench constructs integer RIDs.
_mr=bench
import sglang.srt.model_executor.model_runner as _model_runner_module
_build_span=_model_runner_module._build_step_span_name
def _safe_step_span(fb):
 if getattr(fb,'rids',None): fb.rids=[str(x) for x in fb.rids]
 return _build_span(fb)
_model_runner_module._build_step_span_name=_safe_step_span
bench.TreeCacheNamespace.available_and_evictable_str=lambda self: "standalone no-radix cache"
bench.TreeCacheNamespace.pretty_print=lambda self: None
_prepare=bench.ScheduleBatch.prepare_for_extend
def _prepare_resolve(self):
 _prepare(self)
 if self.input_ids is None and self.prefill_input_ids_cpu is not None:
  self.input_ids=self.prefill_input_ids_cpu.to(self.device,non_blocking=True); self.prefill_input_ids_cpu=None
bench.ScheduleBatch.prepare_for_extend=_prepare_resolve
records=[]
class Timer(DeviceTimer):
 def __init__(self): super().__init__(lambda **kw: records.append(kw))
 def wrap(self,metadata):
  if metadata.get('category')=='decode': return super().wrap(metadata)
  from contextlib import nullcontext
  return nullcontext()
_origload=bench.load_model
def load(*a,**kw):
 runner,tok=_origload(*a,**kw); raw=getattr(runner,'torch_runner',runner); raw.device_timer=Timer(); return runner,tok
bench.load_model=load
_orig=bench.prepare_synthetic_inputs_for_latency_test
def prep(bs,n,custom_inputs=None):
 if custom_inputs is None:
  base=np.asarray(json.load(open(f'/mnt/data/pc01/sglang-v100-migration/experiments/E037/fixtures/prompt-b{bs}-128.json')),dtype=np.int32)
  custom_inputs=np.tile(base,(1,(n+127)//128))[:,:n].tolist()
 return _orig(bs,n,custom_inputs)
bench.prepare_synthetic_inputs_for_latency_test=prep
p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);BenchArgs.add_cli_args(p);a=p.parse_args();s=ServerArgs.from_cli_args(a);b=BenchArgs.from_cli_args(a)
logging.basicConfig(level=getattr(logging,s.log_level.upper()),format='%(message)s')
try: bench.main(s,b)
finally: print(json.dumps({'e037_graph_decode_cuda_events':records}),flush=True)
