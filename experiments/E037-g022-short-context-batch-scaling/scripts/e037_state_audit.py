#!/usr/bin/env python3
"""Targeted real SGLang ModelRunner hybrid request-state audit (not a performance test)."""
import argparse, json, logging, os
from pathlib import Path
import numpy as np
import torch
from sglang.srt.server_args import ServerArgs
from sglang.bench_one_batch import BenchArgs
import sglang.bench_one_batch as bench
from sglang.srt.layers.attention.mamba.ops import initialize_mamba_selective_state_update_backend
originit=bench.initialize_moe_config
def init(s): originit(s); initialize_mamba_selective_state_update_backend(s)
bench.initialize_moe_config=init
import sglang.srt.model_executor.model_runner as mr
origspan=mr._build_step_span_name
def safe_span(f):
    if getattr(f,'rids',None): f.rids=[str(x) for x in f.rids]
    return origspan(f)
mr._build_step_span_name=safe_span
bench.TreeCacheNamespace.available_and_evictable_str=lambda self:'audit radix disabled'
bench.TreeCacheNamespace.pretty_print=lambda self:None
prep0=bench.ScheduleBatch.prepare_for_extend
def prep(self):
    prep0(self)
    if self.input_ids is None and self.prefill_input_ids_cpu is not None:
        self.input_ids=self.prefill_input_ids_cpu.to(self.device,non_blocking=True); self.prefill_input_ids_cpu=None
bench.ScheduleBatch.prepare_for_extend=prep
origload=bench.load_model
out={'schema':'E037-state-audit-v1','cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'steps':[]}
def load(*a,**kw):
    runner,tok=origload(*a,**kw); raw=getattr(runner,'torch_runner',runner); out['model_runner_class']=type(raw).__qualname__
    origext=runner.extend; origdec=runner.decode
    def slots(batch):
        pool=raw.req_to_token_pool; reqidx=batch.req_pool_indices.detach().cpu().to(torch.int64); m=pool.get_mamba_indices(batch.req_pool_indices).detach().cpu().to(torch.int64)
        kv=pool.req_to_token[batch.req_pool_indices].detach().cpu().to(torch.int64); lens=batch.seq_lens.detach().cpu().to(torch.int64)
        return {'req_pool_indices':reqidx.tolist(),'mamba_indices':m.tolist(),'seq_lens':lens.tolist(),'kv_token_indices':[kv[i,:int(lens[i])].tolist() for i in range(len(lens))],'conv_ptrs':[int(x.data_ptr()) for x in pool.mamba_pool.mamba_cache.conv],'temporal_ptr':int(pool.mamba_pool.mamba_cache.temporal.data_ptr()),'conv_shape':[list(x.shape) for x in pool.mamba_pool.mamba_cache.conv],'temporal_shape':list(pool.mamba_pool.mamba_cache.temporal.shape),'kv_k_ptr':int(raw.token_to_kv_pool_allocator.get_kvcache().get_key_buffer(0).data_ptr()),'kv_v_ptr':int(raw.token_to_kv_pool_allocator.get_kvcache().get_value_buffer(0).data_ptr()),'graph_batch_sizes':getattr(raw.graph_runner,'bs',None),'graph_keys':sorted(getattr(raw.graph_runner,'graphs',{}).keys())}
    def extend(reqs):
        ids,logits,b=origext(reqs); torch.cuda.synchronize(); snap=slots(b); snap['phase']='post_prefill'; out['steps'].append(snap); return ids,logits,b
    def decode(ids,b):
        before=slots(b); result=origdec(ids,b); torch.cuda.synchronize(); after=slots(b)
        cache=raw.req_to_token_pool.mamba_pool.mamba_cache
        mids=after['mamba_indices']
        # Read only the actual request slots; never materialize a full-pool dtype copy.
        sums=[]
        for x in cache.conv:
            sums.append([float(x[:,i].float().abs().sum()) for i in mids])
        sums.append([float(cache.temporal[:,i].float().abs().sum()) for i in mids])
        step={'phase':'decode','before':before,'after':after,'output_token_ids':result[0].detach().cpu().reshape(-1).tolist(),'private_slot_state_abs_sums':sums}
        out['steps'].append(step); return result
    runner.extend=extend;runner.decode=decode;return runner,tok
bench.load_model=load
origprep=bench.prepare_synthetic_inputs_for_latency_test
def inputfix(bs,n,custom_inputs=None):
    if custom_inputs is None:
        f=Path(f'/mnt/data/pc01/sglang-v100-migration/experiments/E037/fixtures/prompt-b{bs}-128.json');base=np.asarray(json.loads(f.read_text()),dtype=np.int32);custom_inputs=np.tile(base,(1,(n+127)//128))[:,:n].tolist()
    return origprep(bs,n,custom_inputs)
bench.prepare_synthetic_inputs_for_latency_test=inputfix
p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);BenchArgs.add_cli_args(p);a=p.parse_args();s=ServerArgs.from_cli_args(a);b=BenchArgs.from_cli_args(a);logging.basicConfig(level=logging.INFO,format='%(message)s')
try: bench.main(s,b)
finally:
    path=Path(os.environ.get('E037_AUDIT_OUT','/tmp/e037-state-audit.json'));path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'audit_path':str(path),'steps':len(out['steps'])}),flush=True)
