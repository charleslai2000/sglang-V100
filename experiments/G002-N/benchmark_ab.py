#!/usr/bin/env python3
"""Interleaved CUDA Event kernel and end-to-end seven-layer attention A/B."""
import sys,importlib.util,torch
sys.path.insert(0,'experiments/G002-N/vendor/flash-attention-v100');sys.path.insert(0,'experiments/G002-N/vendor/flash-attention-v100/build/lib.linux-x86_64-cpython-312');sys.path.insert(0,'python')
spec=importlib.util.spec_from_file_location('g002n_wmma_prefill','experiments/G002-N/vendor/flash-attention-v100/build/lib.linux-x86_64-cpython-312/g002n_wmma_prefill.cpython-312-x86_64-linux-gnu.so');mod=importlib.util.module_from_spec(spec);sys.modules['g002n_wmma_prefill']=mod;spec.loader.exec_module(mod);sys.modules['flash_attn_v100_cuda']=mod
from flash_attn_v100.flash_attn_interface import flash_attn_prefill_paged
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
D='cuda'
def call(which,r,L,prepared=None):
 q,k,v=r['q'],r['k'],r['v']; owner_k,owner_v=k,v
 if r['shared']: owner_k,owner_v=prepared['owner_k'],prepared['owner_v']
 if which=='triton':
  qo=torch.tensor([0,L],device=D,dtype=torch.int32);kp=torch.tensor([0,0],device=D,dtype=torch.int32);ki=torch.empty(0,device=D,dtype=torch.int32);dummy=torch.empty((1,2,512),device=D,dtype=q.dtype);o=torch.empty_like(q)
  extend_attention_fwd(q,owner_k,owner_v,o,dummy,dummy,qo,kp,ki,None,True,None,L,1.,1.,sm_scale=1.)
 else:
  pages=L//16; qp=q.unsqueeze(0).contiguous(); ck=owner_k.reshape(pages,16,2,512).contiguous();cv=owner_v.reshape(pages,16,2,512).contiguous();bt=torch.arange(pages,device=D,dtype=torch.int32)[None,:];sl=torch.tensor([L],device=D,dtype=torch.int32)
  flash_attn_prefill_paged(qp,ck,cv,bt,sl,softmax_scale=1.,kv_cache_dtype='auto',causal=True,window=-1)

def elapsed(fn,n=12):
 fn();torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);vals=[]
 for _ in range(n):
  a.record();fn();b.record();b.synchronize();vals.append(a.elapsed_time(b))
 return sorted(vals)[len(vals)//2]
for L in (1024,4096):
 c=torch.load(f'/data/t011/artifacts/g002n/capture-{L}.pt',map_location=D,weights_only=False);rs=c['attention_inputs'];results={'triton':[],'wmma':[],'prep':[]}
 for r in rs.values():
  owner=next(x for x in rs.values() if x['layer_id']==r['layer_id'] and not x['shared']);pr={'owner_k':owner['k'],'owner_v':owner['v']}
  for which in ('triton','wmma'):
   # Warm compile and report per-layer median over interleaved alternating samples.
   call(which,r,L,pr);torch.cuda.synchronize();times=[]
   for _ in range(8):
    for w in ('triton','wmma'):
     if w==which:times.append(elapsed(lambda:call(w,r,L,pr),3))
   results[which].append(sorted(times)[len(times)//2])
  # WMMA preparation includes Q transpose/packing, KV page-layout materialization,
  # block table and sequence metadata creation; separately measured.
  def prep():
   _=r['q'].unsqueeze(0).contiguous();_=owner['k'].reshape(L//16,16,2,512).contiguous();_=owner['v'].reshape(L//16,16,2,512).contiguous();_=torch.arange(L//16,device=D,dtype=torch.int32)[None,:];_=torch.tensor([L],device=D,dtype=torch.int32)
  results['prep'].append(elapsed(prep,10))
 print('L',L,'per_layer_ms',results,'seven_sum_ms',{k:sum(v) for k,v in results.items()},'net_ms',sum(results['triton'])-sum(results['wmma'])-sum(results['prep']),flush=True)
