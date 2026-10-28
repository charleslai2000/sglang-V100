#!/usr/bin/env python3
"""Compare pinned WMMA with production SM70 extend Triton and FP32 oracle."""
import sys,importlib.util,torch
sys.path.insert(0,'experiments/G002-N/vendor/flash-attention-v100');sys.path.insert(0,'experiments/G002-N/vendor/flash-attention-v100/build/lib.linux-x86_64-cpython-312');sys.path.insert(0,'python')
spec=importlib.util.spec_from_file_location('g002n_wmma_prefill','experiments/G002-N/vendor/flash-attention-v100/build/lib.linux-x86_64-cpython-312/g002n_wmma_prefill.cpython-312-x86_64-linux-gnu.so');mod=importlib.util.module_from_spec(spec);sys.modules['g002n_wmma_prefill']=mod;spec.loader.exec_module(mod);sys.modules['flash_attn_v100_cuda']=mod
from flash_attn_v100.flash_attn_interface import flash_attn_prefill_paged
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
DEV='cuda'
def ref(q,k,v):
 qh=q.shape[1];q=q.float().permute(1,0,2);k=k.float().permute(1,0,2).repeat_interleave(qh//k.shape[1],0);v=v.float().permute(1,0,2).repeat_interleave(qh//v.shape[1],0)
 s=torch.einsum('hmd,hnd->hmn',q,k);p=torch.arange(q.shape[1],device=DEV);s.masked_fill_(p[:,None]<p[None,:],-torch.inf);return torch.einsum('hmn,hnd->hmd',s.softmax(-1),v).permute(1,0,2)
def stat(a,b):
 d=(a.float()-b.float()).abs();return (d.max().item(),d.mean().item(),torch.quantile(d.flatten(),.99).item())
for length in (1024,4096):
 c=torch.load(f'/data/t011/artifacts/g002n/capture-{length}.pt',map_location=DEV,weights_only=False);rs=c['attention_inputs'];pages=(length+15)//16;bt=torch.arange(pages,device=DEV,dtype=torch.int32).view(1,-1);lens=torch.tensor([length],device=DEV,dtype=torch.int32)
 for i,r in rs.items():
  owner=next(x for x in rs.values() if x['layer_id']==r['layer_id'] and not x['shared']);q=r['q'].contiguous();k=owner['k'].contiguous();v=owner['v'].contiguous();rr=ref(q,k,v)
  ck=k.reshape(pages,16,2,512).contiguous();cv=v.reshape(pages,16,2,512).contiguous();w=flash_attn_prefill_paged(q.unsqueeze(0),ck,cv,bt,lens,softmax_scale=1.,kv_cache_dtype='auto',causal=True,window=-1).squeeze(0)
  o=torch.empty_like(q); ind0=torch.tensor([0,length],device=DEV,dtype=torch.int32); indp=torch.tensor([0,0],device=DEV,dtype=torch.int32); empty=torch.empty((0,),device=DEV,dtype=torch.int32); dummy=torch.empty((1,2,512),device=DEV,dtype=torch.float16)
  extend_attention_fwd(q,k,v,o,dummy,dummy,ind0,indp,empty,None,True,None,length,1.,1.,sm_scale=1.,sliding_window_size=-1)
  torch.cuda.synchronize();print(length,i,'shared',r['shared'],'owner',r['layer_id'],'WMMA_FP32',stat(w,rr),'Triton_FP32',stat(o,rr),'WMMA_Triton',stat(w,o),flush=True)
