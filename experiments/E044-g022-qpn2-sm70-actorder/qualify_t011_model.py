#!/usr/bin/env python3
"""Sequential opt-in model math/state probe for T011 vs exact-c57 route."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];MODEL=Path('/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--route',choices=('c57','t011'),required=True);p.add_argument('--out',required=True);a=p.parse_args()
 from sglang.srt.server_args import ServerArgs
 from sglang.srt.managers.tp_worker import TpModelWorker
 from sglang.srt.layers.attention.mamba.ops import initialize_mamba_selective_state_update_backend
 # Reuse the exact production ModelRunner/weight loader configuration used by the qualified service.
 sa=ServerArgs(model_path=str(MODEL),tokenizer_path=str(MODEL),trust_remote_code=True,dtype='float16',tp_size=1,pp_size=1,mem_fraction_static=0.70,max_running_requests=4,max_total_tokens=16384,attention_backend='tilelang_fa_v100',disable_radix_cache=True,disable_cuda_graph=False,cuda_graph_bs=[1,2,4],cuda_graph_max_bs=4,disable_piecewise_cuda_graph=True)
 initialize_mamba_selective_state_update_backend(sa)
 worker=TpModelWorker(server_args=sa,gpu_id=0,tp_rank=0,moe_ep_rank=0,pp_rank=0,attn_cp_rank=0,moe_dp_rank=0,dp_rank=None,nccl_port=29591)
 runner=worker.model_runner;model=runner.model.eval();tok=worker.tokenizer
 quant=[m for m in model.modules() if hasattr(m,'qweight') and hasattr(m,'g_idx')]
 print(json.dumps({'route':a.route,'quant_modules':len(quant),'loaded_cuda_uuid':str(torch.cuda.get_device_properties(0).uuid),'model_config':str(MODEL)},default=str),flush=True)
 # Use ModelRunner's production CUDA-graph dummy batch constructor for a valid forward context.
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 import inspect
 def logits_for(n):
  with torch.inference_mode():
   output=runner._dummy_run(batch_size=n)
  return output.logits_output.next_token_logits
 def tokens(n):
  ids=tok.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False);fill=tok.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
  while len(ids)<n:ids=fill+ids
  return ids[-n:]
 results=[]
 for n in (1024,4096):
  st=time.perf_counter();logits=logits_for(n);torch.cuda.synchronize();elapsed=time.perf_counter()-st
  logits=logits.detach().float().cpu();results.append({'prompt_tokens':n,'prefill_s':elapsed,'logits_sha256':hashlib.sha256(logits.numpy().tobytes()).hexdigest(),'logits_shape':list(logits.shape),'argmax_last':int(logits[0,-1].argmax())})
 for module in quant:
  w=module.qweight; g=module.g_idx
  if w.is_cuda and w.dtype==torch.int32 and w.ndim==2 and g.numel():
   gcpu=g.detach().cpu().numpy();nonidentity=not np.array_equal(gcpu,np.arange(len(gcpu),dtype=gcpu.dtype))
   break
 else:nonidentity=None
 out={'route':a.route,'model_path':str(MODEL),'model_config_sha256':sha(MODEL/'config.json'),'model_revision':'52036ae497a2c0c253e17f98cc8652f2f7082ae3','candidate_binary_sha256':sha('/mnt/data/sglang-V100/t011-build/qpn2_multiwarp.so'),'baseline_binary_sha256':sha('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so'),'target_uuid':str(torch.cuda.get_device_properties(0).uuid),'quant_module_count':len(quant),'nonidentity_g_idx':nonidentity,'outputs':results,'state_claim':'No multi-request state isolation performed by this diagnostic'}
 Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
