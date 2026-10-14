#!/usr/bin/env python3
"""Interleaved complete GPTQ op A/B via the isolated production dispatcher."""
import argparse,hashlib,json,os,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--iters',type=int,default=200);p.add_argument('--candidate-module',required=True);a=p.parse_args()
from importlib.machinery import ExtensionFileLoader
ExtensionFileLoader('common_ops',a.candidate_module).load_module()
r=Path(a.fixture);m=json.load(open(r/'manifest.json'))
def t(n,d):
 q=m['tensors'][n];f=r/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(*q['shape']).cuda()
x=t('x',torch.float16);w=t('qweight',torch.uint32);z=t('qzeros',torch.uint32);s=t('scales',torch.float16);g=t('g_idx',torch.int32)
def run(candidate):
 if candidate:os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='1'
 else:os.environ.pop('SGLANG_G017_Q4_FP32_CANDIDATE',None)
 if candidate:
  y=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,w.view(torch.int32),z.view(torch.int32),s,g,False)
 else:
  # Prevent the candidate build's fail-closed dispatcher from treating the
  # baseline measurement as an accidental candidate fallback.
  os.environ.pop('SGLANG_G017_Q4_FP32_CANDIDATE',None)
  y=torch.ops.sgl_kernel.gptq_gemm(x,w.view(torch.int32),z.view(torch.int32),s,g,False,4)
 # Match production GPTQLinearKernel's final output dtype adaptation.
 return y.to(x.dtype) if y.dtype!=x.dtype else y
# Disable device-side probe during timing; a separate server/profile run records dispatch.
os.environ['SGLANG_G017_Q4_FP32_CANDIDATE_PROBE']='0'
for _ in range(10):run(False);run(True)
torch.cuda.synchronize();times={False:[],True:[]};ys={}
for i in range(a.iters):
 for mode in ((False,True) if i%2==0 else (True,False)):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();y=run(mode);en.record();en.synchronize();times[mode].append(st.elapsed_time(en));ys[mode]=y
out={'fixture':m['prefix'],'candidate_module_sha256':hashlib.sha256(open(a.candidate_module,'rb').read()).hexdigest(),'iters_each':a.iters,'s1_fallback':{},'candidate_dispatch':{}}
for mode,key in ((False,'s1_fallback'),(True,'candidate_dispatch')):
 q=torch.tensor(times[mode]);y=ys[mode];out[key]={'median_ms':float(q.median()),'p10_ms':float(torch.quantile(q,.1)),'p90_ms':float(torch.quantile(q,.9)),'dtype':str(y.dtype),'finite':bool(torch.isfinite(y).all())}
out['post_cast_outputs_equal']=torch.equal(ys[False],ys[True]);out['post_cast_max_abs_delta']=float((ys[False].float()-ys[True].float()).abs().max())
print(json.dumps(out))
