#!/usr/bin/env python3
"""Exact Falcon checkpoint-weight probe for Mamba in_proj (not a serving benchmark)."""
import argparse, hashlib, json, os, time
from pathlib import Path
import torch
from safetensors import safe_open
from types import SimpleNamespace
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

P=argparse.ArgumentParser(); P.add_argument('--rows',type=int,choices=[1,2,4],required=True);P.add_argument('--route',choices=['baseline','candidate'],required=True);P.add_argument('--iterations',type=int,default=80);P.add_argument('--report',required=True);a=P.parse_args()
os.environ['SGLANG_SM70_FALCON_MAMBA_INPROJ_SMALLM']='1' if a.route=='candidate' else '0'
model=Path('/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'); shard=model/'model-00001-of-00002.safetensors';key='model.layers.0.mamba.in_proj.weight'
with safe_open(str(shard),framework='pt',device='cpu') as f: w=f.get_tensor(key).to(torch.float16).contiguous().cuda()
torch.manual_seed(20261014+a.rows);x=torch.randn(a.rows,3072,dtype=torch.float16,device='cuda').contiguous();bias=None
layer=SimpleNamespace(prefix='model.layers.0.mixer.in_proj',weight=w)
method=UnquantizedLinearMethod()
# Assert candidate route decision is exact; baseline explicitly disables it.
def call(): return method.apply(layer,x,bias)
with torch.no_grad():
 y=call();torch.cuda.synchronize(); assert torch.isfinite(y).all()
 ref=torch.nn.functional.linear(x,w,bias);torch.cuda.synchronize(); delta=(y.float()-ref.float()).abs()
 torch.testing.assert_close(y,ref,rtol=0.002,atol=0.001)
 # CUDA Event microbenchmark, paired iterations not NCU or service timing.
 for _ in range(10):call()
 torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);samples=[]
 for _ in range(a.iterations):
  start.record();call();end.record();end.synchronize();samples.append(start.elapsed_time(end))
 samples.sort();median=samples[len(samples)//2]
 # One launch for filtered NCU capture, outside timing loop.
 call();torch.cuda.synchronize()
result={'route':a.route,'rows':a.rows,'K':3072,'N':6680,'input_dtype':str(x.dtype),'weight_source':str(shard),'weight_key':key,'weight_source_dtype':'bfloat16','runtime_weight_dtype':str(w.dtype),'weight_shape':list(w.shape),'weight_sha256':hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest(),'SGLANG_SM70_FALCON_MAMBA_INPROJ_SMALLM':os.environ['SGLANG_SM70_FALCON_MAMBA_INPROJ_SMALLM'],'iterations':a.iterations,'median_cuda_event_ms':median,'min_cuda_event_ms':samples[0],'p90_cuda_event_ms':samples[int(.9*(len(samples)-1))],'max_abs_vs_torch_linear':float(delta.max()),'relative_l2_vs_torch_linear':float(torch.linalg.vector_norm(y.float()-ref.float())/torch.linalg.vector_norm(ref.float())),'finite':bool(torch.isfinite(y).all()),'torch_linear_close':True,'torch_cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name()}
Path(a.report).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True),flush=True)
