#!/usr/bin/env python3
import argparse, hashlib, importlib.machinery, json, os, sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
ROOT=Path('/home/charles/Workspaces/3rdparty/sglang-V100');E=ROOT/'experiments/E041-g022-c57-n-tile-parallelism'
FIXROOT=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures')
C57=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so'); C57SHA='c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9'
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
spec=__import__('importlib').util.spec_from_file_location('e041_lock',ROOT/'experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py');L=__import__('importlib').util.module_from_spec(spec);spec.loader.exec_module(L)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadf(name):
 d=FIXROOT/name;m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def ld(n,dt):
  p=d/(n+'.bin');assert sha(p)==t[n]['sha256'];return torch.from_file(str(p),shared=False,size=p.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(*t[n]['shape']).cuda().contiguous()
 return d,ld('x',torch.float16),ld('qweight',torch.int32),ld('qzeros',torch.int32),ld('scales',torch.float16),ld('g_idx',torch.int32).view(-1)
def oracle(x,qw,qz,sc,g):
 M,K=x.shape;N=qw.shape[1];G=qz.shape[0];ki=torch.arange(K,device='cuda',dtype=torch.int64)
 codes=((qw.long().repeat_interleave(8,0)>>(4*(ki%8)).view(K,1))&15).double();z=torch.empty((G,N),device='cuda',dtype=torch.float64)
 for b in range(8):z[:,b::8]=((qz.long()>>(4*b))&15).double()
 w=(codes-(z.index_select(0,g.long())+1))*sc.double().index_select(0,g.long());return (x.double()@w).float()
def stat(y,r):
 e=(y.float()-r).abs();t=1e-4+5e-5*r.abs();return {'strict_bad':int((e>t).sum()),'max_abs':float(e.max()),'max_ratio':float((e/t).max()),'finite':bool(torch.isfinite(y).all())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--M',type=int,required=True);p.add_argument('--arm',choices=['ntile128','ntile64','ntile32','c57'],required=True);p.add_argument('--ab',action='store_true');p.add_argument('--no-bench',action='store_true');a=p.parse_args()
 if sha(C57)!=C57SHA:raise SystemExit('c57 sha mismatch')
 L.assert_target_idle(UUID);fd=L.acquire(UUID)
 try:
  L.assert_cuda_uuid(torch,UUID);d,x0,qw,qz,sc,g=loadf(a.fixture);x=x0[:a.M].contiguous()
  mod=load(name='e041_ntile',sources=[str(E/'ntile.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E041/build',verbose=False)
  common_module=importlib.machinery.ExtensionFileLoader('common_ops',str(C57)).load_module()
  fn=(lambda xx:torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(xx,qw,qz,sc,g,False)) if a.arm=='c57' else (lambda xx:getattr(mod,a.arm)(xx,qw,qz,sc,g))
  # The NTile diagnostic op exposes the pre-cast FP32 reducer output; timed op returns native FP16.
  if not a.ab and a.arm!='c57':
   def fp32op(xx):
    # Dedicated diagnostic binding below; no cast is permitted before oracle comparison.
    return getattr(mod,a.arm+'_fp32')(xx,qw,qz,sc,g)
   fn=fp32op
  elif a.ab and a.arm!='c57': fn=lambda xx:getattr(mod,a.arm)(xx,qw,qz,sc,g)
  y=fn(x);torch.cuda.synchronize();rep=fn(x.clone());torch.cuda.synchronize();gph=torch.cuda.CUDAGraph();gx=x.clone()
  with torch.cuda.graph(gph):gy=fn(gx)
  gph.replay();torch.cuda.synchronize();r=oracle(x,qw,qz,sc,g)
  try: ref=(torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,sc,g,True) if a.arm=='c57' else mod.ntile128_fp32(x,qw,qz,sc,g))
  except Exception: ref=None
  out={'arm':a.arm,'fixture':a.fixture,'M':a.M,'candidate_output_sha256':hashlib.sha256(y.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'grid':[qw.size(1)//(128 if a.arm in ('ntile128','c57') else 64 if a.arm=='ntile64' else 32),1,x.size(1)//128],'threads':32,'input_shape':list(x.shape),'input_dtype':str(x.dtype),'input_stride':list(x.stride()),'weight_shape':list(qw.shape),'zero_shape':list(qz.shape),'scale_shape':list(sc.shape),'gidx_shape':list(g.shape),'output_shape':list(y.shape),'output_dtype':str(y.dtype),'output_stride':list(y.stride()),'output_is_fp16_roundtrip':int((y==y.half().float()).sum()),'output_elements':y.numel(),'output_sha256':hashlib.sha256(y.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'manifest_sha256':sha(d/'manifest.json'),'tensor_sha256':{n:json.loads((d/'manifest.json').read_text())['tensors'][n]['sha256'] for n in ('x','qweight','qzeros','scales','g_idx')},'sha':sha(E/'ntile.cu'),'c57_sha':C57SHA,'route':('frozen_exact_c57_gptq_gemm_use_shuffle_false_bit4' if a.arm=='c57' else ('e041_ntile_fp32_diagnostic' if not a.ab else 'e041_ntile_native_fp16')),'return_fp32':(True if not a.ab and a.arm!='c57' else False),'oracle':stat(y,r),'graph_oracle':stat(gy,r),'repeat_equal':torch.equal(y,rep),'graph_equal':torch.equal(y,gy),'strict_vs_c57':None if ref is None else stat(y,ref),'precast_oracle_output_dtype':str(ref.dtype) if ref is not None else None,'dtype':str(y.dtype)}
  if not a.no_bench:
   def bench(f):
    for _ in range(10):f()
    torch.cuda.synchronize();es=[]
    for _ in range(30):
     s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();f();e.record();e.synchronize();es.append(s.elapsed_time(e)*1000)
    return sorted(es)
   out['eager_us']=bench(lambda:fn(x));graph2=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph2):go=fn(x)
   graph2.replay();torch.cuda.synchronize();out['graph_replay_us']=bench(graph2.replay)
  print(json.dumps(out,sort_keys=True))
 finally:L.release(fd)
if __name__=='__main__':main()
