#!/usr/bin/env python3
"""Strict oracle, scratch/Graph tests and uninstrumented full-call timing."""
import argparse, hashlib, importlib.util, json, statistics, sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT=Path('/home/charles/Workspaces/3rdparty/sglang-V100')
E=ROOT/'experiments/E040-g022-c57-cta-warp-sliced-k'
FIXROOT=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures')
C57=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so')
C57SHA='c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9'
CAND=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E040/build/e040_warp_sliced_k.so')
CANDSHA='1da0e763db534fc9faa201c352047700dd56a17e11e1d1efe81416f828d45857'
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
sp=importlib.util.spec_from_file_location('e040_lock',ROOT/'experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py');L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_fixture(fixture):
 d=FIXROOT/fixture;m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def ld(n,dtype):
  p=d/(n+'.bin'); assert sha(p)==t[n]['sha256'],(n,sha(p),t[n]['sha256'])
  count=p.stat().st_size//torch.empty((),dtype=dtype).element_size()
  return torch.from_file(str(p),shared=False,size=count,dtype=dtype).view(*t[n]['shape']).cuda().contiguous()
 return d,m,ld('x',torch.float16),ld('qweight',torch.int32),ld('qzeros',torch.int32),ld('scales',torch.float16),ld('g_idx',torch.int32).view(-1)
def oracle(x,qw,qz,sc,g):
 M,K=x.shape;N=qw.shape[1];G=qz.shape[0]
 qi=torch.arange(K,device='cuda',dtype=torch.int64)
 codes=((qw.to(torch.int64).repeat_interleave(8,0) >> ((qi%8)*4).view(K,1))&15).double()
 zi=torch.arange(N,device='cuda',dtype=torch.int64)
 zeros=torch.empty((G,N),device='cuda',dtype=torch.float64)
 for b in range(8):zeros[:,b::8]=((qz.to(torch.int64)>>(4*b))&15).double()
 gi=g.long();w=(codes-(zeros.index_select(0,gi)+1))*sc.double().index_select(0,gi)
 return (x.double()@w).float()
def stats(y,r):
 err=(y-r).abs();tol=1e-4+5e-5*r.abs();return {'strict_bad':int((err>tol).sum()),'max_abs':float(err.max()),'max_ratio':float((err/tol).max()),'finite':bool(torch.isfinite(y).all())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',required=True);ap.add_argument('--M',type=int,required=True);ap.add_argument('--iters',type=int,default=50);ap.add_argument('--candidate',choices=['warp2','warp4','c57'],default='warp2');ap.add_argument('--stress',action='store_true');ap.add_argument('--no-bench',action='store_true');args=ap.parse_args()
 if sha(C57)!=C57SHA or (args.candidate!='c57' and sha(CAND)!=CANDSHA):raise SystemExit('binary SHA mismatch')
 L.assert_target_idle(UUID);fd=L.acquire(UUID)
 try:
  L.assert_cuda_uuid(torch,UUID)
  d,manifest,x0,qw,qz,sc,g=load_fixture(args.fixture);x=x0[:args.M].contiguous()
  if args.stress:x=(x*4).half().contiguous()
  cmod=load(name='e040_warp_sliced_k',sources=[str(E/'warp_sliced_k.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E040/build',verbose=False)
  import importlib.machinery
  cm=importlib.machinery.ExtensionFileLoader('common_ops',str(C57)).load_module()
  if args.candidate=='c57': fn=lambda xx:torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(xx,qw,qz,sc,g,True)
  else:
   op=getattr(cmod,args.candidate)
   fn=lambda xx:op(xx,qw,qz,sc,g,True)
  ref=fn(x).detach().clone() if args.candidate=='c57' else oracle(x,qw,qz,sc,g)
  torch.cuda.synchronize()
  y=fn(x);torch.cuda.synchronize();base=stats(y.float(),ref)
  y2=fn(x.clone().contiguous());torch.cuda.synchronize();repeat=torch.equal(y,y2)
  graph=torch.cuda.CUDAGraph();gx=x.clone().contiguous()
  with torch.cuda.graph(graph):gy=fn(gx)
  graph.replay();torch.cuda.synchronize();graph_stats=stats(gy.float(),ref)
  independent=fn(x.clone().contiguous());torch.cuda.synchronize();independent_equal=torch.equal(y,independent)
  cast=y.half();graph_cast=gy.half();cast_finite=bool(torch.isfinite(cast).all());cast_equal=torch.equal(cast,graph_cast)
  eager=ge=None
  if not args.no_bench:
   def bench(call):
    for _ in range(12):call()
    torch.cuda.synchronize();a=[]
    for _ in range(args.iters):
     s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();call();e.record();e.synchronize();a.append(s.elapsed_time(e)*1000)
    return sorted(a)
   eager=bench(lambda:fn(x).half());gg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(gg): gout=fn(x).half()
   gg.replay();torch.cuda.synchronize();ge=bench(gg.replay)
  result={'event':'E040_WARP_SLICE_FIXTURE','arm':args.candidate,'fixture':args.fixture,'M':args.M,'stress_x4':args.stress,'K':x.shape[1],'N':qw.shape[1],'grid':[qw.shape[1]//128,1,x.shape[1]//128],'threads':{'x':32,'y':2 if args.candidate=='warp2' else 4 if args.candidate=='warp4' else 1},'c57_sha256':C57SHA,'candidate_sha256':CANDSHA,'fixture_manifest_sha256':sha(d/'manifest.json'),'output_dtype':str(y.dtype),'eager':base,'graph':graph_stats,'repeat_equal':repeat,'independent_equal':independent_equal,'cast_finite':cast_finite,'graph_cast_equal':cast_equal,'eager_us':eager,'graph_us':ge,'eager_median_us':None if not eager else eager[len(eager)//2],'graph_median_us':None if not ge else ge[len(ge)//2]}
  print(json.dumps(result,sort_keys=True),flush=True)
 finally:L.release(fd)
if __name__=='__main__':main()
