#!/usr/bin/env python3
"""E038 fixture correctness: named isolated op plus strict oracle/Graph/4x stress."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
import importlib.util
_lock_path=Path('/home/charles/Workspaces/3rdparty/sglang-V100/experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py')
_spec=importlib.util.spec_from_file_location('e035_lock_reuse',_lock_path);_lock=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_lock)
assert_target_idle,acquire,release,assert_cuda_uuid=_lock.assert_target_idle,_lock.acquire,_lock.release,_lock.assert_cuda_uuid
ap=argparse.ArgumentParser();ap.add_argument('--fixture-root',default='/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');ap.add_argument('--fixture',required=True);ap.add_argument('--M',type=int,required=True);ap.add_argument('--iters',type=int,default=20);a=ap.parse_args()
ROOT=Path(a.fixture_root);d=ROOT/a.fixture;man=json.loads((d/'manifest.json').read_text());tt=man['tensors'];UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';assert_target_idle(UUID);fd=acquire(UUID)
try:
 assert_cuda_uuid(torch,UUID)
 extpath=Path(__file__).with_name('build_extension.py')
 src=Path(__file__).with_name('actorder_group_q4_simt.cu')
 mod=load(name='e038_group_major_simt',sources=[str(src)],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E038/build',verbose=False)
 binary=Path(mod.__file__).resolve(); bsha=hashlib.sha256(binary.read_bytes()).hexdigest()
 def ld(n,dt):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(tt[n]['shape']).cuda().contiguous()
 x0=ld('x',torch.float16);x=x0[:,:] if a.M==1 else x0[:a.M].contiguous();qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);gi=ld('g_idx',torch.int32).reshape(-1);K,N=x.shape[1],qw.shape[1];G=qz.shape[0]
 # Independent oracle from exact original packed codes and GPTQ-v1 qzero+1.
 codes=((qw.to(torch.int64).repeat_interleave(8,dim=0) >> ((torch.arange(K,device='cuda',dtype=torch.int64)%8)*4).view(K,1))&15).to(torch.int32)
 zeros=torch.empty((G,N),device='cuda',dtype=torch.int32)
 for j in range(8):zeros[:,j::8]=(qz>>(4*j))&15
 w64=(codes.double()-(zeros[gi.long()].double()+1))*sc.double()[gi.long()]
 ref=x.double()@w64;ref=ref.float();tol=1e-4+5e-5*ref.abs()
 qgm,perm,_,_=mod.group_major_pack(qw,qz,sc,gi,128);xp=mod.permute_activation_group_major(x,perm,128).contiguous()
 def call(xx=xp):return mod.group_major_fp32(xx,qgm,qz,sc,128,4)
 def check(y):
  err=(y-ref).abs();ratio=err/tol;bad=err>tol;ii=bad.nonzero().cpu().tolist();return {'strict_bad':int(bad.sum()),'max_abs':float(err.max()),'max_tolerance_ratio':float(ratio.max()),'finite':bool(torch.isfinite(y).all()),'bad_points':[{'m':i,'n':j,'actual':float(y[i,j]),'reference':float(ref[i,j]),'error':float(err[i,j]),'allowed':float(tol[i,j])} for i,j in ii]}
 y=call();torch.cuda.synchronize();ev=check(y)
 # Include per-invocation group-major activation gather in the complete operator path.
 def full_call(xx=x):return mod.group_major_fp32(mod.permute_activation_group_major(xx,perm,128).contiguous(),qgm,qz,sc,128,4)
 # repeat and independent input/scratch
 yr=call();yi=call(xp.clone().contiguous());torch.cuda.synchronize();repeat_equal=torch.equal(y,yr);ind_equal=torch.equal(y,yi)
 graph=torch.cuda.CUDAGraph();gx=xp.clone().contiguous()
 with torch.cuda.graph(graph): gy=call(gx)
 graph.replay();torch.cuda.synchronize();gv=check(gy);graph_exact=torch.equal(gy,y)
 cast=y.half();graph_cast=gy.half();cast_finite=bool(torch.isfinite(cast).all());cast_graph_equal=torch.equal(cast,graph_cast)
 stress=None
 if a.fixture=='43-mlp-down_proj-call0000' and a.M==4:
  xs=(x*4).half().contiguous();rs=xs.double()@w64;rs=rs.float();ts=1e-4+5e-5*rs.abs();xps=mod.permute_activation_group_major(xs,perm,128).contiguous();ys=mod.group_major_fp32(xps,qgm,qz,sc,128,4);err=(ys-rs).abs();rat=err/ts;mask=err>ts;stress={'strict_bad':int(mask.sum()),'max_abs':float(err.max()),'max_tolerance_ratio':float(rat.max()),'input_max_abs':float(xs.abs().max()),'input_sha256':hashlib.sha256(xs.cpu().numpy().tobytes()).hexdigest(),'output_sha256':hashlib.sha256(ys.cpu().numpy().tobytes()).hexdigest(),'reference_sha256':hashlib.sha256(rs.cpu().numpy().tobytes()).hexdigest(),'bad_points':[{'m':i,'n':j,'actual':float(ys[i,j]),'reference':float(rs[i,j]),'error':float(err[i,j]),'allowed':float(ts[i,j])} for i,j in mask.nonzero().cpu().tolist()]}
 if ev['strict_bad'] or gv['strict_bad'] or (stress and stress['strict_bad']):
  print(json.dumps({'fixture':a.fixture,'M':a.M,'binary':str(binary),'binary_sha256':bsha,'eager':ev,'graph':gv,'stress':stress},indent=2));raise SystemExit(1)
 def bench(fn):
  for _ in range(5):fn()
  torch.cuda.synchronize();vals=[]
  for _ in range(a.iters):
   s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();fn();e.record();e.synchronize();vals.append(s.elapsed_time(e)*1000)
  return sorted(vals)
 eager=bench(call);gt=bench(graph.replay);full=bench(full_call)
 print(json.dumps({'fixture':a.fixture,'M':a.M,'K':K,'N':N,'binary':str(binary),'binary_sha256':bsha,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'eager':ev,'graph':gv,'repeat_equal':repeat_equal,'independent_scratch_equal':ind_equal,'graph_exact':graph_exact,'fp16_cast_finite':cast_finite,'graph_cast_equal':cast_graph_equal,'stress':stress,'eager_median_us':eager[len(eager)//2],'graph_median_us':gt[len(gt)//2],'full_operator_median_us':full[len(full)//2],'full_operator_samples_us':full,'eager_samples_us':eager,'graph_samples_us':gt,'pid':__import__('os').getpid()},indent=2))
finally:release(fd)
