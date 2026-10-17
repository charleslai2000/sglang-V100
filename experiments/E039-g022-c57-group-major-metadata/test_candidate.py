#!/usr/bin/env python3
"""One E039 fixture/M strict oracle run; caller serializes processes."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
E=Path(__file__).resolve().parent;LOCK=Path('/home/charles/Workspaces/3rdparty/sglang-V100/experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py');sp=importlib.util.spec_from_file_location('e039_lock',LOCK);L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)
ap=argparse.ArgumentParser();ap.add_argument('--fixture',required=True);ap.add_argument('--M',type=int,required=True);ap.add_argument('--stress4',action='store_true');a=ap.parse_args();UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';L.assert_target_idle(UUID);fd=L.acquire(UUID)
try:
 assert torch.cuda.device_count()==1;L.assert_cuda_uuid(torch,UUID)
 mod=load(name='e039_c57_groupmajor',sources=[str(E/'actorder_groupmajor_c57.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E039/build',verbose=False)
 binary=Path(mod.__file__).resolve();bsha=hashlib.sha256(binary.read_bytes()).hexdigest();root=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');d=root/a.fixture;m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def ld(n,dt):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(t[n]['shape']).cuda().contiguous()
 x0=ld('x',torch.float16);x=x0[:a.M].contiguous();qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);gi=ld('g_idx',torch.int32).view(-1);K,N=x.shape[1],qw.shape[1];G=qz.shape[0]
 perm0=torch.argsort(gi,stable=True).int();qmajor,perm=mod.stable_groupmajor_pack(qw,gi,128)
 assert torch.equal(perm,perm0)
 xp=mod.groupmajor_activation(x,perm);assert torch.equal(xp,x[:,perm.long()])
 # Independent original-layout FP64 GPTQ-v1 oracle.
 codes=((qw.to(torch.int64).repeat_interleave(8,0)>>((torch.arange(K,device='cuda',dtype=torch.int64)%8)*4).view(K,1))&15).double();zeros=torch.empty((G,N),device='cuda',dtype=torch.int32)
 for j in range(8):zeros[:,j::8]=(qz>>(4*j))&15
 w=(codes-(zeros[gi.long()].double()+1))*sc.double()[gi.long()]
 xin=x
 if a.stress4:xin=(x*4).half().contiguous()
 ref=(xin.double()@w).float();tol=1e-4+5e-5*ref.abs();xpg=mod.groupmajor_activation(xin,perm)
 # CUDA Graph replay and standalone run are measured/tested separately; first eager result is the numerical authority.
 def fn(fp32=True):return mod.c57_groupmajor_gptq(xpg,qmajor,qz,sc,128,fp32)
 y=fn();torch.cuda.synchronize();assert y.dtype==torch.float32,f"frozen oracle requires pre-cast FP32, got {y.dtype}";err=(y.float()-ref).abs();bad=err>tol;ev={'strict_bad':int(bad.sum()),'max_abs':float(err.max()),'max_tolerance_ratio':float((err/tol).max()),'finite':bool(torch.isfinite(y).all())}
 if ev['strict_bad']:
  coords=bad.nonzero().cpu().tolist();payload={'failure':'eager strict oracle','fixture':a.fixture,'M':a.M,'stress4':a.stress4,'binary':str(binary),'binary_sha256':bsha,'source_sha256':hashlib.sha256((E/'actorder_groupmajor_c57.cu').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'details':ev,'bad_points':[[i,j,float(y[i,j]),float(ref[i,j]),float(err[i,j]),float(tol[i,j])] for i,j in coords]};Path('/mnt/data/pc01/sglang-v100-migration/experiments/E039/strict-failure-latest.json').write_text(json.dumps(payload,indent=2));raise RuntimeError(json.dumps(payload))
 yr=fn();yi=mod.c57_groupmajor_gptq(mod.groupmajor_activation(xin.clone().contiguous(),perm),qmajor,qz,sc,128,True);torch.cuda.synchronize();repeat=torch.equal(y,yr);ind=torch.equal(y,yi)
 cast=y.half();cast_finite=bool(torch.isfinite(cast).all())
 graph=torch.cuda.CUDAGraph();gx=mod.groupmajor_activation(xin.clone().contiguous(),perm)
 with torch.cuda.graph(graph): gy=mod.c57_groupmajor_gptq(gx,qmajor,qz,sc,128,True)
 graph.replay();torch.cuda.synchronize();gerr=(gy.float()-ref).abs();gbad=gerr>tol;gcast=gy.half();graph_stats={'strict_bad':int(gbad.sum()),'max_abs':float(gerr.max()),'finite':bool(torch.isfinite(gy).all()),'equal_eager':bool(torch.equal(gy,y)),'fp16_cast_finite':bool(torch.isfinite(gcast).all()),'cast_equal_eager':bool(torch.equal(gcast,cast))}
 if graph_stats['strict_bad'] or not repeat or not ind or not graph_stats['equal_eager'] or not cast_finite or not graph_stats['cast_equal_eager']:raise RuntimeError(f'graph/repeat/scratch failure {graph_stats} repeat={repeat} independent={ind}')
 # Full call includes activation gather, FP32 partial/reducer and final cast. Graph is captured with identical FP32 scratch plus cast.
 def full_call(xx=xin):return mod.c57_groupmajor_gptq(mod.groupmajor_activation(xx,perm),qmajor,qz,sc,128,False)
 def graph_full_call():return mod.c57_groupmajor_gptq(gx,qmajor,qz,sc,128,False)
 graph_cast=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph_cast):gout=graph_full_call()
 graph_cast.replay();torch.cuda.synchronize();graph_cast_exact=torch.equal(gout,cast)
 if not graph_cast_exact:raise RuntimeError('Graph FP16 cast differs from eager cast')
 def bench(f,n=20):
  for _ in range(5):f()
  torch.cuda.synchronize();v=[]
  for _ in range(n):
   s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();f();e.record();e.synchronize();v.append(s.elapsed_time(e)*1000)
  return sorted(v)
 eager=bench(full_call);gt=bench(graph_cast.replay)
 print(json.dumps({'fixture':a.fixture,'M':a.M,'stress4':a.stress4,'binary':str(binary),'binary_sha256':bsha,'source_sha256':hashlib.sha256((E/'actorder_groupmajor_c57.cu').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'eager':ev,'graph':graph_stats,'repeat_equal':repeat,'independent_scratch_equal':ind,'fp16_cast_finite':cast_finite,'eager_full_median_us':eager[len(eager)//2],'graph_replay_median_us':gt[len(gt)//2],'graph_cast_exact':graph_cast_exact,'eager_samples_us':eager,'graph_samples_us':gt},indent=2))
finally:L.release(fd)
