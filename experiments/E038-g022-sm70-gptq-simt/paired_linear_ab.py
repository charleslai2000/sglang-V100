#!/usr/bin/env python3
"""Fresh-process, UUID-locked E038 complete projection A/B against S1 and c57."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
E=Path(__file__).resolve().parent;LOCK=Path('/home/charles/Workspaces/3rdparty/sglang-V100/experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py');sp=importlib.util.spec_from_file_location('e038_lock',LOCK);L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';FIXROOT=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');S1=Path('/home/charles/miniconda3/envs/sglang-v100/lib/python3.12/site-packages/sgl_kernel/sm70/common_ops.abi3.so');C57=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so');EXT=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E038/build/e038_group_major_simt.so')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def child(a):
 import torch
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
 L.assert_inherited(int(os.environ['E038_GPU_LOCK_FD']),UUID);L.assert_cuda_uuid(torch,UUID)
 d=FIXROOT/a.fixture;m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def load(n,dt):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(t[n]['shape']).cuda().contiguous()
 x0=load('x',torch.float16);x=x0[:a.M].contiguous();qw=load('qweight',torch.int32);qz=load('qzeros',torch.int32);sc=load('scales',torch.float16);g=load('g_idx',torch.int32).reshape(-1);K,N=x.shape[1],qw.shape[1]
 codes=((qw.to(torch.int64).repeat_interleave(8,dim=0)>>((torch.arange(K,device='cuda',dtype=torch.int64)%8)*4).view(K,1))&15).to(torch.int32);zeros=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
 for j in range(8):zeros[:,j::8]=(qz>>(4*j))&15
 ref=(x.double()@((codes.double()-(zeros[g.long()].double()+1))*sc.double()[g.long()])).float()
 if a.arm=='e038':
  from torch.utils.cpp_extension import load
  mod=load(name='e038_group_major_simt',sources=[str(E/'actorder_group_q4_simt.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E038/build',verbose=False)
  binary=Path(mod.__file__).resolve();qgm,perm,_,_=mod.group_major_pack(qw,qz,sc,g,128)
  def fn(xx=x):return mod.group_major_fp32(mod.permute_activation_group_major(xx,perm,128).contiguous(),qgm,qz,sc,128,4)
  oracle=True;cast=lambda y:y.half()
 else:
  binary=S1 if a.arm=='s1' else C57
  assert sha(binary)==('fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842' if a.arm=='s1' else 'c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9')
  sx=importlib.util.spec_from_file_location('common_ops',str(binary));mod=importlib.util.module_from_spec(sx);sx.loader.exec_module(mod)
  if a.arm=='s1':
   from sgl_kernel.gemm import gptq_gemm
   fn=lambda xx=x:gptq_gemm(xx,qw.view(torch.uint32),qz.view(torch.uint32),sc,g,False,4)
  else:fn=lambda xx=x:torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(xx,qw,qz,sc,g,False)
  oracle=False
  if a.arm=='s1':
   def cast(y):return y
  else:
   def cast(y):return y.half()
 y=fn();torch.cuda.synchronize();finite=bool(torch.isfinite(y).all());strict=None
 if oracle:
  err=(y-ref).abs();tol=1e-4+5e-5*ref.abs();strict=int((err>tol).sum());
  if strict:raise RuntimeError(f'E038 strict failure {strict}')
 ycast=cast(y);cast_finite=bool(torch.isfinite(ycast).all())
 def bench(f):
  for _ in range(8):f()
  torch.cuda.synchronize();v=[]
  for _ in range(a.iters):
   s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();f();e.record();e.synchronize();v.append(s.elapsed_time(e)*1000)
  return sorted(v)
 times=bench(fn);return {'arm':a.arm,'fixture':a.fixture,'M':a.M,'K':K,'N':N,'binary':str(binary),'binary_sha256':sha(binary),'manifest_sha256':sha(d/'manifest.json'),'strict_bad':strict,'finite':finite,'cast_finite':cast_finite,'median_us':times[len(times)//2],'samples_us':times,'pid':os.getpid()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--fixtures',nargs='+',default=['43-mlp-gate_up_proj-call0000','43-mlp-down_proj-call0000']);p.add_argument('--M',type=int,default=4);p.add_argument('--pairs',type=int,default=3);p.add_argument('--iters',type=int,default=30);p.add_argument('--out',required=False);p.add_argument('--child',action='store_true');p.add_argument('--arm',choices=['e038','s1','c57']);p.add_argument('--fixture');p.add_argument('--lock-fd',type=int);a=p.parse_args()
 if a.child:print(json.dumps(child(a)),flush=True);return
 if not a.out:raise SystemExit('--out is required in parent mode')
 L.assert_target_idle(UUID);fd=L.acquire(UUID);os.environ['CUDA_VISIBLE_DEVICES']=UUID;out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 try:
  with out.open('w') as f:
   for fixture in a.fixtures:
    for pair in range(a.pairs):
     order=[['e038','s1','c57'],['s1','c57','e038'],['c57','e038','s1']][pair%3];rows={}
     for arm in order:
      cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--arm',arm,'--fixture',fixture,'--M',str(a.M),'--iters',str(a.iters)]
      env={**os.environ,'CUDA_VISIBLE_DEVICES':UUID,'E038_GPU_LOCK_FD':str(fd)};cp=subprocess.run(cmd,capture_output=True,text=True,env=env,pass_fds=(fd,))
      if cp.returncode:raise RuntimeError(f'{arm} failed: {cp.stderr}')
      rows[arm]=json.loads(cp.stdout.strip().splitlines()[-1]);f.write(json.dumps({'fixture':fixture,'M':a.M,'pair':pair+1,'order':order,'arms':rows})+'\n');f.flush();os.fsync(f.fileno())
     print(json.dumps({'fixture':fixture,'M':a.M,'pair':pair+1,'order':order,'medians_us':{k:v['median_us'] for k,v in rows.items()}}),flush=True)
 finally:L.release(fd)
if __name__=='__main__':main()
