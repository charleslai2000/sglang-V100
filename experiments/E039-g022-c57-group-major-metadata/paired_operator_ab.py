#!/usr/bin/env python3
"""Alternating fresh-process exact-c57 vs E039 full Linear and Graph A/B."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parent;LOCK=Path('/home/charles/Workspaces/3rdparty/sglang-V100/experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py')
sp=importlib.util.spec_from_file_location('e039_pair_lock',LOCK);L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';FIX=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');C57=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so');C57SHA='c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9';EXT=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E039/build/e039_c57_groupmajor.so')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def child(arm,fixture,M,iters,fd):
 import torch
 L.assert_inherited(fd,UUID);L.assert_cuda_uuid(torch,UUID)
 d=FIX/fixture;man=json.loads((d/'manifest.json').read_text());t=man['tensors']
 def ld(n,dt):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(t[n]['shape']).cuda().contiguous()
 x=ld('x',torch.float16)[:M].contiguous();qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);gi=ld('g_idx',torch.int32).view(-1);K,N=x.shape[1],qw.shape[1]
 if arm=='c57':
  assert sha(C57)==C57SHA
  spec=importlib.util.spec_from_file_location('common_ops',str(C57));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  c57_fn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate
  def call(xx=x):return c57_fn(xx,qw,qz,sc,gi,False)
  binary=C57
 else:
  assert sha(EXT)=='1e5cd0ce1b40c2ddf70f3162d1f43c0763736b8222d0b7e60483565283574c18'
  spec=importlib.util.spec_from_file_location('e039_c57_groupmajor',str(EXT));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  qmajor,perm=mod.stable_groupmajor_pack(qw,gi,128)
  def call(xx=x):return mod.c57_groupmajor_gptq(mod.groupmajor_activation(xx,perm),qmajor,qz,sc,128,False)
  binary=EXT
 out=call();torch.cuda.synchronize();assert out.dtype==torch.float16 and torch.isfinite(out).all()
 def graph_call(xx=x):return call(xx)
 graph=torch.cuda.CUDAGraph();gx=x.clone().contiguous()
 with torch.cuda.graph(graph):gy=graph_call(gx)
 graph.replay();torch.cuda.synchronize();assert torch.equal(gy,out)
 def bench(fn,n):
  for _ in range(8):fn()
  torch.cuda.synchronize();v=[]
  for _ in range(n):
   s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();fn();e.record();e.synchronize();v.append(s.elapsed_time(e)*1000)
  return sorted(v)
 eager=bench(call,iters);gt=bench(graph.replay,iters)
 return {'arm':arm,'fixture':fixture,'M':M,'K':K,'N':N,'binary':str(binary),'binary_sha256':sha(binary),'manifest_sha256':sha(d/'manifest.json'),'output_dtype':str(out.dtype),'graph_equal_eager':True,'eager_samples_us':eager,'graph_samples_us':gt,'eager_median_us':eager[len(eager)//2],'graph_median_us':gt[len(gt)//2],'pid':os.getpid()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=False);p.add_argument('--pairs',type=int,default=3);p.add_argument('--iters',type=int,default=40);p.add_argument('--child',action='store_true');p.add_argument('--arm',choices=['e039','c57']);p.add_argument('--fixture');p.add_argument('--M',type=int);a=p.parse_args()
 if a.child:
  fd=int(os.environ['E039_GPU_LOCK_FD']);res=child(a.arm,a.fixture,a.M,a.iters,fd);print(json.dumps(res),flush=True);return
 if not a.out:raise SystemExit('--out required')
 L.assert_target_idle(UUID);fd=L.acquire(UUID);os.environ['CUDA_VISIBLE_DEVICES']=UUID;out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 fixtures=['0-mlp-gate_up_proj-call0000','0-mlp-down_proj-call0000','22-mlp-gate_up_proj-call0000','22-mlp-down_proj-call0000','43-mlp-gate_up_proj-call0000','43-mlp-down_proj-call0000']
 try:
  with out.open('w') as f:
   for fixture in fixtures:
    for M in (1,2,4):
     for pair in range(a.pairs):
      order=[['e039','c57'],['c57','e039']][pair%2];rows={}
      for arm in order:
       cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--arm',arm,'--fixture',fixture,'--M',str(M),'--iters',str(a.iters)]
       env={**os.environ,'CUDA_VISIBLE_DEVICES':UUID,'E039_GPU_LOCK_FD':str(fd)}
       # child lock fd explicitly inherited
       cp=subprocess.run(cmd,capture_output=True,text=True,env=env,pass_fds=(fd,))
       if cp.returncode:raise RuntimeError(f'{arm}/{fixture}/M{M}: {cp.stderr}')
       rows[arm]=json.loads(cp.stdout.strip().splitlines()[-1]);f.write(json.dumps({'fixture':fixture,'M':M,'pair':pair+1,'order':order,'arms':rows})+'\n');f.flush();os.fsync(f.fileno())
      print(json.dumps({'fixture':fixture,'M':M,'pair':pair+1,'eager':{k:v['eager_median_us'] for k,v in rows.items()},'graph':{k:v['graph_median_us'] for k,v in rows.items()}}),flush=True)
 finally:L.release(fd)
if __name__=='__main__':main()
