#!/usr/bin/env python3
"""Alternating T011 multiwarp vs exact-c57 complete-Linear CUDA Graph timing."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'E035-g022-actorder-q4-wmma'))
from gpu_uuid_lock import acquire,release,assert_target_idle,assert_cuda_uuid,assert_inherited
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def child(a):
 import numpy as np, torch
 fd=int(os.environ['E035_GPU_LOCK_FD']);assert_inherited(fd,UUID);assert_cuda_uuid(torch,UUID)
 if a.arm=='c57':
  spec=importlib.util.spec_from_file_location('common_ops',a.c57);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);fn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate;binary=Path(mod.__file__)
 else:
  spec=importlib.util.spec_from_file_location('qpn2_multiwarp',a.module);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);binary=Path(mod.__file__)
 d=Path(a.fixture);mf=json.loads((d/'manifest.json').read_text());T=mf['tensors'];M=a.M;K=T['g_idx']['shape'][0];N=T['qweight']['shape'][1];G=T['scales']['shape'][0]
 def ld(n,dtype):
  p=d/(n+'.bin');return torch.from_file(str(p),shared=False,size=p.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(T[n]['shape']).cuda()
 x=ld('x',torch.float16)[:M].contiguous();qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1)
 if a.arm=='qpn':
  qsrc=np.empty((K,N),np.uint8);qraw=qw.cpu().numpy().view(np.uint32)
  for b in range(8):qsrc[b::8]=((qraw>>(4*b))&15).astype(np.uint8)
  perm_cpu=np.argsort(g.cpu().numpy(),kind='stable').astype(np.int32);tiles=N//32;groups16=K//16
  col=np.array([((l>>2)&3)*8+(l&3)+(4 if l&16 else 0) for l in range(32)]);ko=np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);pn=np.empty((tiles,groups16,32,8),np.uint8)
  for tile in range(tiles):
   for kg in range(groups16):
    sk=perm_cpu[(kg//8)*128+(kg%8)*16+ko];v=qsrc[sk[:,None],(tile*32+col)[None,:]].T;pn[tile,kg]=v[:,0::2]|(v[:,1::2]<<4)
  qt=torch.from_numpy(pn.view(np.int32).copy()).cuda();pt=torch.from_numpy(perm_cpu).cuda();gx=torch.empty_like(x);o=torch.empty((M,N),device='cuda',dtype=torch.float32)
  def compute():return mod.run(x,qt,sc,pt,gx,o,a.split)
 else:
  def compute():return fn(x,qw,qz,sc,g,False)
 # Full graph boundary: includes gather+HMMA+bounded CTA FP32 reduction+FP16 output cast.
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):
   value=compute();out=value.half() if value.dtype==torch.float32 else value
  graph.replay();torch.cuda.synchronize()
 torch.cuda.current_stream().wait_stream(stream)
 for _ in range(20):graph.replay()
 torch.cuda.synchronize();vals=[]
 for _ in range(a.iters):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();graph.replay();en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000)
 vals.sort();return {'arm':a.arm,'M':M,'split':a.split if a.arm=='qpn' else None,'fixture':str(d),'binary':str(binary),'binary_sha256':sha(binary),'graph_full_linear_cast_inclusive':True,'median_us':(vals[len(vals)//2-1]+vals[len(vals)//2])/2 if len(vals)%2==0 else vals[len(vals)//2],'p10_us':vals[int(.1*(len(vals)-1))],'p90_us':vals[int(.9*(len(vals)-1))],'n':len(vals)}
def main(a):
 assert_target_idle(UUID);fd=acquire(UUID);rid=str(uuid.uuid4());raw=Path(a.out).with_name(Path(a.out).stem+'.'+rid+'.jsonl');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,CUDA_HOME='/usr/local/cuda-12.8',E035_GPU_LOCK_FD=str(fd))
 try:
  with raw.open('w') as f:
   for pair in range(a.pairs):
    order=('qpn','c57') if pair%2==0 else ('c57','qpn');r={}
    for arm in order:
     cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--fixture',a.fixture,'--module',a.module,'--c57',a.c57,'--arm',arm,'--M',str(a.M),'--split',str(a.split),'--iters',str(a.iters),'--out',a.out]
     cp=subprocess.run(cmd,text=True,capture_output=True,env=env,pass_fds=(fd,));
     if cp.returncode:raise RuntimeError(cp.stderr)
     r[arm]=json.loads(cp.stdout.strip().splitlines()[-1])
    row={'pair':pair+1,'order':order,'arms':r};f.write(json.dumps(row)+'\n');f.flush();print(json.dumps({k:v['median_us'] for k,v in r.items()}),flush=True)
  print(str(raw))
 finally:release(fd)
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--module',default='/mnt/data/sglang-V100/t011-build/qpn2_multiwarp.so');p.add_argument('--c57',default='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so');p.add_argument('--arm',choices=['qpn','c57']);p.add_argument('--child',action='store_true');p.add_argument('--M',type=int,default=4);p.add_argument('--split',type=int,choices=(8,16,32),default=8);p.add_argument('--iters',type=int,default=100);p.add_argument('--pairs',type=int,default=3);p.add_argument('--out',default=str(HERE/'results/t011-paired.jsonl'));a=p.parse_args()
if a.child:print(json.dumps(child(a)))
else:main(a)
