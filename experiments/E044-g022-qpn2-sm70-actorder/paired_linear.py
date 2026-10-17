#!/usr/bin/env python3
"""Identity-safe E044 QPN vs exact-c57 complete Linear paired benchmark.

Each arm/cell uses a fresh process; the parent holds the target V100 UUID lock.
Eager and CUDA Graph timings include activation gather/permutation and output
cast for QPN. QPN prepack is separately timed/reported as one-time setup.
"""
import argparse,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'E035-g022-actorder-q4-wmma'))
from gpu_uuid_lock import acquire,release,assert_target_idle,assert_inherited,assert_cuda_uuid
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def child(a):
 import importlib.util,torch,numpy as np
 fd=int(os.environ['E035_GPU_LOCK_FD']);assert_inherited(fd,UUID);assert_cuda_uuid(torch,UUID)
 if a.arm=='c57':
  spec=importlib.util.spec_from_file_location('common_ops',a.c57);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);fn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate;binary=Path(mod.__file__)
 else:
  spec=importlib.util.spec_from_file_location('e044_qpn2_gptq',a.qpn);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);fn=mod.run;binary=Path(mod.__file__)
 root=Path(a.fixtures);d=root if (root/'manifest.json').exists() else next(x for x in root.iterdir() if x.is_dir() and json.loads((x/'manifest.json').read_text())['prefix']==a.fixture);mf=json.loads((d/'manifest.json').read_text());T=mf['tensors'];K=T['g_idx']['shape'][0];N=T['qweight']['shape'][1];G=T['scales']['shape'][0]
 def ld(n,dtype):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(T[n]['shape']).cuda()
 x4=ld('x',torch.float16);qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1);M=a.M;x=x4[:M].contiguous()
 # Donor fragment-order prepack on CPU from frozen quantized source; include upload.
 t0=time.perf_counter();perm_cpu=np.argsort(g.cpu().numpy(),kind='stable').astype(np.int32)
 qsrc=np.empty((K,N),np.uint8);qraw=qw.cpu().numpy().view(np.uint32)
 for b in range(8):qsrc[b::8]=((qraw>>(4*b))&15).astype(np.uint8)
 tiles=N//32;groups16=K//16;col_cpu=np.array([((l>>2)&3)*8+(l&3)+(4 if l&16 else 0) for l in range(32)]);ko_cpu=np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);pn=np.empty((tiles,groups16,32,8),np.uint8)
 for tile in range(tiles):
  for kg in range(groups16):
   sk=perm_cpu[(kg//8)*128+(kg%8)*16+ko_cpu];v=qsrc[sk[:,None],(tile*32+col_cpu)[None,:]].T;pn[tile,kg]=v[:,0::2]|(v[:,1::2]<<4)
 pn_t=torch.from_numpy(pn.view(np.int32).copy()).cuda();perm=torch.from_numpy(perm_cpu).cuda();torch.cuda.synchronize();prepack_ms=(time.perf_counter()-t0)*1e3
 gx=torch.empty_like(x);out=torch.empty((M,N),device='cuda',dtype=torch.float32);workspace_bytes=gx.numel()*gx.element_size()+out.numel()*out.element_size()
 if a.arm=='c57':
  def call():return fn(x,qw,qz,sc,g,False)
 else:
  def call():return mod.run(x,pn_t,sc,perm,gx,out)
 # Frozen independent oracle and correctness.
 z=torch.empty((qz.shape[0],qz.shape[1]*8),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>>(4*b))&15
 qq=torch.empty((K,N),device='cuda',dtype=torch.int32)
 for b in range(8):qq[b::8]=(qw>>(4*b))&15
 w64=(qq.double()-(z[g.long()]+1).double())*sc.double()[g.long()];ref=(x.double()@w64).float()
 y32=call();torch.cuda.synchronize()
 if a.arm=='qpn':
  tol=1e-4+5e-5*ref.abs();bad=int(((y32-ref).abs()>tol).sum().item())
  if bad:raise RuntimeError(f'QPN strict oracle bad={bad}')
 y=call();torch.cuda.synchronize();
 if not torch.isfinite(y).all():raise RuntimeError('nonfinite output')
 def measure(f):
  for _ in range(5):f()
  torch.cuda.synchronize();vals=[]
  for _ in range(a.iters):
   st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();f();en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000)
  vals.sort();return {'median_us':vals[len(vals)//2],'p10_us':vals[int(.1*(len(vals)-1))],'p90_us':vals[int(.9*(len(vals)-1))],'n':len(vals)}
 eager=measure(call)
 # Graph over full operator; QPN event includes gather + kernel; c57 is native path.
 xg=x.clone();gg=torch.empty_like(xg);oo=torch.empty((M,N),device='cuda',dtype=torch.float32)
 if a.arm=='qpn':
  oq=torch.empty((M,N),device='cuda',dtype=torch.float16)
  def gcall():
   val=mod.run(xg,pn_t,sc,perm,gg,oo);oq.copy_(val.half());return oq
 else:
  def gcall():return fn(xg,qw,qz,sc,g,False)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):gy=gcall()
  graph.replay();torch.cuda.synchronize()
 torch.cuda.current_stream().wait_stream(stream);go=oo.clone() if a.arm=='qpn' else gy.float().clone()
 if a.arm=='qpn':
  gb=int(((go-ref).abs()>(1e-4+5e-5*ref.abs())).sum().item());castbad=int((gy!=go.half()).sum().item())
  if gb or castbad:raise RuntimeError(f'Graph failed strict/cast: {gb}/{castbad}')
 graph_t=measure(graph.replay)
 return {'arm':a.arm,'fixture':a.fixture,'M':M,'binary':str(binary),'binary_sha256':sha(binary),'eager':eager,'graph':graph_t,'prepack_ms':prepack_ms if a.arm=='qpn' else None,'scratch_bytes':workspace_bytes if a.arm=='qpn' else 0,'fp32_bad':bad if a.arm=='qpn' else None,'graph_bad':gb if a.arm=='qpn' else None,'graph_cast_bad':castbad if a.arm=='qpn' else None,'actual_M':M}

def main(a):
 assert_target_idle(UUID);fd=acquire(UUID);rid=str(uuid.uuid4());raw=Path(a.out).with_name(Path(a.out).stem+'.'+rid+'.jsonl');log=raw.with_suffix('.run.jsonl');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,CUDA_HOME='/usr/local/cuda-12.8',E035_GPU_LOCK_FD=str(fd))
 try:
  with log.open('w') as lf:
   for fixture in a.fixture:
    for M in (1,2,4):
     for pair in range(a.pairs):
      order=('qpn','c57') if pair%2==0 else ('c57','qpn');arms={}
      for arm in order:
       cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--fixtures',a.fixtures,'--qpn',a.qpn,'--c57',a.c57,'--fixture-one',fixture,'--M',str(M),'--arm',arm,'--iters',str(a.iters),'--out',a.out]
       cp=subprocess.run(cmd,text=True,capture_output=True,env=env,pass_fds=(fd,));lf.write(json.dumps({'fixture':fixture,'M':M,'pair':pair+1,'arm':arm,'exit':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})+'\n');lf.flush()
       if cp.returncode:raise RuntimeError(cp.stderr)
       arms[arm]=json.loads(cp.stdout.strip().splitlines()[-1])
      row={'fixture':fixture,'M':M,'pair':pair+1,'order':order,'arms':arms};raw.open('a').write(json.dumps(row)+'\n');print(json.dumps({'fixture':fixture,'M':M,'pair':pair+1,'order':order,'eager_us':{k:v['eager']['median_us'] for k,v in arms.items()},'graph_us':{k:v['graph']['median_us'] for k,v in arms.items()}}),flush=True)
  print(json.dumps({'run_id':rid,'raw':str(raw),'runlog':str(log)}))
 finally:release(fd)

p=argparse.ArgumentParser();p.add_argument('--fixtures',default=str(HERE/'fixtures/l43-down/packed-word-fixture-package'));p.add_argument('--qpn',default=str(HERE/'build-qpn/e044_qpn2_gptq.so'));p.add_argument('--c57',default='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so');p.add_argument('--fixture-one');p.add_argument('--fixture',action='append',default=[]);p.add_argument('--M',type=int,choices=(1,2,4));p.add_argument('--arm',choices=('qpn','c57'));p.add_argument('--child',action='store_true');p.add_argument('--iters',type=int,default=100);p.add_argument('--pairs',type=int,default=3);p.add_argument('--out',default=str(HERE/'results/paired-linear.jsonl'));a=p.parse_args()
if a.child:print(json.dumps(child(a)),flush=True)
else:
 if a.fixture_one:raise SystemExit('fixture-one is child-only')
 if not a.fixture:a.fixture=['43.mlp.down_proj']
 main(a)
