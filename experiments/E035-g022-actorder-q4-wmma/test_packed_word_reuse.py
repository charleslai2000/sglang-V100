#!/usr/bin/env python3
"""Identity-safe c57 vs packed-word-reuse operator oracle/Graph/paired test."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time, uuid
from pathlib import Path
from gpu_uuid_lock import acquire, release, assert_target_idle, assert_inherited, assert_cuda_uuid
HERE=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def arm(a):
 fd=int(os.environ.get('E035_GPU_LOCK_FD','-1'))
 if fd<0 or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise RuntimeError('child requires inherited lock and target-only visibility')
 assert_inherited(fd,a.gpu_uuid)
 import torch
 assert_cuda_uuid(torch,a.gpu_uuid)
 binary=Path(a.binary).resolve();spec=importlib.util.spec_from_file_location('common_ops',str(binary));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 actual=Path(mod.__file__).resolve()
 if actual!=binary:raise RuntimeError(f'loaded path mismatch: {actual} != {binary}')
 fn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate
 root=Path(a.fixtures);d=next(q for q in root.iterdir() if q.is_dir() and json.loads((q/'manifest.json').read_text())['prefix']==a.fixture)
 manifest=json.loads((d/'manifest.json').read_text());t=manifest['tensors']
 def ld(name,dtype):
  f=d/(name+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(t[name]['shape']).cuda()
 x4=ld('x',torch.float16);qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1)
 M=a.M;x=x4[:M].contiguous();K,N=x.shape[1],qw.shape[1]
 z=torch.empty((qz.shape[0],qz.shape[1]*8),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>>(4*b))&15
 q=torch.empty((K,N),device='cuda',dtype=torch.int32)
 for b in range(8):q[b::8]=(qw>>(4*b))&15
 w64=(q.double()-(z[g.long()]+1).double())*sc.double()[g.long()]
 ref=(x.double()@w64).float();tol=1e-4+5e-5*ref.abs()
 def call(xx=x,fp32=False):return fn(xx,qw,qz,sc,g,fp32)
 y32=call(fp32=True);torch.cuda.synchronize();strict_bad=int(((y32-ref).abs()>tol).sum().item())
 if strict_bad and a.arm=='candidate':raise RuntimeError(f'candidate strict eager FP32 oracle bad={strict_bad}')
 y=call();torch.cuda.synchronize()
 if y.dtype!=torch.float16 or not torch.isfinite(y).all().item():raise RuntimeError('invalid/nonfinite native FP16 output')
 gx=x.clone().contiguous();go=torch.empty((M,N),device='cuda',dtype=torch.float32)
 graph32=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph32):go_graph=fn(gx,qw,qz,sc,g,True)
 graph32.replay();torch.cuda.synchronize()
 graph_bad=int(((go_graph-ref).abs()>tol).sum().item());go_cast=go_graph.half()
 gy=fn(gx,qw,qz,sc,g,False);torch.cuda.synchronize()
 cast_bad=int((gy!=go_cast).sum().item());eager_bad=int((gy!=y).sum().item())
 if cast_bad or eager_bad:raise RuntimeError(f'Graph cast/eager mismatch {cast_bad}/{eager_bad}')
 repeat=call();torch.cuda.synchronize()
 repeat_equal=bool(torch.equal(repeat,y))
 out2=fn(x.clone().contiguous(),qw,qz,sc,g,False);torch.cuda.synchronize()
 independent_equal=bool(torch.equal(out2,y))
 if not repeat_equal or not independent_equal:raise RuntimeError('repeat/independent input mismatch')
 stress=[]
 if a.fixture=='43.mlp.down_proj' and M==4:
  xs=(x*4).half().contiguous();rs=(xs.double()@w64).float();ts=1e-4+5e-5*rs.abs();ys=fn(xs,qw,qz,sc,g,True);torch.cuda.synchronize()
  # Always record baseline strict failures; candidate failures below are fatal.
  stress_bad_mask=(ys-rs).abs()>ts
  stress_bad=int(stress_bad_mask.sum().item())
  if not torch.isfinite(ys).all().item() or not torch.isfinite(ys.half()).all().item():raise RuntimeError(f'4x L43/down produced nonfinite result strict={stress_bad}')
  stress_out=ys.half().detach().cpu().contiguous().numpy().tobytes()
  coords=stress_bad_mask.nonzero(as_tuple=False).cpu().tolist()
  stress=[{'factor':4,'strict_bad':stress_bad,'input_max_abs':float(xs.abs().max().item()),'input_fp16_sha256':hashlib.sha256(xs.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'oracle_fp32_sha256':hashlib.sha256(rs.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'fp32_output_sha256':hashlib.sha256(ys.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'fp16_sha256':hashlib.sha256(stress_out).hexdigest(),'bad_points':[{'m':i,'n':j,'actual_fp32':float(ys[i,j].item()),'reference_fp32':float(rs[i,j].item()),'abs_error':float((ys[i,j]-rs[i,j]).abs().item()),'allowed':float(ts[i,j].item())} for i,j in coords]}]
  if a.arm=='candidate' and stress_bad:raise RuntimeError(f'candidate violates frozen 4x stress oracle at {stress_bad} outputs')
 for _ in range(8):call()
 torch.cuda.synchronize()
 def measure(f):
  for _ in range(3):f()
  torch.cuda.synchronize();vals=[]
  for _ in range(a.iters):
   st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();f();en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000)
  vals.sort();return vals[len(vals)//2],vals
 eager,eager_samples=measure(call)
 gx2=x.clone().contiguous();gout=torch.empty((M,N),device='cuda',dtype=torch.float32)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):gyg=fn(gx2,qw,qz,sc,g,False)
 graph.replay();torch.cuda.synchronize();graph_bad_output=int((gyg!=y).sum().item())
 if graph_bad_output:raise RuntimeError('captured graph output differs from eager output')
 graph_t,graph_samples=measure(graph.replay)
 return {'run_id':os.environ['E035_RUN_ID'],'arm':a.arm,'fixture':a.fixture,'M':M,'K':K,'N':N,'binary':str(binary),'binary_sha256':sha(binary),'loaded_file':str(actual),'manifest_sha256':sha(d/'manifest.json'),'script_sha256':sha(Path(__file__)),'strict_bad_fp32':strict_bad,'strict_graph_bad_fp32':graph_bad,'graph_cast_bad':cast_bad,'eager_graph_mismatch':eager_bad,'repeat_equal':repeat_equal,'independent_input_equal':independent_equal,'stress':stress,'native_fp16_sha256':hashlib.sha256(y.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'native_fp16_finite':True,'eager_median_us':eager,'graph_median_us':graph_t,'eager_samples_us':eager_samples,'graph_samples_us':graph_samples,'pid':os.getpid()}

def child_parser():
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--fixtures',required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--arm',choices=('baseline','candidate'),required=True);p.add_argument('--binary',required=True);p.add_argument('--fixture',required=True);p.add_argument('--M',type=int,choices=(1,2,3,4),required=True);p.add_argument('--iters',type=int,default=50);p.add_argument('--stress',action='store_true');return p

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--baseline');p.add_argument('--candidate');p.add_argument('--out');p.add_argument('--gpu-uuid',default='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');p.add_argument('--prefix',action='append');p.add_argument('--iters',type=int,default=50);p.add_argument('--pairs',type=int,default=3);p.add_argument('--child',action='store_true');p.add_argument('--arm',choices=('baseline','candidate'));p.add_argument('--binary');p.add_argument('--fixture');p.add_argument('--M',type=int,choices=(1,2,3,4));p.add_argument('--stress',action='store_true');a=p.parse_args()
 if a.child:
  r=arm(a);print(json.dumps(r),flush=True);return
 if not a.baseline or not a.candidate or not a.out or not a.prefix:raise SystemExit('launcher requires --baseline, --candidate, --out and --prefix')
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'',a.gpu_uuid):raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
 assert_target_idle(a.gpu_uuid);fd=acquire(a.gpu_uuid);rid=str(uuid.uuid4());base=Path(a.out);base.parent.mkdir(parents=True,exist_ok=True);raw=base.with_name(f'{base.stem}.{rid}.jsonl');log=raw.with_suffix('.run.jsonl');paths={'baseline':Path(a.baseline).resolve(),'candidate':Path(a.candidate).resolve()}
 try:
  with log.open('w') as lf,raw.open('w') as rf:
   def rec(stream,row):stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
   rec(lf,{'event':'run_start','run_id':rid,'uuid':a.gpu_uuid,'launcher_pid':os.getpid(),'script_sha256':sha(Path(__file__)),'started_ns':time.time_ns()})
   for fixture in a.prefix:
    for M in (1,2,3,4):
     for pair in range(a.pairs):
      order=['baseline','candidate'] if pair%2==0 else ['candidate','baseline'];arms={}
      for name in order:
       assert_target_idle(a.gpu_uuid)
       if int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])<8388608:raise RuntimeError('MemAvailable below 8-GiB hard floor')
       argv=[sys.executable,str(Path(__file__).resolve()),'--child','--fixtures',a.fixtures,'--gpu-uuid',a.gpu_uuid,'--arm',name,'--binary',str(paths[name]),'--fixture',fixture,'--M',str(M),'--iters',str(a.iters)]
       if fixture=='43.mlp.down_proj' and M==4:argv.append('--stress')
       env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'E035_GPU_LOCK_FD':str(fd),'E035_RUN_ID':rid}
       start=time.time_ns();cp=subprocess.run(argv,text=True,capture_output=True,env=env,pass_fds=(fd,));end=time.time_ns()
       rec(lf,{'event':'arm_end','run_id':rid,'fixture':fixture,'M':M,'pair':pair+1,'arm':name,'started_ns':start,'ended_ns':end,'binary_sha256':sha(paths[name]),'exit_status':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})
       if cp.returncode:raise RuntimeError(f'{name}/{fixture}/M{M} failed: {cp.stderr}')
       row=json.loads(cp.stdout.strip().splitlines()[-1]);arms[name]=row
       rec(rf,{'kind':'arm_complete','run_id':rid,'fixture':fixture,'M':M,'pair':pair+1,'arm':name,'result':row})
      output_equal=arms['baseline']['native_fp16_sha256']==arms['candidate']['native_fp16_sha256']
      if not output_equal:raise RuntimeError(f'native FP16 output differs baseline/candidate fixture={fixture} M={M} pair={pair+1}')
      if fixture=='43.mlp.down_proj' and M==4:
       stress_equal=arms['baseline']['stress'][0]['fp16_sha256']==arms['candidate']['stress'][0]['fp16_sha256']
       if not stress_equal:raise RuntimeError('4x L43/down native FP16 output differs baseline/candidate')
      rec(rf,{'kind':'pair_complete','run_id':rid,'fixture':fixture,'M':M,'pair':pair+1,'order':order,'native_fp16_equal':output_equal,'arms':arms})
      print(json.dumps({'run_id':rid,'fixture':fixture,'M':M,'pair':pair+1,'order':order,'eager_us':{k:v['eager_median_us'] for k,v in arms.items()},'graph_us':{k:v['graph_median_us'] for k,v in arms.items()}}),flush=True)
   rec(lf,{'event':'run_end','run_id':rid,'exit_status':0,'ended_ns':time.time_ns(),'raw':str(raw)})
  print(json.dumps({'run_id':rid,'raw':str(raw),'runlog':str(log)}),flush=True)
 finally:release(fd)
if __name__=='__main__':main()
