#!/usr/bin/env python3
"""Identity-safe isolated-process E035 / exact-c57 paired benchmark.

A parent holds a per-GPU UUID flock for the whole cohort. Every arm/cell gets a
fresh interpreter. Completed arms are recorded immediately; incomplete pairs
are never included in output statistics.
"""
import argparse, hashlib, json, os, subprocess, sys, time, uuid
from pathlib import Path
from gpu_uuid_lock import acquire as acquire_gpu_lock, release as release_gpu_lock, assert_target_idle, assert_inherited, assert_cuda_uuid
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def child(a):
 # Refuse before importing torch if this child is not part of the parent's held lock.
 lock_fd=int(os.environ.get('E035_GPU_LOCK_FD','-1'))
 if lock_fd<0 or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:
  raise RuntimeError('GPU child must inherit parent UUID lock and target-only CUDA_VISIBLE_DEVICES')
 assert_inherited(lock_fd,a.gpu_uuid)
 import torch, importlib.util
 assert_cuda_uuid(torch,a.gpu_uuid)
 os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='1';os.environ['SGLANG_G017_Q4_FP32_CANDIDATE_PROBE']='0'
 arm=a.arm
 if arm=='c57':
  binary=Path(a.c57).resolve();sp=importlib.util.spec_from_file_location('common_ops',str(binary));mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
  actual=Path(mod.__file__).resolve();fn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate
 else:
  binary=Path(a.binary).resolve();module_name='e035_grouped_q4_parallel_sm70' if arm=='parallel' else 'e035_grouped_q4_sm70'
  sp=importlib.util.spec_from_file_location(module_name,str(binary));mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
  actual=Path(mod.__file__).resolve();fn=mod.grouped_q4_sm70_parallel if arm=='parallel' else mod.grouped_q4_sm70
  if fn.__module__!=module_name:raise RuntimeError(f'native function owner mismatch: {fn.__module__}')
 if actual!=binary:raise RuntimeError(f'loaded path mismatch {actual} != {binary}')
 root=Path(a.fixtures);d=next(x for x in root.iterdir() if x.is_dir() and json.loads((x/'manifest.json').read_text())['prefix']==a.fixture)
 m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def ld(n,dtype):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(t[n]['shape']).cuda()
 x4=ld('x',torch.float16);qw0=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);scales=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1)
 M=a.M;x=x4[:M].contiguous();K,N=x.shape[1],qw0.shape[1];G=scales.shape[0]
 perm=torch.argsort(g,stable=True).int();q=torch.empty((K,N),device='cuda',dtype=torch.int32)
 for b in range(8):q[b::8]=(qw0>>(4*b))&15
 qg=q.index_select(0,perm);packed=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
 for b in range(8):packed|=qg[b::8]<<(4*b)
 part=torch.empty((G,M,N) if arm=='parallel' else (M,N),device='cuda',dtype=torch.float32);gx=torch.empty_like(x);out=torch.empty((M,N),device='cuda',dtype=torch.float32)
 def call(fp32=False):
  if arm=='c57':return fn(x,qw0,qz,scales,g,False)
  return fn(x,packed,scales,perm,part,gx,out,fp32)
 z=torch.empty((qz.shape[0],qz.shape[1]*8),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>>(4*b))&15
 w64=(q.double()-(z[g.long()]+1).double())*scales.double()[g.long()]
 ref32=(x.double()@w64).float()
 if arm=='c57':
  ycheck=call();torch.cuda.synchronize();ref=ref32.half().float();bad=None
  if not torch.isfinite(ycheck).all().item():raise RuntimeError('c57 output non-finite')
 else:
  ycheck=call(True);torch.cuda.synchronize();ref=ref32;tol=1e-4+5e-5*ref.abs()
  bad=int(((ycheck.float()-ref).abs()>tol).sum().item())
  if bad:raise RuntimeError(f'{arm} strict oracle bad={bad}')
 y=call();torch.cuda.synchronize();cast=y if y.dtype==torch.float16 else y.half()
 if not torch.isfinite(cast).all().item():raise RuntimeError('final FP16 cast non-finite')
 for _ in range(8):call()
 torch.cuda.synchronize()
 def measure(f):
  for _ in range(3):f()
  torch.cuda.synchronize();vals=[]
  for _ in range(a.iters):
   st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();f();en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000)
  vals.sort();return {'median_us':vals[len(vals)//2],'p10_us':vals[max(0,int(.1*(len(vals)-1)))],'p90_us':vals[min(len(vals)-1,int(.9*(len(vals)-1)))],'n':len(vals)}
 eager=measure(call)
 gx_input=x.clone().contiguous();gp=torch.empty_like(part);gg=torch.empty_like(gx);go=torch.empty_like(out)
 def graph_call():
  if arm=='c57':return fn(gx_input,qw0,qz,scales,g,False)
  return fn(gx_input,packed,scales,perm,gp,gg,go,False)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):gy=graph_call()
  graph.replay();torch.cuda.synchronize()
 torch.cuda.current_stream().wait_stream(stream)
 graph_eager_mismatch=int((gy.float()!=y.float()).sum().item())
 graph_go_finite=bool(torch.isfinite(go).all().item()) if arm!='c57' else bool(torch.isfinite(gy).all().item())
 graph_oracle_bad=0 if arm=='c57' else int(((go-ref32).abs()>(1e-4+5e-5*ref32.abs())).sum().item())
 graph_cast_bad=0 if arm=='c57' else int((gy!=go.half()).sum().item())
 if graph_eager_mismatch or not graph_go_finite or graph_oracle_bad or graph_cast_bad:raise RuntimeError('Graph go/oracle/cast/eager check failed')
 graph_t=measure(graph.replay)
 return {'kind':'result','arm':arm,'fixture':a.fixture,'M':M,'K':K,'N':N,'manifest_sha256':sha(d/'manifest.json'),'binary':str(binary),'binary_sha256':sha(binary),'loaded_file':str(actual),'native_fn_id':id(fn),'native_fn_module':getattr(fn,'__module__',None),'benchmark_script_sha256':sha(Path(__file__).resolve()),'pybind11_headers':'torch/include/pybind11 3.0.1 (PyTorch vendored)','fp32_oracle_bad':bad,'fp32_max_abs_error':float((ycheck.float()-ref).abs().max().item()) if arm!='c57' else None,'fp16_finite':True,'graph_eager_mismatch':graph_eager_mismatch,'graph_go_finite':graph_go_finite,'graph_oracle_bad':graph_oracle_bad,'graph_cast_bad':graph_cast_bad,'graph_output_dtype':str(gy.dtype),'eager':eager,'graph_replay':graph_t,'pid':os.getpid()}

p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--old',default='');p.add_argument('--r1',required=True);p.add_argument('--parallel',default='');p.add_argument('--c57',required=True);p.add_argument('--binary');p.add_argument('--out',default='/tmp/e035-child-unused.jsonl');p.add_argument('--iters',type=int,default=40);p.add_argument('--pairs',type=int,default=3);p.add_argument('--gpu-uuid',default='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');p.add_argument('--inherited-lock-fd',type=int);p.add_argument('--only-prefix',action='append');p.add_argument('--only-M',type=int,choices=(1,2,4));p.add_argument('--arm',choices=('old','r1','parallel','c57'));p.add_argument('--fixture');p.add_argument('--M',type=int,choices=(1,2,4));p.add_argument('--child',action='store_true');a=p.parse_args()
if a.child:
 result=child(a);print(json.dumps(result),flush=True);raise SystemExit(0)
if a.arm:raise SystemExit('use --child for single-arm invocation')
if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'',a.gpu_uuid):raise SystemExit(f'CUDA_VISIBLE_DEVICES must be unset or exactly {a.gpu_uuid}')
if a.inherited_lock_fd is None:assert_target_idle(a.gpu_uuid)
if a.inherited_lock_fd is not None:
 lock_fd=a.inherited_lock_fd
 try:os.fstat(lock_fd)
 except OSError as e:raise SystemExit(f'inherited UUID lock fd invalid: {e}')
 if os.environ.get('E035_GPU_LOCK_FD')!=str(lock_fd):raise SystemExit('inherited lock fd marker mismatch')
 assert_inherited(lock_fd,a.gpu_uuid)
 owns_lock=False
else:
 lock_fd=acquire_gpu_lock(a.gpu_uuid);owns_lock=True
try:
 root=Path(a.fixtures);dirs=sorted(d for d in root.iterdir() if d.is_dir() and (d/'manifest.json').exists())
 if len(dirs)!=6:raise SystemExit(f'expected six fixtures, got {len(dirs)}')
 fixtures=[json.loads((d/'manifest.json').read_text())['prefix'] for d in dirs]
 if a.only_prefix:fixtures=[x for x in fixtures if x in set(a.only_prefix)]
 ms=[a.only_M] if a.only_M else [1,2,4]
 paths={'old':a.old,'r1':a.r1,'parallel':a.parallel,'c57':a.c57};out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 run_id=str(uuid.uuid4());raw=out.with_name(f'{out.stem}.{run_id}.jsonl')
 runlog=raw.with_suffix('.run.jsonl')
 with runlog.open('w') as log:
  log.write(json.dumps({'event':'run_start','run_id':run_id,'uuid':a.gpu_uuid,'lock_pid':os.getpid(),'started_ns':time.time_ns(),'script_sha256':sha(Path(__file__))})+'\n');log.flush();os.fsync(log.fileno())
  completed=[];failure=None
  try:
   for fixture in fixtures:
    for M in ms:
     for pair in range(a.pairs):
      active=tuple(k for k in ('old','r1','parallel','c57') if paths[k])
      if len(active)<2:raise SystemExit('at least two nonempty arms required')
      order=[active[i:]+active[:i] for i in range(len(active))][pair%len(active)];timings={};arm_rows=[]
      for arm in order:
       started=time.time_ns();childlog={'event':'arm_start','run_id':run_id,'fixture':fixture,'M':M,'pair':pair+1,'arm':arm,'started_ns':started,'binary_sha256':sha(paths[arm])}
       log.write(json.dumps(childlog)+'\n');log.flush();os.fsync(log.fileno())
       cmd=[sys.executable,str(HERE/'paired_isolated.py'),'--child','--inherited-lock-fd',str(lock_fd),'--fixtures',a.fixtures,'--old',a.old,'--r1',a.r1,'--parallel',a.parallel,'--c57',a.c57,'--gpu-uuid',a.gpu_uuid,'--arm',arm,'--binary',paths[arm],'--fixture',fixture,'--M',str(M),'--iters',str(a.iters)]
       env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'E035_GPU_LOCK_FD':str(lock_fd)}
       t0=time.monotonic();cp=subprocess.run(cmd,text=True,capture_output=True,env=env,pass_fds=(lock_fd,));ended=time.time_ns()
       record={'event':'arm_end','run_id':run_id,'fixture':fixture,'M':M,'pair':pair+1,'arm':arm,'started_ns':started,'ended_ns':ended,'elapsed_s':time.monotonic()-t0,'exit_status':cp.returncode,'binary_sha256':sha(paths[arm]),'stdout':cp.stdout if cp.returncode else '' ,'stderr':cp.stderr}
       log.write(json.dumps(record)+'\n');log.flush();os.fsync(log.fileno())
       if cp.returncode:raise RuntimeError(f'arm {arm} failed: {cp.stderr}')
       result=json.loads(cp.stdout.strip().splitlines()[-1]);timings[arm]=result;arm_rows.append(result)
       # Checkpoint every successful arm immediately; only pair_complete rows are statistical pairs.
       raw.parent.mkdir(parents=True,exist_ok=True)
       with raw.open('a') as f:f.write(json.dumps({'kind':'arm_complete','run_id':run_id,'fixture':fixture,'M':M,'pair':pair+1,'arm':arm,'result':result})+'\n');f.flush();os.fsync(f.fileno())
      pairrow={'kind':'pair_complete','run_id':run_id,'fixture':fixture,'M':M,'pair':pair+1,'order':order,'arms':timings}
      with raw.open('a') as f:f.write(json.dumps(pairrow)+'\n');f.flush();os.fsync(f.fileno())
      completed.append(pairrow);print(json.dumps({'run_id':run_id,'fixture':fixture,'M':M,'pair':pair+1,'eager_us':{k:v['eager']['median_us'] for k,v in timings.items()},'graph_us':{k:v['graph_replay']['median_us'] for k,v in timings.items()}}),flush=True)
  except BaseException as e:
   failure=repr(e);raise
  finally:
   log.write(json.dumps({'event':'run_end','run_id':run_id,'ended_ns':time.time_ns(),'exit_status':0 if failure is None else 1,'completed_pairs':len(completed),'failure':failure,'raw':str(raw)})+'\n');log.flush();os.fsync(log.fileno())
 print(json.dumps({'run_id':run_id,'raw':str(raw),'runlog':str(runlog),'complete_pairs':len(completed)}),flush=True)
finally:
 if owns_lock:release_gpu_lock(lock_fd)
