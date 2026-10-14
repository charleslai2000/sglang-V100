#!/usr/bin/env python3
"""Identity-safe isolated-process E035 / exact-c57 paired benchmark.

Each arm/cell executes in a fresh interpreter. An interpreter loads at most one
E035 native module; process startup, imports, fixture setup, repack, warmup and
CUDA graph capture are outside CUDA-event timing.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def child(a):
 import torch
 import importlib.util
 os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='1'
 os.environ['SGLANG_G017_Q4_FP32_CANDIDATE_PROBE']='0'
 arm=a.arm
 if arm=='c57':
  binary=Path(a.c57).resolve()
  cs=importlib.util.spec_from_file_location('common_ops',str(binary))
  cm=importlib.util.module_from_spec(cs);cs.loader.exec_module(cm)
  actual=Path(cm.__file__).resolve();fn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate;cfn=fn
  if actual!=binary:raise RuntimeError(f'loaded wrong c57 module file: requested={binary} actual={actual}')
 else:
  binary=Path(a.binary).resolve()
  spec=importlib.util.spec_from_file_location('e035_grouped_q4_sm70',str(binary))
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  actual=Path(mod.__file__).resolve()
  if actual!=binary:raise RuntimeError(f'loaded wrong module file: requested={binary} actual={actual}')
  fn=mod.grouped_q4_sm70;cfn=None
  if fn.__module__!='e035_grouped_q4_sm70':raise RuntimeError(f'unexpected native function owner: {fn.__module__}')
 root=Path(a.fixtures);d=next(x for x in root.iterdir() if x.is_dir() and json.loads((x/'manifest.json').read_text())['prefix']==a.fixture)
 m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def ld(n,dtype):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(t[n]['shape']).cuda()
 x4=ld('x',torch.float16);qw0=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);scales=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1)
 M=a.M;x=x4[:M].contiguous();K,N=x.shape[1],qw0.shape[1];G=scales.shape[0]
 # Match original benchmark's deterministic Q4 repack and group permutation.
 perm=torch.argsort(g,stable=True).to(torch.int32);q=torch.empty((K,N),device='cuda',dtype=torch.int32)
 for b in range(8):q[b::8]=(qw0>>(4*b))&15
 qg=q.index_select(0,perm);packed=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
 for b in range(8):packed|=qg[b::8]<<(4*b)
 part=torch.empty((M,N),device='cuda',dtype=torch.float32);gx=torch.empty_like(x);out=torch.empty((M,N),device='cuda',dtype=torch.float32)
 def call():
  if arm=='c57':return cfn(x,qw0,qz,scales,g,False)
  return fn(x,packed,scales,perm,part,gx,out,False)
 # Validate this exact loaded extension: E035 exposes pre-cast FP32 for the frozen strict oracle.
 z=torch.empty((qz.shape[0],qz.shape[1]*8),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>>(4*b))&15
 w64=(q.double()-(z[g.long()]+1).double())*scales.double()[g.long()]
 ref32=(x.double()@w64).float()
 if arm=='c57':
  ycheck=call();torch.cuda.synchronize();ref=ref32.half().float();bad=None
  if not torch.isfinite(ycheck).all().item():raise RuntimeError('c57 non-finite output')
 else:
  ycheck=fn(x,packed,scales,perm,part,gx,out,True);torch.cuda.synchronize();ref=ref32;tol=1e-4+5e-5*ref.abs()
  bad=int(((ycheck.float()-ref).abs()>tol).sum().item())
  if bad:raise RuntimeError(f'correctness failed arm={arm} strict_bad={bad}')
 y=call();torch.cuda.synchronize()
 cast=y if y.dtype==torch.float16 else y.half()
 if not torch.isfinite(cast).all().item():raise RuntimeError('non-finite final FP16 cast')
 # Warmup and CUDA-event timing of complete call (gather + calculation + cast).
 for _ in range(8):call()
 torch.cuda.synchronize()
 def measure(fncall):
  for _ in range(3):fncall()
  torch.cuda.synchronize();vals=[]
  for _ in range(a.iters):
   st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
   st.record();fncall();en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000)
  vals.sort();return {'median_us':vals[len(vals)//2],'p10_us':vals[max(0,int(.1*(len(vals)-1)))],'p90_us':vals[min(len(vals)-1,int(.9*(len(vals)-1)))],'n':len(vals)}
 eager=measure(call)
 # Capture exactly the same full call; report graph replay as a distinct path.
 gx_input=x.clone().contiguous();gp=torch.empty_like(part);gg=torch.empty_like(gx);go=torch.empty_like(out)
 if arm=='c57':
  def graph_call():return cfn(gx_input,qw0,qz,scales,g,False)
 else:
  def graph_call():return fn(gx_input,packed,scales,perm,gp,gg,go,False)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):gy=graph_call()
  graph.replay();torch.cuda.synchronize()
 torch.cuda.current_stream().wait_stream(stream)
 graph_bad=int((gy.float()!=y.float()).sum().item())
 graph_strict_bad=0 if arm=='c57' else int(((gy.float()-ref).abs()>tol).sum().item())
 if graph_bad:raise RuntimeError(f'graph/eager output mismatch bad={graph_bad}')
 if graph_strict_bad:raise RuntimeError(f'graph strict oracle failed bad={graph_strict_bad}')
 graph_t=measure(graph.replay)
 return {'kind':'result','arm':arm,'fixture':a.fixture,'M':M,'K':K,'N':N,'manifest_sha256':sha(d/'manifest.json'),'binary':str(binary),'binary_sha256':sha(binary),'loaded_file':str(actual),'native_fn_id':id(fn),'native_fn_module':getattr(fn,'__module__',None),'native_fn_doc':getattr(fn,'__doc__',None),'c57_sha256':sha(a.c57) if arm=='c57' else None,'pybind11_headers':'torch/include/pybind11 (version from installed headers)','fp32_oracle_bad':bad,'fp16_finite':True,'graph_eager_mismatch':graph_bad,'graph_oracle_bad':graph_strict_bad,'eager':eager,'graph_replay':graph_t,'pid':os.getpid()}

p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--old',required=True);p.add_argument('--r1',required=True);p.add_argument('--c57',required=True);p.add_argument('--binary');p.add_argument('--out',default='/tmp/e035-child-unused.jsonl');p.add_argument('--iters',type=int,default=40);p.add_argument('--pairs',type=int,default=3);p.add_argument('--only-prefix');p.add_argument('--only-M',type=int,choices=(1,2,4));p.add_argument('--arm',choices=('old','r1','c57'));p.add_argument('--fixture');p.add_argument('--M',type=int,choices=(1,2,4));p.add_argument('--child',action='store_true');a=p.parse_args()
if a.child:
 r=child(a);print(json.dumps(r),flush=True);raise SystemExit(0)
if a.arm:raise SystemExit('use --child for single-arm invocation')
root=Path(a.fixtures);dirs=sorted(d for d in root.iterdir() if d.is_dir() and (d/'manifest.json').exists())
if len(dirs)!=6:raise SystemExit(f'expected six fixtures, got {len(dirs)}')
fixtures=[json.loads((d/'manifest.json').read_text())['prefix'] for d in dirs]
if a.only_prefix:fixtures=[x for x in fixtures if x==a.only_prefix]
ms=[a.only_M] if a.only_M else [1,2,4]
paths={'old':a.old,'r1':a.r1,'c57':a.c57};rows=[];out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
# Three cyclic orders; each measurement arm gets a new process. Save each child
# JSON immediately so interruption never loses completed evidence.
for fixture in fixtures:
 for M in ms:
  for pair in range(a.pairs):
   order=[('old','r1','c57')[i:]+('old','r1','c57')[:i] for i in range(3)][pair%3]
   timings={}
   for arm in order:
    cmd=[sys.executable,str(HERE/'paired_isolated.py'),'--child','--fixtures',a.fixtures,'--old',a.old,'--r1',a.r1,'--c57',a.c57,'--arm',arm,'--binary',paths[arm],'--fixture',fixture,'--M',str(M),'--iters',str(a.iters)]
    # child-only arguments are parsed explicitly below
    cp=subprocess.run(cmd,text=True,capture_output=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189')})
    if cp.returncode:raise RuntimeError(f'child failed arm={arm} fixture={fixture} M={M} rc={cp.returncode}\\nstdout={cp.stdout}\\nstderr={cp.stderr}')
    result=json.loads(cp.stdout.strip().splitlines()[-1]);timings[arm]=result
   row={'kind':'pair','fixture':fixture,'M':M,'pair':pair+1,'order':order,'arms':timings};rows.append(row)
   with out.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps({'fixture':fixture,'M':M,'pair':pair+1,'order':order,'eager_us':{k:v['eager']['median_us'] for k,v in timings.items()},'graph_us':{k:v['graph_replay']['median_us'] for k,v in timings.items()},'pids':{k:v['pid'] for k,v in timings.items()}}),flush=True)
