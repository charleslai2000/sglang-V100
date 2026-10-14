#!/usr/bin/env python3
"""Alternating complete-call timings for E035 and exact accepted c57."""
import argparse,hashlib,importlib.util,json,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--prototype',required=True);p.add_argument('--c57',required=True);p.add_argument('--iters',type=int,default=40);p.add_argument('--pairs',type=int,default=3);p.add_argument('--partial-groups',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.fixtures);fixture_dirs=sorted(d for d in root.iterdir() if d.is_dir() and (d/'manifest.json').exists())
if len(fixture_dirs)!=6:raise SystemExit(f'expected six archived real fixtures, got {len(fixture_dirs)}')
sp=importlib.util.spec_from_file_location('e035_grouped_q4_sm70',a.prototype);proto=importlib.util.module_from_spec(sp);sp.loader.exec_module(proto)
sc=importlib.util.spec_from_file_location('common_ops',a.c57);c57=importlib.util.module_from_spec(sc);sc.loader.exec_module(c57)
import os
os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='1';os.environ['SGLANG_G017_Q4_FP32_CANDIDATE_PROBE']='0'
allrows=[]
for r in fixture_dirs:
 m=json.loads((r/'manifest.json').read_text());t=m['tensors']
 def ld(n,d):
  f=r/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(t[n]['shape']).cuda()
 x4=ld('x',torch.float16);qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);s=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1)
 for M in (1,2,4):
  x=x4[:M].contiguous();K,N=x.shape[1],qw.shape[1];G=s.shape[0]
  perm=torch.argsort(g,stable=True).int();q=torch.empty((K,N),device='cuda',dtype=torch.int32)
  for b in range(8):q[b::8]=(qw>>(4*b))&15
  qg=q.index_select(0,perm);packed=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
  for b in range(8):packed|=qg[b::8]<<(4*b)
  part=torch.empty((G,M,N) if a.partial_groups else (M,N),device='cuda',dtype=torch.float32);gx=torch.empty_like(x);out=torch.empty((M,N),device='cuda',dtype=torch.float32)
  def new():return proto.grouped_q4_sm70(x,packed,s,perm,part,gx,out,False)
  def old():return torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,s,g,False)
  yn=new();yo=old();torch.cuda.synchronize();records=[]
  for rep in range(a.pairs):
   order=('new','c57') if rep%2==0 else ('c57','new');vals={}
   for name in order:
    fn=new if name=='new' else old
    for _ in range(3):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(a.iters):
     st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();fn();en.record();en.synchronize();times.append(st.elapsed_time(en)*1000)
    vals[name]={'median_us':float(torch.tensor(times).median()),'p10_us':float(torch.quantile(torch.tensor(times),.1)),'p90_us':float(torch.quantile(torch.tensor(times),.9)),'n':len(times)}
   records.append({'pair':rep+1,'order':list(order),'timings':vals})
  row={'fixture':m['prefix'],'fixture_manifest_sha256':hashlib.sha256((r/'manifest.json').read_bytes()).hexdigest(),'M':M,'K':K,'N':N,'pairs':records,'prototype_sha256':hashlib.sha256(Path(a.prototype).read_bytes()).hexdigest(),'c57_sha256':hashlib.sha256(Path(a.c57).read_bytes()).hexdigest()}
  allrows.append(row);print(json.dumps(row),flush=True)
  Path(a.out).write_text(''.join(json.dumps(z)+'\n' for z in allrows))
