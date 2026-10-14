#!/usr/bin/env python3
"""One identity-bound, CUDA-profiler-range capture for one full Linear call."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
from gpu_uuid_lock import assert_inherited, assert_cuda_uuid
p=argparse.ArgumentParser()
for k in ('arm','binary','fixtures','c57'):p.add_argument('--'+k,required=True)
p.add_argument('--gpu-uuid',default='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');p.add_argument('--lock-fd',type=int,required=True)
a=p.parse_args();assert_inherited(a.lock_fd,a.gpu_uuid)
if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise SystemExit('wrong visible GPU')
import torch
assert_cuda_uuid(torch,a.gpu_uuid)
arm=a.arm;binary=Path(a.binary).resolve();run_id=os.environ.get('E035_RUN_ID','unknown')
if arm=='c57':
 s=importlib.util.spec_from_file_location('common_ops',str(binary));mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod);fn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate;actual=Path(mod.__file__).resolve()
else:
 name='e035_grouped_q4_parallel_sm70' if arm=='parallel' else 'e035_grouped_q4_sm70';s=importlib.util.spec_from_file_location(name,str(binary));mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod);actual=Path(mod.__file__).resolve();fn=mod.grouped_q4_sm70_parallel if arm=='parallel' else mod.grouped_q4_sm70
if actual!=binary:raise RuntimeError(f'loaded {actual}, expected {binary}')
root=Path(a.fixtures);d=next(q for q in root.iterdir() if q.is_dir() and json.loads((q/'manifest.json').read_text())['prefix']=='43.mlp.down_proj');m=json.loads((d/'manifest.json').read_text());t=m['tensors']
def ld(n,dtype):
 f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(t[n]['shape']).cuda()
x=ld('x',torch.float16)[:4].contiguous();qw0=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1);K,N=x.shape[1],qw0.shape[1];G=sc.shape[0]
perm=torch.argsort(g,stable=True).int();q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8):q[b::8]=(qw0>>(4*b))&15
qg=q.index_select(0,perm);packed=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
for b in range(8):packed|=qg[b::8]<<(4*b)
part=torch.empty((G,4,N) if arm=='parallel' else (4,N),device='cuda',dtype=torch.float32);gx=torch.empty_like(x);out=torch.empty((4,N),device='cuda',dtype=torch.float32)
def call():
 if arm=='c57':return fn(x,qw0,qz,sc,g,False)
 return fn(x,packed,sc,perm,part,gx,out,False)
for _ in range(8):call()
torch.cuda.synchronize()
print(json.dumps({'event':'ready','run_id':run_id,'arm':arm,'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'fixture':m['prefix'],'M':4,'K':K,'N':N,'grid':([N//16,1,G] if arm=='parallel' else ([N//16,1,1] if arm=='r1' else None))}),flush=True)
torch.cuda.profiler.start();call();torch.cuda.synchronize();torch.cuda.profiler.stop()
