#!/usr/bin/env python3
import argparse,importlib.util,json,torch
from pathlib import Path
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--build',required=True);p.add_argument('--c57',required=True);a=p.parse_args()
osmod=importlib.util
# Register exact accepted extension to make identities/ABI deterministic (prototype calls own pybind op).
s=importlib.util.spec_from_file_location('common_ops',a.c57);common=importlib.util.module_from_spec(s);s.loader.exec_module(common)
mod=load(name='e035_grouped_q4_sm70',sources=[str(Path(__file__).with_name('grouped_q4_sm70.cu'))],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory=a.build,verbose=False)
r=Path(a.fixture);m=json.loads((r/'manifest.json').read_text());t=m['tensors']
def ld(n,d):
 f=r/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(t[n]['shape']).cuda()
x4=ld('x',torch.float16);qw0=ld('qweight',torch.int32);s=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1);M=4;x=x4[:M].contiguous();K,N=x.shape[1],qw0.shape[1];G=s.shape[0]
perm=torch.argsort(g,stable=True).int();q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8):q[b::8]=(qw0>>(4*b))&15
qg=q.index_select(0,perm);qw=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
for b in range(8):qw|=qg[b::8]<<(4*b)
gather=torch.empty_like(x);out=torch.empty((M,N),device='cuda',dtype=torch.float32)
# warmup then single named WMMA launch is target of NCU filter.
mod.grouped_q4_sm70(x,qw,s,perm,out,gather,out,True);torch.cuda.synchronize()
for _ in range(2):mod.grouped_q4_sm70(x,qw,s,perm,out,gather,out,True)
torch.cuda.synchronize();print(json.dumps({'fixture':m['prefix'],'M':M,'N':N,'K':K,'G':G,'prototype_sha256':__import__('hashlib').sha256(Path(mod.__file__).read_bytes()).hexdigest(),'c57_sha256':__import__('hashlib').sha256(Path(a.c57).read_bytes()).hexdigest(),'grid_expected':[N//16,(M+15)//16,1]}),flush=True)
