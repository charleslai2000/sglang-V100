#!/usr/bin/env python3
import argparse,importlib.util,json,torch,hashlib
from pathlib import Path
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--build',required=True);p.add_argument('--c57',required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('common_ops',a.c57);cm=importlib.util.module_from_spec(s);s.loader.exec_module(cm)
mod=load(name='e035_grouped_q4_sm70',sources=[str(Path(__file__).with_name('grouped_q4_sm70.cu'))],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory=a.build,verbose=False)
r=Path(a.fixture);m=json.loads((r/'manifest.json').read_text());t=m['tensors']
def ld(n,d):
 f=r/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(t[n]['shape']).cuda()
x4=ld('x',torch.float16);qw0=ld('qweight',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1);M=4;x=x4[:M].contiguous();K,N=x.shape[1],qw0.shape[1]
perm=torch.argsort(g,stable=True).int();q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8):q[b::8]=(qw0>>(4*b))&15
qg=q.index_select(0,perm);qw=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
for b in range(8):qw|=qg[b::8]<<(4*b)
gx=torch.empty_like(x);scratch=torch.empty((M,N),device='cuda',dtype=torch.float32);out=torch.empty_like(scratch)
mod.grouped_q4_sm70(x,qw,sc,perm,scratch,gx,out,True);torch.cuda.synchronize()
print(json.dumps({'event':'E035_NCU_READY','fixture':m['prefix'],'M':M,'K':K,'N':N,'grid':[N//16,(M+15)//16,1],'prototype_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),'c57_sha256':hashlib.sha256(Path(a.c57).read_bytes()).hexdigest()}),flush=True)
for _ in range(20):mod.grouped_q4_sm70(x,qw,sc,perm,scratch,gx,out,True)
torch.cuda.synchronize()
