#!/usr/bin/env python3
import argparse,hashlib,importlib.util,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--module',required=True);p.add_argument('--c57',required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('common_ops',a.c57);cm=importlib.util.module_from_spec(s);s.loader.exec_module(cm)
s=importlib.util.spec_from_file_location('e035_grouped_q4_parallel_sm70',a.module);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
r=Path(a.fixture);m=json.loads((r/'manifest.json').read_text());t=m['tensors']
def ld(n,d):
 f=r/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(t[n]['shape']).cuda()
x4=ld('x',torch.float16);qw0=ld('qweight',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1);M=4;x=x4[:M].contiguous();K,N=qw0.shape[0]*8,qw0.shape[1];G=sc.shape[0]
perm=torch.argsort(g,stable=True).int();q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8):q[b::8]=(qw0>>(4*b))&15
qg=q.index_select(0,perm);qw=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
for b in range(8):qw|=qg[b::8]<<(4*b)
partial=torch.empty((G,M,N),device='cuda',dtype=torch.float32);gx=torch.empty_like(x);out=torch.empty((M,N),device='cuda',dtype=torch.float32)
mod.grouped_q4_sm70_parallel(x,qw,sc,perm,partial,gx,out,True);torch.cuda.synchronize()
print(json.dumps({'event':'E035_PARALLEL_NCU_READY','fixture':m['prefix'],'M':M,'K':K,'N':N,'G':G,'expected_partial_bytes':G*M*N*4,'group_grid':[N//16,(M+15)//16,G],'binary_sha256':hashlib.sha256(Path(a.module).read_bytes()).hexdigest(),'c57_sha256':hashlib.sha256(Path(a.c57).read_bytes()).hexdigest()}),flush=True)
for _ in range(3):mod.grouped_q4_sm70_parallel(x,qw,sc,perm,partial,gx,out,True)
torch.cuda.synchronize()
