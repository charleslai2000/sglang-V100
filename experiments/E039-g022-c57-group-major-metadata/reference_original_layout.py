#!/usr/bin/env python3
"""CPU-only E039 stable group-major permutation retaining [K/8,N] GPTQ storage."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');p.add_argument('--fixture',default='all');a=p.parse_args();root=Path(a.root);names=sorted(d.parent.name for d in root.glob('*/manifest.json')) if a.fixture=='all' else [a.fixture]
def load(d,m,n,dt):
 t=m['tensors'][n];f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(t['shape']).contiguous()
rows=[]
for name in names:
 d=root/name;m=json.loads((d/'manifest.json').read_text());qw=load(d,m,'qweight',torch.int32);qz=load(d,m,'qzeros',torch.int32);sc=load(d,m,'scales',torch.float16);gi=load(d,m,'g_idx',torch.int32);x=load(d,m,'x',torch.float16);K,N=x.shape[1],qw.shape[1];G=qz.shape[0]
 assert qw.shape==(K//8,N) and qz.shape==(G,N//8) and sc.shape==(G,N) and K==G*128
 perm=[]
 for g in range(G):
  ids=torch.where(gi==g)[0].tolist();assert len(ids)==128,(name,g,len(ids));perm.extend(ids)
 perm=torch.tensor(perm,dtype=torch.int64);assert torch.equal(gi[perm],torch.arange(G).repeat_interleave(128))
 packed=torch.zeros_like(qw)
 for dst in range(K):
  src=int(perm[dst]);code=((qw[src//8].to(torch.int64)>>(4*(src%8)))&15).to(torch.int32)
  packed[dst//8] |= code << (4*(dst%8))
 unpack=((packed.to(torch.int64).repeat_interleave(8,0) >> ((torch.arange(K)%8)*4).view(K,1))&15).to(torch.int32)
 orig=((qw.to(torch.int64).repeat_interleave(8,0) >> ((torch.arange(K)%8)*4).view(K,1))&15).to(torch.int32)
 assert torch.equal(unpack,orig[perm])
 # qzeros/scales group rows are preserved; verify group-wise effective weights
 z=((qz.to(torch.int64).unsqueeze(-1)>>(torch.arange(8,dtype=torch.int64)*4).view(1,1,8))&15).reshape(G,N).to(torch.int32)
 xp=x[:,perm].contiguous();w=((orig-z[gi.long()]-1).double()*sc[gi.long()].double());ref=x.double()@w
 wg=((unpack-z.repeat_interleave(128,0)-1).double()*sc.repeat_interleave(128,0).double());out=xp.double()@wg
 assert torch.equal(ref,out),(name,(ref-out).abs().max().item())
 assert torch.equal(xp,x[:,perm])
 rows.append({'fixture':name,'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'shapes':{'qweight':list(qw.shape),'qzeros':list(qz.shape),'scales':list(sc.shape),'g_idx':list(gi.shape)},'perm_nonidentity':int((perm!=torch.arange(K)).sum()),'packed_codes_exact':True,'layout':'GPTQ [K/8,N], N-contiguous','qzero_plus_one':True,'scale_group_exact':True,'activation_fp16_bitwise':True,'fp64_outputs_bitwise_equal':True,'max_abs':float((ref-out).abs().max())})
print(json.dumps({'schema':'E039-original-layout-group-major-cpu-v1','rows':rows},indent=2))
