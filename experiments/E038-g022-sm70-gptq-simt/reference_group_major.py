#!/usr/bin/env python3
"""CPU-only exact-layout reference for group-major GPTQ-Q4 conversion."""
import argparse, hashlib, json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--fixture-root',default='/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');p.add_argument('--fixture',default='all');a=p.parse_args()
root=Path(a.fixture_root);names=sorted(x.parent.name for x in root.glob('*/manifest.json')) if a.fixture=='all' else [a.fixture]
def load_tensor(directory,manifest,name,dtype):
 t=manifest['tensors'].get(name,manifest.get(name));f=directory/(name+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(*t['shape']).contiguous()
rows=[]
for fixture in names:
 r=root/fixture;m=json.loads((r/'manifest.json').read_text())
 qw=load_tensor(r,m,'qweight',torch.int32);qz=load_tensor(r,m,'qzeros',torch.int32);sc=load_tensor(r,m,'scales',torch.float16);gi=load_tensor(r,m,'g_idx',torch.int32);x=load_tensor(r,m,'x',torch.float16)
 K,N=x.shape[1],qw.shape[1];group=128;G=qz.shape[0]
 assert K%group==0 and G==K//group and N%8==0 and qw.shape==(K//8,N) and gi.numel()==K
 qcodes=((qw.to(torch.int64).repeat_interleave(8,dim=0) >> ((torch.arange(K,dtype=torch.int64)%8)*4).view(K,1))&15).to(torch.int16)
 perm=[]
 for g in range(G):
  ids=torch.where(gi==g)[0].tolist();assert len(ids)==group,(fixture,g,len(ids));perm+=ids
 perm=torch.tensor(perm,dtype=torch.int64);assert torch.equal(gi[perm],torch.arange(G).repeat_interleave(group))
 qgm=qcodes[perm].reshape(G,group,N).contiguous();qpack=torch.zeros((G,group,N//8),dtype=torch.int32)
 for j in range(8):qpack |= (qgm[:,:,j::8].to(torch.int32)&15)<<(4*j)
 qround=torch.empty_like(qgm)
 for j in range(8):qround[:,:,j::8]=((qpack.to(torch.int64)>>(4*j))&15).to(torch.int16)
 assert torch.equal(qround,qgm)
 z=((qz.to(torch.int64).unsqueeze(-1)>>(torch.arange(8,dtype=torch.int64)*4).view(1,1,8))&15).reshape(G,N).to(torch.int16)
 xperm=x[:,perm].contiguous()
 w=((qcodes.to(torch.int32)-(z[gi.long()].to(torch.int32)+1)).to(torch.float64)*sc[gi.long()].to(torch.float64));ref=x.to(torch.float64)@w
 wgm=((qgm.to(torch.int32)-(z[:,None,:].to(torch.int32)+1)).to(torch.float64)*sc[:,None,:].to(torch.float64));refgm=xperm.to(torch.float64)@wgm.reshape(K,N)
 assert torch.equal(ref,refgm),(fixture,(ref-refgm).abs().max().item())
 rows.append({'fixture':fixture,'manifest_sha256':hashlib.sha256((r/'manifest.json').read_bytes()).hexdigest(),'qweight_shape':list(qw.shape),'qzeros_shape':list(qz.shape),'scales_shape':list(sc.shape),'g_idx_shape':list(gi.shape),'group_size':group,'groups':G,'group_major_shape':list(qpack.shape),'nonidentity_perm_entries':int((perm!=torch.arange(K)).sum()),'packed_codes_exact':True,'activation_gather_shape':list(xperm.shape),'reference_fp64_exact':True,'reference_max_abs_diff':float((ref-refgm).abs().max()),'qzero_plus_one_preserved':True})
print(json.dumps({'schema':'E038-cpu-group-major-reference-v1','rows':rows},indent=2))
