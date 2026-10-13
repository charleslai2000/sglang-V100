#!/usr/bin/env python3
import json,torch
from pathlib import Path
from torch.utils.cpp_extension import load
r=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures/43-mlp-down_proj-call0000');m=json.load(open(r/'manifest.json'))
def t(n,d):
 z=m['tensors'].get(n,m.get(n));f=r/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(*z['shape']).cuda()
x=t('x',torch.float16);qw=t('qweight',torch.int32);qz=t('qzeros',torch.int32);s=t('scales',torch.float16);g=t('g_idx',torch.int32);capt=t('s1_output',torch.float16)
ext=load(name='g017_q4_fp32_candidate',sources=[str(Path(__file__).with_name('packed_fp32_candidate.cu'))],extra_cuda_cflags=['-O3','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-build',verbose=False)
y=ext.packed_fp32_gptq(x,qw,qz,s,g,True)
K,N=x.shape[1],qw.shape[1];q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8):q[b::8]=(qw>>(b*4))&15
z=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
for b in range(8):z[:,b::8]=(qz>>(b*4))&15
# Independent unpack/layout invariants.
assert torch.equal(q[0:64:8],(qw[0:8] & 15))
assert torch.equal(z[:,0::8],qz & 15)
w64=(q.double()-(z[g.long()]+1).double())*s.double()[g.long()]
ref64=x.double()@w64; ref32=ref64.float();lim=1e-4+5e-5*ref32.abs();err=y-ref32;bad=torch.nonzero(err.abs()>lim)
print(json.dumps({'shape':list(x.shape)+[N],'bad':int(bad.shape[0]),'max_abs_fp32':float(err.abs().max()),'rel_l2':float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(ref32)),'rows':[{ 'm':int(i),'n':int(j),'y':float(y[i,j]),'ref64':float(ref64[i,j]),'ref32':float(ref32[i,j]),'capt':float(capt[i,j]),'err':float(err[i,j]),'limit':float(lim[i,j]),'relative_error':float(err[i,j]/ref32[i,j]) if ref32[i,j]!=0 else None} for i,j in bad[:20].tolist()]}))
# Compute exact outputs for each output with different grouping only at the offending coordinates.
if bad.numel():
 for i,j in bad.tolist():
  qcol=q[:,j].double(); zcol=z[g.long(),j].double(); sc=s[g.long(),j].double(); xv=x[i].double(); terms=xv*(qcol-zcol-1)*sc
  # Reproduce candidate chunking and alternate reduction precisions exactly at this output.
  chunks=terms.reshape(-1,256)
  chunk64=chunks.sum(1)
  chunk32=chunks.float().sum(1)
  chunk_round=(chunk32.double()).sum()
  serial32=terms.float().sum()
  vals={'m':i,'n':j,'abs_terms_sum':float(terms.abs().sum()),'sum_fp64':float(terms.sum()),'sum_fp32torch':float((x[i].float()*(q[:,j].float()-(z[g.long(),j].float()+1))*s[g.long(),j].float()).sum()),'termwise_fp32_sum':float(serial32),'fp64_chunks_then_fp64_reduce':float(chunk64.sum()),'fp64_chunks_then_fp32_reduce':float(chunk64.float().sum()),'fp32_chunks_then_fp64_reduce':float(chunk_round),'fp32_chunks_then_fp32_reduce':float(chunk32.sum()),'candidate32':float(y[i,j]),'oracle64':float(ref64[i,j])}
  print(json.dumps(vals))
