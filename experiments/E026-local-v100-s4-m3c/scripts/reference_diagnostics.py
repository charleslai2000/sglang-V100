#!/usr/bin/env python3
"""Compare captured/current S1 GPTQ route with the documented packed-FP32 oracle.

No acceptance tolerance is introduced here. This records existing F6B strict
criterion results and separates route repeatability from oracle disagreement.
"""
import argparse, hashlib, json
from pathlib import Path
import torch
from sgl_kernel.gemm import gptq_gemm

p=argparse.ArgumentParser(); p.add_argument('--fixture',required=True); p.add_argument('--repeats',type=int,default=5); a=p.parse_args(); root=Path(a.fixture)
m=json.loads((root/'manifest.json').read_text())
def load(name,dtype):
    tm=m['tensors'][name] if name in m['tensors'] else m[name]
    path=root/(name+'.bin'); count=path.stat().st_size//torch.empty((),dtype=dtype).element_size()
    return torch.from_file(str(path),shared=False,size=count,dtype=dtype).reshape(tm['shape']).cuda()
x=load('x',torch.float16); qw=load('qweight',torch.int32); qz=load('qzeros',torch.int32); sc=load('scales',torch.float16); gi=load('g_idx',torch.int32).long()
# Frozen GPTQ-v1 layout: qword packs eight consecutive K values; qzero word
# packs eight N values; decoded qzero is stored-minus-one, hence +1.
K,N=x.shape[1],qw.shape[1]; q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8): q[b::8]=torch.bitwise_and(torch.bitwise_right_shift(qw,b*4),15)
z=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
for b in range(8): z[:,b::8]=torch.bitwise_and(torch.bitwise_right_shift(qz,b*4),15)
# Independent high-precision mathematical oracle. All inputs are exact FP16 or
# integer values; FP64 prevents the reference GEMM reduction from dominating
# the strict per-element tolerance on small/cancellation-heavy outputs.
w=(q.double()-(z[gi]+1).double())*sc.double()[gi]
ref=(x.double()@w).float()
captured=load('s1_output',torch.float16).float()
def stats(v):
    d=v.float()-ref; lim=1e-4+5e-5*ref.abs()
    return {'max_abs':float(d.abs().max()),'relative_l2':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(ref)),
            'strict_bad':int((d.abs()>lim).sum()),'elements':d.numel(),'nonfinite':int((~torch.isfinite(v)).sum())}
outs=[]
for _ in range(a.repeats):
    y=gptq_gemm(x,qw.view(torch.uint32),qz.view(torch.uint32),sc,gi.int(),False,4)
    torch.cuda.synchronize(); outs.append(y.float())
repeat_deltas=[float((outs[i]-outs[0]).abs().max()) for i in range(1,len(outs))]
print(json.dumps({'fixture':root.name,'prefix':m['prefix'],'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
 'semantics':'q=(qweight >> 4*(k%8))&15; qzero=(qzeros >> 4*(n%8))&15; w=(q-(qzero[g_idx]+1))*scale[g_idx]; FP32 torch matmul',
 'captured_vs_fp32':stats(captured),'current_s1_vs_fp32':stats(outs[0]),
 'current_s1_vs_captured_max_abs':float((outs[0]-captured).abs().max()),
 'repeat_max_abs_vs_first':repeat_deltas,'repeat_bitwise_equal':all(torch.equal(outs[0],o) for o in outs[1:]),
 'finite_all':all(bool(torch.isfinite(o).all()) for o in outs)},),flush=True)
