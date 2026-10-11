#!/usr/bin/env python3
"""Bounded, existing-op NCU target for saved real Falcon S1 GPTQ M4 fixtures."""
import json, os, sys, hashlib
from pathlib import Path
import torch, sgl_kernel

fixture=Path(sys.argv[1]); manifest=json.loads((fixture/'manifest.json').read_text())

def load(name,dtype):
    info=manifest['tensors'][name]; p=fixture/f'{name}.bin'
    t=torch.from_file(str(p),shared=False,size=p.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(*info['shape']).cuda()
    assert torch.isfinite(t).all() if t.dtype.is_floating_point else True
    return t
x=load('x',torch.float16); qw=load('qweight',torch.int32); qz=load('qzeros',torch.int32); sc=load('scales',torch.float16); gi=load('g_idx',torch.int32)
module=Path(sgl_kernel.common_ops.__file__); print(json.dumps({'fixture':manifest['prefix'],'module':str(module),'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'target_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'shape':list(x.shape)},sort_keys=True),flush=True)
def run(): return torch.ops.sgl_kernel.gptq_gemm(x,qw.view(torch.uint32),qz.view(torch.uint32),sc,gi,False,4)
y=run();torch.cuda.synchronize();assert torch.isfinite(y).all()
for _ in range(3): run()
torch.cuda.synchronize()
for i in range(16): run()
torch.cuda.synchronize()
print(json.dumps({'status':'complete','warmups':3,'profile_launches':16,'output_shape':list(y.shape),'finite':True}),flush=True)
