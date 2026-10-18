#!/usr/bin/env python3
"""Stable fixed-binary CUDA target for NCU Application Replay; no JIT rebuild."""
import hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures')
BIN=Path('/mnt/data/sglang-V100/t011-build/qpn2_multiwarp.so')
FIXTURE=os.environ.get('T011_FIXTURE','43-mlp-gate_up_proj-call0000');M=int(os.environ.get('T011_M','4'));SPLIT=int(os.environ.get('T011_SPLIT','8'))
assert torch.cuda.device_count()==1
from gpu_uuid_lock import assert_cuda_uuid
assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189')
spec=importlib.util.spec_from_file_location('qpn2_multiwarp',BIN);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
d=ROOT/FIXTURE;mf=json.loads((d/'manifest.json').read_text());T=mf['tensors'];K=T['g_idx']['shape'][0];N=T['qweight']['shape'][1];G=T['scales']['shape'][0]
def load(name,dtype):
 p=d/(name+'.bin');return torch.from_file(str(p),shared=False,size=p.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(T[name]['shape']).cuda()
x=load('x',torch.float16)[:M].contiguous();qw=load('qweight',torch.int32);sc=load('scales',torch.float16);g=load('g_idx',torch.int32).reshape(-1)
qraw=qw.cpu().numpy().view(np.uint32);q=np.empty((K,N),np.uint8)
for b in range(8):q[b::8]=((qraw>>(4*b))&15).astype(np.uint8)
gcpu=g.cpu().numpy();perm=np.argsort(gcpu,kind='stable').astype(np.int32);tiles=N//32;g16=K//16
col=np.array([((l>>2)&3)*8+(l&3)+(4 if l&16 else 0) for l in range(32)]);ko=np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);pn=np.empty((tiles,g16,32,8),np.uint8)
for t in range(tiles):
 for kg in range(g16):
  sk=perm[(kg//8)*128+(kg%8)*16+ko];v=q[sk[:,None],(t*32+col)[None,:]].T;pn[t,kg]=v[:,0::2]|(v[:,1::2]<<4)
qt=torch.from_numpy(pn.view(np.int32).copy()).cuda();pt=torch.from_numpy(perm).cuda();gx=torch.empty_like(x);o=torch.empty((M,N),device='cuda',dtype=torch.float32)
y=mod.run(x,qt,sc,pt,gx,o,SPLIT);torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(json.dumps({'event':'T011_FIXED_BINARY_NCU_TARGET','fixture':FIXTURE,'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'M':M,'splitk':SPLIT,'binary_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),'uuid':str(torch.cuda.get_device_properties(0).uuid)}),flush=True)
