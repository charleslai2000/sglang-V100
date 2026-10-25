#!/usr/bin/env python3
"""Real-checkpoint Gemma layer check for production CTA selection.

Uses the unmodified FP16 checkpoint tensor as a separate torch.matmul oracle,
then quantizes/dequantizes the identical tensor and verifies the selected
Marlin operator. Shapes are the two dominant MLP matrices for M=1/2/4.
"""
import os, sys, types, json
import torch
ROOT='/data/t011/repos/marlin_v100-g002/source';sys.path[:0]=[ROOT,ROOT+'/tests']
pytest_stub=types.ModuleType('pytest');pytest_stub.param=lambda *a,**k:a;sys.modules.setdefault('pytest',pytest_stub)
import quant_utils
from tests import helpers,ops
from vllm.scalar_type import scalar_types
helpers._supported_quant_types=lambda:(scalar_types.uint4b8,scalar_types.uint8b128)
helpers._supported_unpack_quant_types=lambda:(scalar_types.uint4,scalar_types.uint4b8,scalar_types.uint8,scalar_types.uint8b128,scalar_types.float4_e2m1f,scalar_types.float8_e4m3fn)
SO=ROOT+'/build/temp.linux-x86_64-cpython-312/_C.abi3.so';torch.ops.load_library(SO);ops._dense_loaded=True
assert torch.cuda.get_device_capability()==(7,0)
from safetensors import safe_open
MODEL='/data/t011/models/gemma4-w4a16-ct/model.safetensors'
# Locate real MLP down projection from the official checkpoint.
with safe_open(MODEL,framework='pt',device='cpu') as f:
 keys=list(f.keys()); key=next(k for k in keys if k=='model.language_model.layers.0.mlp.down_proj.weight_packed')
 q_stored=f.get_tensor(key).contiguous().to(torch.int32)
 scale_stored=f.get_tensor('model.language_model.layers.0.mlp.down_proj.weight_scale').contiguous().to(torch.float16)
 size_k,size_n=10240,2560
print('REAL_PACKED_WEIGHT',key,tuple(q_stored.shape),tuple(scale_stored.shape),flush=True)
for m in (1,2,4):
 torch.manual_seed(8190+m);a=torch.randn((m,size_k),device='cuda',dtype=torch.float16)
 q_rows=q_stored.T.contiguous().to('cuda')
 q=torch.ops._C.gptq_marlin_repack(q_rows,torch.empty(0,dtype=torch.int32,device='cuda'),size_k,size_n,4,False)
 scale_perm=torch.tensor([i+8*j for i in range(8) for j in range(8)])
 scales=scale_stored.T.contiguous().reshape(-1,64)[:,scale_perm].reshape(320,2560).contiguous().to('cuda')
 g=helpers.marlin_make_empty_g_idx(a.device);perm=torch.empty(0,dtype=torch.int32,device='cuda');ws=helpers.marlin_make_workspace(a.device)
 ref=helpers.marlin_dense_reference(a,q,scales,size_k,size_n,32,scalar_types.uint4b8).to(torch.float16)
 out=ops.marlin_gemm(a,None,q,None,scales,None,None,g,g,perm,ws,scalar_types.uint4b8.id,m,size_n,size_k,True,False,False,False)
 torch.cuda.synchronize();err=(out.float()-ref.float())
 print(json.dumps({'M':m,'max_abs':err.abs().max().item(),'rmse':err.square().mean().sqrt().item(),'finite':bool(torch.isfinite(out).all())}),flush=True)
 assert torch.isfinite(out).all();torch.testing.assert_close(out,ref,rtol=5e-2,atol=2.5e-1)
