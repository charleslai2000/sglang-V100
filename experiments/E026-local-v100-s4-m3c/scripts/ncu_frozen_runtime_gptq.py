#!/usr/bin/env python3
"""Target exact c4 fixture through the selected installed runtime and NCU."""
import json, os, sys, torch
from pathlib import Path
from importlib.machinery import ExtensionFileLoader
mode, root = sys.argv[1], Path(sys.argv[2])
m=json.loads((root/'manifest.json').read_text())
def t(n,d):
 q=m['tensors'][n]; p=root/(n+'.bin'); return torch.from_file(str(p),shared=False,size=p.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(*q['shape']).cuda()
x=t('x',torch.float16); w=t('qweight',torch.uint32); z=t('qzeros',torch.uint32); s=t('scales',torch.float16); g=t('g_idx',torch.int32)
if mode=='candidate':
 ext=os.environ['G017_CANDIDATE_MODULE']; ExtensionFileLoader('common_ops',ext).load_module()
 op=lambda: torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,w.view(torch.int32),z.view(torch.int32),s,g,False)
 route='candidate '+ext
else:
 from sgl_kernel import gptq_gemm
 op=lambda: torch.ops.sgl_kernel.gptq_gemm(x,w,z,s,g,False,4)
 route='frozen installed dispatcher gptq_gemm'
y=op();torch.cuda.synchronize();assert torch.isfinite(y).all()
for _ in range(8):op()
torch.cuda.synchronize();print(json.dumps({'route':route,'fixture':m['prefix'],'shape':list(y.shape),'dtype':str(y.dtype),'finite':True,'extension':getattr(__import__('sgl_kernel'),'common_ops').__file__}),flush=True)
for _ in range(30):op()
torch.cuda.synchronize()
