#!/usr/bin/env python3
import json,os,sys,torch,hashlib
from importlib.machinery import ExtensionFileLoader
ext='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-runtime/sgl_kernel/sm70/common_ops.abi3.so';mod=ExtensionFileLoader('common_ops',ext).load_module();root=sys.argv[1]; mode=sys.argv[2];os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='1' if mode=='candidate' else '0';os.environ['SGLANG_G017_Q4_FP32_CANDIDATE_PROBE']='0';m=json.load(open(root+'/manifest.json'))
def t(n,dt):
 q=m['tensors'][n];p=root+'/'+n+'.bin';return torch.from_file(p,shared=False,size=__import__('os').stat(p).st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(*q['shape']).cuda()
x=t('x',torch.float16);w=t('qweight',torch.int32);z=t('qzeros',torch.int32);s=t('scales',torch.float16);g=t('g_idx',torch.int32)
# This directly exercises the exact M4 dispatcher geometry and opt-in path.
if mode=='candidate':y=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,w,z,s,g,False)
else:y=torch.ops.sgl_kernel.gptq_gemm(x,w.view(torch.uint32),z.view(torch.uint32),s,g,False,4)
torch.cuda.synchronize();assert torch.isfinite(y).all()
for _ in range(5):
 if mode=='candidate':torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,w,z,s,g,False)
 else:torch.ops.sgl_kernel.gptq_gemm(x,w.view(torch.uint32),z.view(torch.uint32),s,g,False,4)
torch.cuda.synchronize();print(json.dumps({'route':mode,'fixture':m['prefix'],'shape':list(y.shape),'dtype':str(y.dtype),'finite':True,'extension_path':ext,'extension_sha256':hashlib.sha256(open(ext,'rb').read()).hexdigest()}),flush=True)
for _ in range(100):
 if mode=='candidate':torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,w,z,s,g,False)
 else:torch.ops.sgl_kernel.gptq_gemm(x,w.view(torch.uint32),z.view(torch.uint32),s,g,False,4)
torch.cuda.synchronize()
