#!/usr/bin/env python3
"""Identical event-timed operator harness for full common_ops or packed prototype."""
import argparse,hashlib,importlib.util,json,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--module',required=True);p.add_argument('--iters',type=int,default=200);a=p.parse_args()
mod_path=Path(a.module)
if 'candidate-integrated' in mod_path.name:
 spec=importlib.util.spec_from_file_location('common_ops',str(mod_path));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kind='candidate'
elif 'g017_q4_fp32_candidate' in mod_path.name:
 spec=importlib.util.spec_from_file_location('g017_q4_fp32_candidate',str(mod_path));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kind='prototype'
else:
 spec=importlib.util.spec_from_file_location('common_ops',str(mod_path));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kind='production'
r=Path(a.fixture);m=json.load(open(r/'manifest.json'))
def t(n,d):
 tm=m['tensors'].get(n,m.get(n));f=r/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=d).element_size(),dtype=d).view(*tm['shape']).cuda()
x=t('x',torch.float16);qw=t('qweight',torch.int32);qz=t('qzeros',torch.int32);s=t('scales',torch.float16);g=t('g_idx',torch.int32);capt=t('s1_output',torch.float16)
def op():
 if kind=='candidate':return torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,s,g)
 if kind=='production':return torch.ops.sgl_kernel.gptq_gemm(x,qw.view(torch.uint32),qz.view(torch.uint32),s,g,False,4)
 return mod.packed_fp32_gptq(x,qw,qz,s,g,False)
for _ in range(5):y=op()
torch.cuda.synchronize();times=[]
for _ in range(a.iters):
 st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();y=op();en.record();torch.cuda.synchronize();times.append(st.elapsed_time(en))
t=torch.tensor(times)
print(json.dumps({'fixture':m['prefix'],'kind':kind,'module':str(mod_path),'sha256':hashlib.sha256(open(mod_path,'rb').read()).hexdigest(),'median_ms':float(t.median()),'p10_ms':float(torch.quantile(t,.1)),'p90_ms':float(torch.quantile(t,.9)),'iters':a.iters,'finite':bool(torch.isfinite(y).all()),'max_abs_vs_captured_s1':float((y-capt).abs().max()),'shape':list(y.shape)}))
