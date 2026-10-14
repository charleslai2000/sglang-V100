#!/usr/bin/env python3
import argparse,json,torch
from pathlib import Path
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--iters',type=int,default=20);p.add_argument('--stress-factors',default='');a=p.parse_args();r=Path(a.fixture);m=json.load(open(r/'manifest.json'))
def load_t(n,dtype):
 tm=m['tensors'][n] if n in m['tensors'] else m[n]; f=r/(n+'.bin'); num=f.stat().st_size//torch.empty((),dtype=dtype).element_size(); return torch.from_file(str(f),shared=False,size=num,dtype=dtype).view(*tm['shape']).cuda()
x=load_t('x',torch.float16);qw=load_t('qweight',torch.int32);qz=load_t('qzeros',torch.int32);s=load_t('scales',torch.float16);g=load_t('g_idx',torch.int32);capt=load_t('s1_output',torch.float16)
# Use a private build tree, one compilation worker, and only sm70.
src=str(Path(__file__).with_name('packed_fp32_candidate.cu')); ext=load(name='g017_q4_fp32_candidate',sources=[src],extra_cuda_cflags=['-O3','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-build',verbose=False)
y=ext.packed_fp32_gptq(x,qw,qz,s,g,True);torch.cuda.synchronize();
# Mathematical oracle, only for observed fixture output; explicit no tolerance.
K,N=x.shape[1],qw.shape[1];q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8):q[b::8]=(qw>> (b*4))&15
z=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
for b in range(8):z[:,b::8]=(qz>>(b*4))&15
w64=(q.double()-(z[g.long()]+1).double())*s.double()[g.long()];ref64=x.double()@w64; ref=ref64.float()
err=(y.float()-ref); yhalf=y.half();herr=yhalf.float()-ref;captured_delta=(yhalf-capt).abs()
# Frozen strict elementwise criterion is applied to the FP32 accumulator.
# Separately report FP16 conversion behavior; do not apply the FP32 strict
# oracle to a quantized FP16 tensor.
lim=1e-4+5e-5*ref.abs();res={'fixture':m['prefix'],'candidate_vs_fp32_max_abs':float(err.abs().max()),'candidate_vs_fp32_rel_l2':float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(ref)),'candidate_vs_fp32_strict_bad':int((err.abs()>lim).sum()),'candidate_fp16_vs_fp32_max_abs':float(herr.abs().max()),'candidate_fp16_vs_fp32_strict_bad_not_acceptance':int((herr.abs()>lim).sum()),'captured_S1_vs_fp32_max_abs':float((capt.float()-ref).abs().max()),'candidate_vs_captured_s1_max_abs':float(captured_delta.max()),'candidate_finite':bool(torch.isfinite(y).all()),'output_shape':list(y.shape)}
# Stress same real weights/metadata under scaled fixture activations. Record all
# factors including overflow; no output/reference threshold changes.
stress=[]
if a.stress_factors:
 for factor in map(float,a.stress_factors.split(',')):
  ys=ext.packed_fp32_gptq(x*factor,qw,qz,s,g,True); yh=ys.half();torch.cuda.synchronize()
  stress.append({'factor':factor,'input_max_abs':float((x*factor).abs().max()),'fp32_finite':bool(torch.isfinite(ys).all()),'fp16_finite':bool(torch.isfinite(yh).all()),'fp16_max_abs':float(yh.abs().max())})
res['high_amplitude_stress']=stress
print(json.dumps(res))
# Operator timing includes tmp/output allocation, deterministic K-partial layout and reducer. CUDA Graph can capture this path later.
for _ in range(5):ext.packed_fp32_gptq(x,qw,qz,s,g,False)
torch.cuda.synchronize();st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
for _ in range(a.iters):ext.packed_fp32_gptq(x,qw,qz,s,g,False)
en.record();torch.cuda.synchronize();print(json.dumps({'fixture':m['prefix'],'iters':a.iters,'full_op_ms':st.elapsed_time(en)/a.iters,'includes_temp_allocation_and_two_kernels':True}))
