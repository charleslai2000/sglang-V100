#!/usr/bin/env python3
"""Forensic comparison of candidate FP32 output against FP32 and FP64 oracles."""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('common_ops',a.module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path(a.fixtures);results=[]
for d in sorted(root.iterdir()):
 if not d.is_dir():continue
 m=json.load(open(d/'manifest.json'))
 def load(n,dt):
  md=m['tensors'][n];f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(*md['shape']).cuda()
 x=load('x',torch.float16);qw=load('qweight',torch.int32);qz=load('qzeros',torch.int32);sc=load('scales',torch.float16);g=load('g_idx',torch.int32)
 # Numerical authority is explicitly the pre-cast FP32 accumulator. Never
 # infer it by promoting the production FP16 result back to FP32.
 y=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,sc,g,True);torch.cuda.synchronize()
 K,N=x.shape[1],qw.shape[1]
 q=torch.empty((K,N),device='cuda',dtype=torch.int32)
 for b in range(8):q[b::8]=(qw>>(4*b))&15
 z=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>>(4*b))&15
 # Preserve both historical FP32 reference and authoritative high precision.
 w32=(q.float()-(z[g.long()]+1).float())*sc.float()[g.long()]
 ref32=x.float()@w32
 w64=(q.double()-(z[g.long()]+1).double())*sc.double()[g.long()]
 ref64=(x.double()@w64).float()
 yhalf=y.half().float();lim32=1e-4+5e-5*ref32.abs();lim64=1e-4+5e-5*ref64.abs()
 bad32=torch.nonzero((y-ref32).abs()>lim32);bad64=torch.nonzero((y-ref64).abs()>lim64);bhalf=torch.nonzero((yhalf-ref32).abs()>lim32)
 row={'fixture':m['prefix'],'reference_contract':'E026 numerical authority v2: FP64 unpack/dequant/GEMM -> FP32 cast; frozen strict 1e-4+5e-5*abs(reference) applies to actual pre-cast candidate FP32 accumulator','accumulator_boundary':'candidate custom op return_fp32=True; candidate_dtype asserted torch.float32','manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'candidate_module_sha256':hashlib.sha256(Path(a.module).read_bytes()).hexdigest(),'x_dtype':str(x.dtype),'qw_dtype':str(qw.dtype),'scale_dtype':str(sc.dtype),'candidate_dtype':str(y.dtype),'ref32_dtype':str(ref32.dtype),'ref64_cast_dtype':str(ref64.dtype),'fp32_bad_vs_fp32_ref':len(bad32),'fp32_bad_vs_fp64_ref_cast32':len(bad64),'fp16_cast_bad_vs_fp32_ref':len(bhalf),'fp32_ref_max_abs_delta':float((y-ref32).abs().max()),'fp64_ref_max_abs_delta':float((y-ref64).abs().max()),'fp32_vs_fp64_ref_max_abs':float((ref32-ref64).abs().max()),'fp32_bad_coords_first20':[[int(i),int(j)] for i,j in bad32[:20].tolist()],'fp64_bad_coords_first20':[[int(i),int(j)] for i,j in bad64[:20].tolist()]}
 # Representative coordinates from first bad FP64 comparisons; inspect actual packed values and term contribution decomposition.
 reps=bad64[:3].tolist(); details=[]
 for ii,jj in reps:
  i,j=int(ii),int(jj);gi=g.long();qcol=q[:,j].long();zcol=z[gi,j].long();scol=sc[gi,j]
  diff=qcol-zcol-1
  wcol32=diff.float()*scol.float();wcol64=diff.double()*scol.double()
  xrow32=x[i].float();xrow64=x[i].double()
  prod32=xrow32*wcol32; prod64=xrow64*wcol64
  details.append({'m':i,'n':j,'candidate':float(y[i,j]),'fp16_candidate':float(yhalf[i,j]),'ref32':float(ref32[i,j]),'ref64_cast32':float(ref64[i,j]),'limit32':float(lim32[i,j]),'candidate_error32':float(y[i,j]-ref32[i,j]),'q_first8':qcol[:8].tolist(),'zero_plus1_first8':(zcol[:8]+1).tolist(),'gidx_first8':gi[:8].tolist(),'scales_first8':scol[:8].float().tolist(),'weight32_first8':wcol32[:8].tolist(),'term32_first8':prod32[:8].tolist(),'term64_first8':prod64[:8].tolist(),'matmul_dot_fp32_serial_first8':float(prod32[:8].sum()),'dot_fp64_first8':float(prod64[:8].sum())})
 row['representative_terms']=details;results.append(row);print(json.dumps(row),flush=True)
Path(a.out).write_text(''.join(json.dumps(r)+'\n' for r in results))
