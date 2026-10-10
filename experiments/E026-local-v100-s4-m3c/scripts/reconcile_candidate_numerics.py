#!/usr/bin/env python3
"""Run frozen G017 FP64-derived strict FP32 accumulator oracle on six fixtures."""
import argparse, hashlib, json
from pathlib import Path
import torch
from importlib.machinery import ExtensionFileLoader
p=argparse.ArgumentParser(); p.add_argument('--module', required=True); p.add_argument('--fixtures', required=True); p.add_argument('--out', required=True); a=p.parse_args()
module_path=Path(a.module).resolve(); ext=ExtensionFileLoader('common_ops',str(module_path)).load_module(); root=Path(a.fixtures)
rows=[]
for d in sorted(root.iterdir()):
    if not d.is_dir() or not (d/'manifest.json').exists(): continue
    m=json.loads((d/'manifest.json').read_text())
    def load(n,dt):
        info=m['tensors'][n]; f=d/(n+'.bin')
        return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(*info['shape']).cuda()
    x=load('x',torch.float16); qw=load('qweight',torch.int32); qz=load('qzeros',torch.int32); scales=load('scales',torch.float16); g=load('g_idx',torch.int32)
    y=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,scales,g,True); diag=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,scales,g,False); torch.cuda.synchronize()
    K,N=x.shape[1],qw.shape[1]
    q=torch.empty((K,N),device='cuda',dtype=torch.int32)
    for b in range(8): q[b::8]=(qw>>(4*b))&15
    z=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
    for b in range(8): z[:,b::8]=(qz>>(4*b))&15
    w64=(q.double()-(z[g.long()]+1).double())*scales.double()[g.long()]
    ref=(x.double()@w64).float(); limit=1e-4+5e-5*ref.abs(); err=(y-ref)
    bad=int((err.abs()>limit).sum().item())
    diag_ref=x.float() @ ((q.float()-(z[g.long()]+1).float())*scales.float()[g.long()]); diag_bad=int(((diag-diag_ref).abs() > 1e-4+5e-5*diag_ref.abs()).sum().item())
    hist=next((r for r in map(json.loads,open('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-final-fp64-authority.jsonl')) if r.get('fixture')==m['prefix']),None)
    row={'fixture':m['prefix'],'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'module_sha256':hashlib.sha256(module_path.read_bytes()).hexdigest(),'candidate_dtype':str(y.dtype),'reference_dtype':str(ref.dtype),'strict_bad':bad,'historical_fp32_gemm_diagnostic_bad':(hist.get('fp32_bad_vs_fp32_ref') if hist else None),'historical_fp32_gemm_bad_vs_fp64_cast32':(hist.get('fp32_bad_vs_fp64_ref_cast32') if hist else None),'historical_fp32_gemm_max_abs':(hist.get('fp32_ref_max_abs_delta') if hist else None),'historical_fp32_vs_fp64_cast32_max_abs':(hist.get('fp32_vs_fp64_ref_max_abs') if hist else None),'historical_fp32_bad_coords_first20':(hist.get('fp32_bad_coords_first20') if hist else None),'local_fp32_gemm_diagnostic_bad':diag_bad,'local_fp32_gemm_max_abs':float((diag-diag_ref).abs().max()),'max_abs':float(err.abs().max()),'finite':bool(torch.isfinite(y).all()),'pass':bad==0}
    rows.append(row); print(json.dumps(row),flush=True)
Path(a.out).write_text(''.join(json.dumps(r)+'\n' for r in rows))
if len(rows)!=6 or not all(r['pass'] and r['finite'] for r in rows): raise SystemExit('strict oracle failed')
