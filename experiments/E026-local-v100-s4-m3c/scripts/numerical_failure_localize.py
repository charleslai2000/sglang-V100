#!/usr/bin/env python3
"""Locate candidate FP32 and FP16 strict-oracle failures on real GPTQ fixtures."""
import argparse, importlib.util, json
from pathlib import Path
import torch
P=argparse.ArgumentParser();P.add_argument('--module',required=True);P.add_argument('--fixtures',required=True);P.add_argument('--out',required=True);a=P.parse_args()
spec=importlib.util.spec_from_file_location('common_ops',a.module);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
root=Path(a.fixtures); records=[]
for d in sorted(root.iterdir()):
 if not d.is_dir(): continue
 m=json.load(open(d/'manifest.json'))
 def t(n,dt):
  q=m['tensors'][n];p=d/(n+'.bin');return torch.from_file(str(p),shared=False,size=p.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(*q['shape']).cuda()
 x=t('x',torch.float16);qw=t('qweight',torch.int32);qz=t('qzeros',torch.int32);s=t('scales',torch.float16);g=t('g_idx',torch.int32)
 y=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,s,g);torch.cuda.synchronize()
 K,N=x.shape[1],qw.shape[1]
 q=torch.empty((K,N),device='cuda',dtype=torch.int32)
 for b in range(8): q[b::8]=(qw>>(b*4))&15
 z=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
 for b in range(8): z[:,b::8]=(qz>>(b*4))&15
 # Validate independent packed semantics exactly, including g_idx use.
 assert torch.equal(q[::8],qw&15)
 assert torch.equal(z[:,::8],qz&15)
 ref64=x.double() @ ((q.double()-(z[g.long()]+1).double())*s.double()[g.long()])
 ref=ref64.float(); lim=1e-4+5e-5*ref.abs()
 def bad(v): return torch.nonzero((v.float()-ref).abs()>lim)
 # The registered candidate returns production FP16. It cannot expose the
 # accumulator; use the isolated FP32 prototype for the oracle gate instead.
 # This script's op result is diagnostics only and must not be labeled FP32.
 yh=y.half(); bh=bad(yh)
 records.append({'fixture':m['prefix'],'registered_op_dtype':str(y.dtype),'registered_op_strict_bad_diagnostic_only':int(bh.shape[0]),'registered_op_max_abs':float((yh.float()-ref).abs().max()),'registered_op_finite':bool(torch.isfinite(yh).all()),'registered_op_bad_coords':[[int(i),int(j)] for i,j in bh[:100].tolist()]})
 print(json.dumps(records[-1]),flush=True)
Path(a.out).write_text(''.join(json.dumps(r)+'\n' for r in records))
