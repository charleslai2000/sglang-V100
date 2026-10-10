#!/usr/bin/env python3
"""Real-fixture FP16 conversion/high-amplitude checks for the G017 split-K op."""
import hashlib,json,os
from pathlib import Path
import torch
from importlib.machinery import ExtensionFileLoader
module=Path(os.environ['SGLANG_KERNEL_G017_CANDIDATE_PATH']).resolve(); ext=ExtensionFileLoader('common_ops',str(module)).load_module(); root=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures'); rows=[]
for d in sorted(root.iterdir()):
 if not d.is_dir() or not (d/'manifest.json').exists(): continue
 m=json.loads((d/'manifest.json').read_text())
 def t(n,dt):
  f=d/(n+'.bin');q=m['tensors'][n];return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(*q['shape']).cuda()
 x=t('x',torch.float16);w=t('qweight',torch.int32);z=t('qzeros',torch.int32);s=t('scales',torch.float16);g=t('g_idx',torch.int32);pts=[]
 for k in (1,2.5,4,8,16):
  xx=(x*k).to(torch.float16).contiguous(); acc=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(xx,w,z,s,g,True); out=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(xx,w,z,s,g,False);torch.cuda.synchronize()
  pts.append({'factor':k,'input_finite':bool(torch.isfinite(xx).all()),'fp32_finite':bool(torch.isfinite(acc).all()),'fp16_finite':bool(torch.isfinite(out).all()),'equals_cast':bool(torch.equal(out,acc.half()))})
 row={'fixture':m['prefix'],'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'points':pts}; rows.append(row);print(json.dumps(row),flush=True)
print(json.dumps({'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'fixture_count':len(rows),'all_real_finite':all(p['fp16_finite'] for r in rows for p in r['points'] if p['factor']<=8),'all_cast_equal':all(p['equals_cast'] for r in rows for p in r['points'])}))
