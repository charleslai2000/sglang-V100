#!/usr/bin/env python3
"""Exercise saved CPU-prepacked Q/permutation/scales directly on the V100 prototype."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--converted',required=True);p.add_argument('--module',required=True);p.add_argument('--out',required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('e035_grouped_q4_sm70',a.module);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
root=Path(a.fixtures);conv=Path(a.converted);results=[]
for d in sorted(root.iterdir()):
 if not d.is_dir() or not (d/'manifest.json').exists():continue
 m=json.loads((d/'manifest.json').read_text());c=json.loads((conv/d.name/'manifest.json').read_text())
 if c['M4_fixture_sha256']!=hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest():raise RuntimeError('converter/source manifest identity mismatch')
 t=m['tensors']
 def ld(n,dtype):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(t[n]['shape']).cuda()
 x4=ld('x',torch.float16);qw0=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);s0=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1)
 perm=torch.from_numpy(np.load(conv/d.name/'perm.npy',allow_pickle=False).copy()).cuda();packed=torch.from_numpy(np.load(conv/d.name/'packed_q.npy',allow_pickle=False).copy()).cuda();sc=torch.from_numpy(np.load(conv/d.name/'scales.npy',allow_pickle=False).copy()).cuda()
 # Verify loaded saved format maps back to all original Q codes and scale rows.
 q=torch.empty((len(g),qw0.shape[1]),device='cuda',dtype=torch.int32)
 for b in range(8):q[b::8]=(qw0>>(4*b))&15
 qg=torch.empty_like(q);qg.copy_(q.index_select(0,perm));qback=torch.empty_like(qg)
 for b in range(8):qback[b::8]=(packed>>(4*b))&15
 codebad=int((qback!=qg).sum().item());
 if codebad:raise RuntimeError(f'{d.name}: saved pack mismatch {codebad}')
 if not torch.equal(sc,s0):raise RuntimeError(f'{d.name}: scales mismatch')
 z=torch.empty((qz.shape[0],qz.shape[1]*8),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>>(4*b))&15
 if not torch.all(z==7):raise RuntimeError('qualified effective zero=8 assertion changed')
 w64=(q.double()-(z[g.long()]+1).double())*s0.double()[g.long()]
 for M in (1,2,4):
  x=x4[:M].contiguous();K,N=x.shape[1],q.shape[1];G=sc.shape[0];scratch=torch.empty((M,N),device='cuda',dtype=torch.float32);out=torch.empty_like(scratch);gx=torch.empty_like(x)
  y=mod.grouped_q4_sm70(x,packed,sc,perm,scratch,gx,out,True);torch.cuda.synchronize();ref=(x.double()@w64).float();lim=1e-4+5e-5*ref.abs();bad=int(((y-ref).abs()>lim).sum().item())
  row={'fixture':m['prefix'],'M':M,'K':K,'N':N,'converter_manifest_sha256':hashlib.sha256((conv/d.name/'manifest.json').read_bytes()).hexdigest(),'packed_q_sha256':c['packed_q_sha256'],'perm_sha256':c['permutation_sha256'],'code_bad':codebad,'strict_bad':bad,'max_abs_error':float((y-ref).abs().max().item()),'output_fp16_finite':bool(torch.isfinite(y.half()).all().item())}
  results.append(row);print(json.dumps(row),flush=True)
Path(a.out).write_text(''.join(json.dumps(r)+'\n' for r in results))
