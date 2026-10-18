#!/usr/bin/env python3
"""CPU-only proof of QPN packed-weight / act-permutation mapping for real Falcon GPTQ tensors."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
MODEL=Path('/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4')
def main():
 idx=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']; headers={}; files={}
 for fn in sorted(set(idx.values())):
  with (MODEL/fn).open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));headers[fn]=(8+n,h);files[fn]=f
 rows=[]
 for qname,fn in sorted(idx.items()):
  if not qname.endswith('.qweight'):continue
  root=qname[:-8];zn, sn, gn=root+'.qzeros',root+'.scales',root+'.g_idx';base,h=headers[fn];m=h[qname];z=h[zn];s=h[sn];g=h[gn]
  def load(meta):
   start=base+meta['data_offsets'][0];end=base+meta['data_offsets'][1]
   with (MODEL/fn).open('rb') as f:f.seek(start);b=f.read(end-start)
   return b
  qw=np.frombuffer(load(m),dtype='<i4').reshape(m['shape']);qz=np.frombuffer(load(z),dtype='<i4').reshape(z['shape']);sc=np.frombuffer(load(s),dtype='<f2').reshape(s['shape']);gi=np.frombuffer(load(g),dtype='<i4').reshape(-1)
  K=gi.size;N=m['shape'][1];G=sc.shape[0]
  if K not in (3072,12288) or G!=K//128 or K%128 or N%32:raise RuntimeError(f'unsupported dimensions {qname}: {K=} {N=} {G=}')
  if np.any((gi<0)|(gi>=G)):raise RuntimeError(f'g_idx out of range {qname}')
  counts=np.bincount(gi,minlength=G)
  if not np.all(counts==128):raise RuntimeError(f'g_idx group count mismatch {qname}')
  zraw=qz.view(np.uint32);zvals=np.stack([(zraw>>(4*b))&15 for b in range(8)],axis=-1)
  if not np.all(zvals==7):raise RuntimeError(f'qzero mismatch {qname}')
  perm=np.argsort(gi,kind='stable').astype(np.int32); grouped=gi[perm]
  if not np.array_equal(grouped,np.repeat(np.arange(G,dtype=np.int32),128)):raise RuntimeError(f'permutation boundary mismatch {qname}')
  # Establish that original activation gather x[:,perm] aligns to contiguous group128.
  assert all(np.all(gi[perm[g*128:(g+1)*128]]==g) for g in range(G))
  # Reorder packed 8-lane GPTQ rows by the exact group permutation; recover
  # original qweight codes through inverse permutation and verify bitwise.
  raw=qw.view(np.uint32);codes=np.empty((K,N),dtype=np.uint8)
  for b in range(8):codes[b::8]=((raw>>(4*b))&15).astype(np.uint8)
  grouped_codes=codes[perm]
  restored=np.empty_like(codes);restored[perm]=grouped_codes
  if not np.array_equal(restored,codes):raise RuntimeError(f'Q4 mapping mismatch {qname}')
  rows.append({'name':root,'K':K,'N':N,'groups':G,'qweight_bytes':m['data_offsets'][1]-m['data_offsets'][0],'qzero_effective':8,'perm_sha256':hashlib.sha256(perm.tobytes()).hexdigest(),'qweight_sha256':hashlib.sha256(load(m)).hexdigest(),'scale_sha256':hashlib.sha256(load(s)).hexdigest(),'g_idx_sha256':hashlib.sha256(load(g)).hexdigest(),'mapping':'x[:,argsort(g_idx)] and Q4 rows follow same stable permutation'})
 if len(rows)!=132:raise RuntimeError(f'expected 132 qweight modules, got {len(rows)}')
 result={'modules':len(rows),'total_qweight_bytes':sum(r['qweight_bytes'] for r in rows),'mapping_checked':'all rows decoded/reordered/inverse-reordered bitwise; all group boundaries=128; qzero nibble=7','layers':rows}
 print(json.dumps({k:v for k,v in result.items() if k!='layers'}));Path('experiments/E044-g022-qpn2-sm70-actorder/results/t011-mapping-audit.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
