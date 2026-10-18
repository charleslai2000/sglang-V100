#!/usr/bin/env python3
"""Compare CPU load-time packing with existing GPU fixture packer and measure conversion."""
import json,hashlib,time
from pathlib import Path
import numpy as np,torch
from qpn2_multiwarp import prepack_cpu
HERE=Path(__file__).resolve().parent;FIX=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures')
# CUDA modules use python utility import path from the E035 experiment.
import sys;sys.path.insert(0,str(HERE.parent/'E035-g022-actorder-q4-wmma'))
from gpu_uuid_lock import assert_cuda_uuid

def main():
 if torch.cuda.device_count()!=1:raise RuntimeError('require exactly one visible GPU')
 assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189')
 rows=[]
 for d in sorted(FIX.iterdir()):
  if not d.is_dir() or not (d/'manifest.json').exists():continue
  mf=json.loads((d/'manifest.json').read_text());T=mf['tensors'];K=T['g_idx']['shape'][0];N=T['qweight']['shape'][1]
  qw=np.fromfile(d/'qweight.bin','<i4').reshape(T['qweight']['shape']);g=np.fromfile(d/'g_idx.bin','<i4')
  qt=torch.from_numpy(qw.copy());gt=torch.from_numpy(g.copy());torch.cuda.synchronize();t=time.perf_counter();packed,perm=prepack_cpu(qt,gt);cpu_s=time.perf_counter()-t
  torch.cuda.synchronize();t=time.perf_counter();pd=packed.cuda();permd=perm.cuda();torch.cuda.synchronize();h2d_s=time.perf_counter()-t
  qraw=qw.view(np.uint32);codes=np.empty((K,N),np.uint8)
  for b in range(8):codes[b::8]=((qraw>>(4*b))&15).astype(np.uint8)
  srcper=np.argsort(g,kind='stable').astype(np.int32)
  if not np.array_equal(perm.numpy(),srcper):raise RuntimeError('permutation difference')
  packed_u=packed.numpy().view(np.uint32); col=np.array([((l>>2)&3)*8+(l&3)+(4 if l&16 else 0) for l in range(32)]);ko=np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);bad=0
  for tile in range(N//32):
   for kg in range(K//16):
    sk=srcper[(kg//8)*128+(kg%8)*16+ko];v=codes[sk[:,None],(tile*32+col)[None,:]].T;expected=(v[:,0::2]|(v[:,1::2]<<4)).copy().view(np.uint32);bad+=int(np.count_nonzero(packed_u[tile,kg]!=expected))
  row={'fixture':mf['prefix'],'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'K':K,'N':N,'packed_bytes':packed.numel(),'cpu_pack_seconds':cpu_s,'h2d_seconds':h2d_s,'mismatch_bytes':bad,'packed_sha256':hashlib.sha256(packed.numpy().tobytes()).hexdigest(),'perm_sha256':hashlib.sha256(perm.numpy().tobytes()).hexdigest()};rows.append(row);print(json.dumps(row),flush=True)
 Path(HERE/'results/t011-prepack-conversion.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
