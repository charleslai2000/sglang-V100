#!/usr/bin/env python3
"""Qualify QPN2 prototype against six E035 conversion assets, M1/M2/M4."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'E035-g022-actorder-q4-wmma'))
from gpu_uuid_lock import assert_cuda_uuid
p=argparse.ArgumentParser();p.add_argument('--fixtures',default=str(HERE/'fixtures/converted-six'));p.add_argument('--source-fixtures',default='/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');p.add_argument('--module',default=str(HERE/'build-qpn/e044_qpn2_gptq.so'));p.add_argument('--out',default=str(HERE/'results/six-fixture-correctness.jsonl'));a=p.parse_args()
root=Path(a.fixtures);source=Path(a.source_fixtures);out=Path(a.out);modpath=Path(a.module)
if torch.cuda.device_count()!=1:raise SystemExit('refusing multi-GPU visibility; use only target UUID')
assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189')
spec=__import__('importlib.util').util.spec_from_file_location('e044_qpn2_gptq',modpath);mod=__import__('importlib.util').util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
for copied in sorted(root.iterdir()):
 if not copied.is_dir() or copied.name.startswith('.'):continue
 d=source/copied.name
 m=json.loads((copied/'manifest.json').read_text());K,N,G=m['K'],m['N'],m['G'];M4=4
 def raw(path,dt,shape):return np.fromfile(path,dtype=dt).reshape(shape)
 perm=np.load(copied/'perm.npy',allow_pickle=False);qpacked=np.load(copied/'packed_q.npy',allow_pickle=False).view('<u4').reshape(K//8,N);sc=np.load(copied/'scales.npy',allow_pickle=False).reshape(G,N)
 # Reconstruct grouped Q4 codes from the validated group-major prepack.
 qg=np.empty((K,N),np.uint8)
 for b in range(8):qg[b::8]=((qpacked>>(4*b))&15).astype(np.uint8)
 q=np.empty_like(qg);q[perm]=qg
 # The E044 local fixture package has L43/down raw tensors; other original
 # raw fixture folders are in the existing remote-evidence archive. Read-only.
 froot=(HERE/'fixtures/l43-down/packed-word-fixture-package') if m['prefix']=='43.mlp.down_proj' else Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures')/copied.name
 fm=json.loads((froot/'manifest.json').read_text());T=fm['tensors'];x4=raw(froot/'x.bin','<f2',(M4,K));qzero=raw(froot/'qzeros.bin','<u4',(G,N//8));gidx=raw(froot/'g_idx.bin','<i4',(K,))
 z=np.empty((G,N),np.uint8)
 for b in range(8):z[:,b::8]=((qzero>>(4*b))&15).astype(np.uint8)
 if not np.all(z==7):raise RuntimeError('unsupported qzero')
 if not np.array_equal(np.sort(perm),np.arange(K)):raise RuntimeError('non-bijective permutation')
 if not np.array_equal(gidx[perm],np.repeat(np.arange(G),128)):raise RuntimeError('converter permutation does not group original g_idx')
 col=np.array([((l>>2)&3)*8+(l&3)+(4 if l&16 else 0) for l in range(32)]);ko=np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);packed=np.empty((N//32,K//16,32,8),np.uint8)
 for tile in range(N//32):
  for kg in range(K//16):
   sk=perm[(kg//8)*128+(kg%8)*16+ko];v=q[sk[:,None],(tile*32+col)[None,:]].T;packed[tile,kg]=v[:,0::2]|(v[:,1::2]<<4)
 qt=torch.from_numpy(packed.view(np.int32).copy()).cuda();st=torch.from_numpy(sc.copy()).cuda();pt=torch.from_numpy(perm.copy()).cuda()
 w=(q.astype(np.float64)-(z[gidx].astype(np.float64)+1))*sc.astype(np.float64)[gidx]
 for M in (1,2,4):
  tests=[('ordinary',x4[:M])]
  if m['prefix']=='43.mlp.down_proj' and M==4:tests.append(('4x_saved_stress',raw(Path('experiments/E044-g022-qpn2-sm70-actorder/fixtures/l43-down/packed-word-fixture-package/c57_stress_x_fp16.bin'),'<f2',(4,K))))
  for label,xnp in tests:
   xt=torch.from_numpy(np.ascontiguousarray(xnp)).cuda();gx=torch.empty_like(xt);o=torch.empty((M,N),device='cuda',dtype=torch.float32);ref=torch.from_numpy((xnp.astype(np.float64)@w).astype(np.float32)).cuda();y=mod.run(xt,qt,st,pt,gx,o);torch.cuda.synchronize();lim=1e-4+5e-5*ref.abs();err=(y-ref).abs();bad=err>lim;repeat=mod.run(xt,qt,st,pt,gx,o).clone();torch.cuda.synchronize();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):gy=mod.run(xt,qt,st,pt,gx,o)
    graph.replay();torch.cuda.synchronize()
   torch.cuda.current_stream().wait_stream(stream);go=o.clone();gb=int(((go-ref).abs()>lim).sum().item());row={'fixture':m['prefix'],'source_fixture_manifest_sha256':hashlib.sha256((froot/'manifest.json').read_bytes()).hexdigest(),'converter_manifest_sha256':hashlib.sha256((copied/'manifest.json').read_bytes()).hexdigest(),'M':M,'input':label,'binary_sha256':hashlib.sha256(modpath.read_bytes()).hexdigest(),'strict_bad':int(bad.sum().item()),'graph_strict_bad':gb,'max_abs_error':float(err.max().item()),'max_ratio':float((err/lim).max().item()),'repeat_equal':bool(torch.equal(y,repeat)),'graph_cast_equal':bool(torch.equal(gy.half(),y.half())),'finite':bool(torch.isfinite(y).all().item()),'output_sha256':hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest()}
   if bad.any():row['bad_points']=[{'m':i,'n':j,'actual':float(y[i,j]),'reference':float(ref[i,j]),'error':float(err[i,j]),'allowed':float(lim[i,j])} for i,j in bad.nonzero()[:20].tolist()]
   rows.append(row);print(json.dumps(row),flush=True)
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
if any(r['strict_bad'] or r['graph_strict_bad'] or not r['repeat_equal'] or not r['graph_cast_equal'] or not r['finite'] for r in rows):raise SystemExit(1)
