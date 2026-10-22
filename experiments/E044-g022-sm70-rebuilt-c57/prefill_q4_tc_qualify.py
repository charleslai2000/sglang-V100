#!/usr/bin/env python3
"""T012 full-L43-Q4 M sweep against independent original-GPTQ FP64."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
import prefill_q4_tc as op

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--build',required=True);p.add_argument('--M',type=int,required=True);p.add_argument('--tile',type=int,default=64);p.add_argument('--split',type=int,default=4);p.add_argument('--input-npy');p.add_argument('--graph',action='store_true');a=p.parse_args()
 d=Path(a.fixture);meta=json.loads((d/'manifest.json').read_text())['tensors']
 def load(n,dt):return torch.from_numpy(np.fromfile(d/(n+'.bin'),dtype=dt).reshape(meta[n]['shape']).copy())
 qw=load('qweight','<i4');qz=load('qzeros','<i4');s=load('scales','<f2');g=load('g_idx','<i4');
 if a.input_npy:x=torch.from_numpy(np.load(a.input_npy).astype(np.float16))
 elif a.M==4:x=load('x','<f2')
 else:x=torch.arange(a.M*12288,dtype=torch.int64).remainder(2047).to(torch.float16).reshape(a.M,12288).div_(128).contiguous()
 assert tuple(x.shape)==(a.M,12288)
 q,p=op.prepack(qw,g,3072);K,N=12288,3072;G=96
 qraw=qw.numpy().view(np.uint32);codes=np.empty((K,N),np.uint8)
 for b in range(8):codes[b::8]=((qraw>>(4*b))&15).astype(np.uint8)
 zr=qz.numpy().view(np.uint32);z=np.empty((G,N),np.uint8)
 for b in range(8):z[:,b::8]=((zr>>(4*b))&15).astype(np.uint8)
 refw=(codes.astype(np.float64)-(z[g.numpy()].astype(np.float64)+1))*s.numpy().astype(np.float64)[g.numpy()]
 ref=torch.from_numpy((x.numpy().astype(np.float64)@refw).astype(np.float32))
 mod=op.load(a.build);binary=Path(mod.__file__);dev=lambda t:t.cuda().contiguous();x,q,p,qz,s,ref=map(dev,(x,q,p,qz,s,ref));gx=torch.empty_like(x);out=torch.empty((a.M,N),device='cuda',dtype=torch.float32)
 y=mod.run(x,q,qz,s,p,gx,out,a.split,a.tile).clone();torch.cuda.synchronize();lim=1e-4+5e-5*ref.abs();err=(y-ref).abs();bad=int((err>lim).sum());rec={'M':a.M,'K':K,'N':N,'G':G,'tile':a.tile,'split':a.split,'strict_bad':bad,'max_ratio':float((err/lim).max()),'max_abs':float(err.max()),'finite':bool(torch.isfinite(y).all() and torch.isfinite(y.half()).all()),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'g_idx_nonidentity':not torch.equal(torch.sort(g)[0],g)}
 if a.graph:
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):yg=mod.run(x,q,qz,s,p,gx,out,a.split,a.tile).clone()
  graph.replay();torch.cuda.synchronize();e=(yg-ref).abs();rec.update(graph_bad=int((e>lim).sum()),graph_cast_equal=torch.equal(yg.half(),y.half()),graph_finite=bool(torch.isfinite(yg).all()))
 print(json.dumps(rec),flush=True)
 if bad or (a.graph and (rec['graph_bad'] or not rec['graph_cast_equal'])):raise SystemExit(2)
if __name__=='__main__':main()
