#!/usr/bin/env python3
"""Exercise per-layer T011 binding and CUDA Graph output on one frozen fixture."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parent/'E035-g022-actorder-q4-wmma')]
from qpn2_multiwarp import load_module
from qpn2_layer_binding import attach,apply
from gpu_uuid_lock import assert_cuda_uuid
class Layer(torch.nn.Module):pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',default='/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures/43-mlp-down_proj-call0000');p.add_argument('--build',default='/mnt/data/sglang-V100/t011-build');a=p.parse_args()
 if torch.cuda.device_count()!=1:raise RuntimeError('exact one CUDA device required')
 assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');mod,b=load_module(a.build);sha=hashlib.sha256(b.read_bytes()).hexdigest();d=Path(a.fixture);m=json.loads((d/'manifest.json').read_text());T=m['tensors'];K=T['g_idx']['shape'][0];N=T['qweight']['shape'][1]
 def raw(n,dt):return np.fromfile(d/(n+'.bin'),dt).reshape(T[n]['shape'])
 l=Layer().cuda();l.qweight=torch.nn.Parameter(torch.from_numpy(raw('qweight','<i4').copy()).cuda(),requires_grad=False);l.qzeros=torch.nn.Parameter(torch.from_numpy(raw('qzeros','<i4').copy()).cuda(),requires_grad=False);l.scales=torch.nn.Parameter(torch.from_numpy(raw('scales','<f2').copy()).cuda(),requires_grad=False);l.g_idx=torch.nn.Parameter(torch.from_numpy(raw('g_idx','<i4').copy()).cuda(),requires_grad=False)
 bound=attach(l,4,128,str(b),sha);print('binding',json.dumps(bound),'packed_shape',tuple(l._t011_qpn2_packed.shape),'perm',tuple(l._t011_qpn2_perm.shape),'scales',tuple(l.scales.shape),flush=True)
 xs=raw('x','<f2')[:4];x=torch.from_numpy(xs.copy()).cuda();# strict FP64 reference from original GPTQ semantics
 qw=l.qweight.detach().cpu().numpy().view(np.uint32);q=np.empty((K,N),np.uint8)
 for bit in range(8):q[bit::8]=((qw>>(4*bit))&15).astype(np.uint8)
 qz=l.qzeros.detach().cpu().numpy().view(np.uint32);z=np.empty((K//128,N),np.uint8)
 for bit in range(8):z[:,bit::8]=((qz>>(4*bit))&15).astype(np.uint8)
 g=l.g_idx.detach().cpu().numpy();sc=l.scales.detach().cpu().numpy();w=(q.astype(np.float64)-(z[g].astype(np.float64)+1))*sc.astype(np.float64)[g];ref=torch.from_numpy((xs.astype(np.float64)@w).astype(np.float32)).cuda();gx=torch.empty_like(x);scratch=torch.empty((4,N),device='cuda',dtype=torch.float32);raw=mod.run(x,l._t011_qpn2_packed,l.scales,l._t011_qpn2_perm,gx,scratch,16).clone();torch.cuda.synchronize();tol=1e-4+5e-5*ref.abs();err=(raw-ref).abs();bad=int((err>tol).sum().item());y=apply(l,x,None);again=apply(l,x,None).clone();torch.cuda.synchronize();
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):gy=apply(l,x,None)
 graph.replay();torch.cuda.synchronize();go=gy.clone();gb=int(((go.float()-raw.half().float()).abs()>0).sum().item());print(json.dumps({'fixture':m['prefix'],'M':4,'K':K,'N':N,'binding':bound,'binary_sha256':sha,'strict_bad':bad,'graph_vs_eager_cast_mismatch':gb,'repeat_equal':torch.equal(y,again),'graph_cast_equal':torch.equal(go,y),'max_ratio':float((err/tol).max().item()),'output_sha256':hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),'graph_sha256':hashlib.sha256(go.cpu().numpy().tobytes()).hexdigest()}))
if __name__=='__main__':main()
