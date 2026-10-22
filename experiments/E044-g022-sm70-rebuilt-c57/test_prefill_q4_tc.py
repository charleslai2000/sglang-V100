#!/usr/bin/env python3
"""Bit-exact Q4 packing roundtrip and T012 SM70 kernel qualification entry."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prefill_q4_tc", HERE / "prefill_q4_tc.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def unpack_words(qweight: torch.Tensor) -> np.ndarray:
    raw = qweight.cpu().contiguous().numpy().view(np.uint32)
    k, n = raw.shape[0] * 8, raw.shape[1]
    out = np.empty((k, n), dtype=np.uint8)
    for bit in range(8):
        out[bit::8] = ((raw >> (4 * bit)) & 15).astype(np.uint8)
    return out


def check_roundtrip(qweight, g_idx):
    k, n = qweight.shape[0] * 8, qweight.shape[1]
    packed, perm = mod.prepack(qweight, g_idx, n)
    assert packed.dtype == torch.int32
    assert tuple(packed.shape) == (n // 32, k // 16, 32, 2)
    bytes_ = packed.numpy().view(np.uint8).reshape(n // 32, k // 16, 32, 8)
    original = unpack_words(qweight)
    p = perm.numpy()
    inv = np.empty_like(p); inv[p] = np.arange(k, dtype=np.int32)
    col = np.array([((lane >> 2) & 3) * 8 + (lane & 3) + (4 if lane & 16 else 0) for lane in range(32)])
    ko = np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15])
    recovered = np.empty_like(original)
    for nt in range(n // 32):
        cols = nt * 32 + col
        for kg in range(k // 16):
            ks = p[(kg // 8) * 128 + (kg % 8) * 16 + ko]
            b = bytes_[nt, kg]
            vals = np.empty((32,16),dtype=np.uint8)
            vals[:,0::2] = b & 15; vals[:,1::2] = b >> 4
            recovered[ks[:,None],cols[None,:]] = vals.T
    if not np.array_equal(recovered, original):
        bad = np.argwhere(recovered != original)[0].tolist()
        raise AssertionError(f"Q4 roundtrip mismatch at {bad}")
    return {"K":k,"N":n,"packed_bytes":packed.numel()*packed.element_size(),"q4_codes":int(original.size),"roundtrip":"bitwise exact","nonidentity_g_idx":bool(not np.array_equal(p,np.arange(k,dtype=np.int32)))}


def main():
    import argparse, json
    ap=argparse.ArgumentParser();ap.add_argument("--fixture",required=True);ap.add_argument("--build",required=True);ap.add_argument("--M",type=int,default=0);ap.add_argument("--tile",type=int,default=0);ap.add_argument("--splits",default="");args=ap.parse_args()
    d=Path(args.fixture)
    import json as js
    meta=js.loads((d/'manifest.json').read_text())['tensors']
    def read(name,dtype): return torch.from_numpy(np.fromfile(d/(name+'.bin'),dtype=dtype).reshape(meta[name]['shape']).copy())
    q=read('qweight','<i4');g=read('g_idx','<i4');qz=read('qzeros','<i4');sc=read('scales','<f2');x=read('x','<f2');packed_info=check_roundtrip(q,g);print('BITWISE_ROUNDTRIP',json.dumps(packed_info),flush=True)
    # Independent FP64 reference follows original GPTQ-v1 zero+1, group128,
    # nonidentity g_idx and original scales. Dense FP64 weights exist only in
    # this test reference, never in the implementation or runtime path.
    binary=mod.load(args.build)
    p,perm=mod.prepack(q,g,q.shape[1]);print('PACKED_ABI',p.dtype,tuple(p.shape),perm.dtype,flush=True)
    print('BUILD_MODULE',binary.__file__,flush=True)
    K,N=x.shape[1],q.shape[1];G=sc.shape[0];M=args.M or x.shape[0]
    if M!=x.shape[0]:
        if M not in (512,1024,4096): raise ValueError('M override supported only for requested qualification shapes')
        x=torch.arange(M*K,dtype=torch.int64).remainder(2047).to(torch.float16).reshape(M,K).div_(128).contiguous()
    qcodes=unpack_words(q);zraw=qz.numpy().view(np.uint32);z=np.empty((G,N),dtype=np.uint8)
    for bit in range(8):z[:,bit::8]=((zraw>>(4*bit))&15).astype(np.uint8)
    groups=g.numpy();w=(qcodes.astype(np.float64)-(z[groups].astype(np.float64)+1))*sc.numpy().astype(np.float64)[groups]
    ref=torch.from_numpy((x.numpy().astype(np.float64)@w).astype(np.float32)).cuda()
    dev=lambda t:t.cuda().contiguous()
    qd, pd, qzd, sd, xd = dev(p),dev(perm),dev(qz),dev(sc),dev(x)
    gx=torch.empty_like(xd);out=torch.empty((M,N),device='cuda',dtype=torch.float32)
    bads=[]
    for mtile,split_set in ((8,(8,16,32)),(32,(8,)),(64,(4,))):
     if args.tile and mtile!=args.tile: continue
     for split in split_set:
      if args.splits and split not in tuple(int(v) for v in args.splits.split(',')): continue
      if G%split: continue
      got=binary.run(xd,qd,qzd,sd,pd,gx,out,split,mtile).clone();torch.cuda.synchronize();tol=1e-4+5e-5*ref.abs();err=(got-ref).abs();bad=int((err>tol).sum());bads.append(bad)
      print('EAGER',json.dumps({'M':M,'mtile':mtile,'split':split,'bad':bad,'max_ratio':float((err/tol).max()),'max_abs':float(err.max()),'finite':bool(torch.isfinite(got).all())}),flush=True)
      if bad: raise AssertionError('strict FP64 oracle failed')
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph): gout=binary.run(xd,qd,qzd,sd,pd,gx,out,split,mtile).clone()
      graph.replay();torch.cuda.synchronize();ge=(gout-ref).abs();gb=int((ge>1e-4+5e-5*ref.abs()).sum());print('GRAPH',json.dumps({'M':M,'mtile':mtile,'split':split,'bad':gb,'finite':bool(torch.isfinite(gout).all())}),flush=True)
      if gb or not torch.equal(gout.half(),got.half()): raise AssertionError('Graph oracle/cast mismatch')
    print('ORACLE_FINAL',json.dumps({'cases':len(bads),'strict_bad_total':sum(bads),'binary_sha256':__import__('hashlib').sha256(Path(binary.__file__).read_bytes()).hexdigest()}),flush=True)

if __name__=='__main__':main()
