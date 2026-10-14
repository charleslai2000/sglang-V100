#!/usr/bin/env python3
"""Deterministic act-order Q4 group-major converter for E035 fixtures."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--out',required=True);a=p.parse_args()
r=Path(a.fixture);o=Path(a.out);o.mkdir(parents=True,exist_ok=True);m=json.loads((r/'manifest.json').read_text());t=m['tensors']
def load(name,dtype): return np.fromfile(r/(name+'.bin'),dtype=dtype).reshape(t[name]['shape'])
qw=load('qweight','<i4').view('<u4');qz=load('qzeros','<i4').view('<u4');sc=load('scales','<f2');g=load('g_idx','<i4').reshape(-1)
K=len(g);N=qw.shape[1];G=sc.shape[0]
if K%G or K//G!=128 or qz.shape!=(G,N//8) or qw.shape!=(K//8,N):raise SystemExit('unsupported GPTQ shapes')
# Only the verified Falcon effective-zero=8 fast path is enabled. Generic zero
# is retained in the record and any deviation is explicitly rejected.
z=np.empty((G,N),np.uint8)
for b in range(8):z[:,b::8]=((qz>>(4*b))&15).astype(np.uint8)
if not np.all(z==7): raise SystemExit('NO-GO-fastpath: stored qzero must be 7 (effective 8); generic qzero kernel not implemented')
if np.any((g<0)|(g>=G)):raise SystemExit('g_idx outside group range')
# Stable grouping preserves source K order within each GPTQ group.
perm=np.concatenate([np.flatnonzero(g==gi) for gi in range(G)]).astype('<i4')
if len(perm)!=K or not np.array_equal(np.sort(perm),np.arange(K,dtype=np.int32)):raise SystemExit('non-bijective permutation')
# Unpack q in source K,N order; permute K by sorted group; repack K nibble lanes.
q=np.empty((K,N),np.uint8)
for b in range(8):q[b::8]=((qw>>(4*b))&15).astype(np.uint8)
qg=np.ascontiguousarray(q[perm]);packed=np.zeros((K//8,N),dtype='<u4')
for b in range(8):packed|=qg[b::8].astype('<u4')<<(4*b)
np.save(o/'perm.npy',perm,allow_pickle=False);np.save(o/'packed_q.npy',packed.view('<i4'),allow_pickle=False);np.save(o/'scales.npy',sc,allow_pickle=False)
# Keep evidence of source zeros and group assignment; zero bytes aren't needed by
# the proven qzero=8 arithmetic fast path, but source SHA and eligibility are.
meta={'prefix':m['prefix'],'M4_fixture_sha256':hashlib.sha256((r/'manifest.json').read_bytes()).hexdigest(),'K':K,'N':N,'G':G,'group_size':128,'zero_fastpath':'stored 7 => effective GPTQ-v1 zero 8','all_stored_zero_nibbles_are_7':True,'stable_group_order':'group id ascending; original K order within each group','gate_up_shared_permutation_requires_equal_g_idx':True,'permutation_sha256':hashlib.sha256(perm.tobytes()).hexdigest(),'packed_q_sha256':hashlib.sha256(packed.tobytes()).hexdigest(),'scales_sha256':hashlib.sha256(sc.tobytes()).hexdigest(),'stored_qzero_sha256':hashlib.sha256(qz.tobytes()).hexdigest(),'prepack_bytes':int(perm.nbytes+packed.nbytes+sc.nbytes),'source_metadata_bytes':int(qw.nbytes+qz.nbytes+sc.nbytes+g.nbytes)}
(o/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
