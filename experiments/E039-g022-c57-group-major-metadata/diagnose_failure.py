#!/usr/bin/env python3
"""Read-only diagnostic for first E039 strict mismatch; no performance timings."""
import hashlib,importlib.util,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
E=Path(__file__).resolve().parent;LOCK=Path('/home/charles/Workspaces/3rdparty/sglang-V100/experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py');sp=importlib.util.spec_from_file_location('e039_diag_lock',LOCK);L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';L.assert_target_idle(UUID);fd=L.acquire(UUID)
try:
 L.assert_cuda_uuid(torch,UUID);root=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');d=root/'43-mlp-down_proj-call0000';m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def ld(n,dt):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(t[n]['shape']).cuda().contiguous()
 x0=ld('x',torch.float16)[:4].contiguous();x=(x0*4).half().contiguous();qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).view(-1);K,N=x.shape[1],qw.shape[1];G=qz.shape[0]
 codes=((qw.to(torch.int64).repeat_interleave(8,0)>>((torch.arange(K,device='cuda',dtype=torch.int64)%8).view(K,1)*4))&15).int();zeros=torch.empty((G,N),device='cuda',dtype=torch.int32)
 for j in range(8):zeros[:,j::8]=(qz>>(4*j))&15
 w=(codes.double()-(zeros[g.long()].double()+1))*sc.double()[g.long()];ref=(x.double()@w).float()
 c57path='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so';sp=importlib.util.spec_from_file_location('common_ops',c57path);cm=importlib.util.module_from_spec(sp);sp.loader.exec_module(cm);c57=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,sc,g,True)
 mod=load(name='e039_c57_groupmajor',sources=[str(E/'actorder_groupmajor_c57.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E039/build',verbose=False);qmajor,perm=mod.stable_groupmajor_pack(qw,g,128);xp=mod.groupmajor_activation(x,perm);y=mod.c57_groupmajor_gptq(xp,qmajor,qz,sc,128,True)
 # Repeat the eager op in the same process to detect unexpected arithmetic nondeterminism.
 y2=mod.c57_groupmajor_gptq(xp,qmajor,qz,sc,128,True)
 # Same packed weights/permutation/kernel with reference activation cast to FP32 isolates activation rounding.
 xf=xp.float().contiguous()
 yf=mod.c57_groupmajor_gptq(xf.half().contiguous(),qmajor,qz,sc,128,True)
 # Repack extraction parity and original-K mapping, separate from output reduction.
 u=((qmajor.to(torch.int64).repeat_interleave(8,0)>>((torch.arange(K,device='cuda',dtype=torch.int64)%8).view(K,1)*4))&15).int();codes_at_perm=codes[perm.long()];pack_bad=int((u!=codes_at_perm).sum())
 limit=1e-4+5e-5*ref.abs();rows=[]
 for name,out in [('e039',y.float()),('c57_fp32',c57.float())]:
  err=(out-ref).abs();ix=(err>limit).nonzero().cpu().tolist();rows.append({'arm':name,'strict_bad':len(ix),'max_abs':float(err.max()),'max_ratio':float((err/limit).max()),'output_sha256':hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest(),'first_bad':[[i,j,float(out[i,j]),float(ref[i,j]),float(err[i,j]),float(limit[i,j])] for i,j in ix[:20]]})
 delta=(y.float()-c57.float()).abs();dix=(delta>0).nonzero().cpu().tolist()
 print(json.dumps({'fixture':'43-mlp-down_proj-call0000','M':4,'stress4':True,'candidate_binary_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),'c57_binary_sha256':hashlib.sha256(Path(c57path).read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'permutation_equal_stable_argsort':bool(torch.equal(perm,torch.argsort(g,stable=True).int())),'q4_unpack_bad_codes':pack_bad,'activation_permutation_sha256':hashlib.sha256(perm.cpu().numpy().tobytes()).hexdigest(),'activation_vs_source_permuted_bitwise':bool(torch.equal(xp,x[:,perm.long()])),'activation_float16_hash':hashlib.sha256(xp.cpu().numpy().tobytes()).hexdigest(),'input_max_abs':float(x.abs().max()),'oracle_coordinate_272':[[i,272,float(ref[i,272]),float(limit[i,272])] for i in range(ref.shape[0])],'e039_repeat_equal':bool(torch.equal(y,y2)),'e039_vs_c57_nonidentical':len(dix),'e039_c57_max_abs_delta':float(delta.max()),'e039_input_max_abs':float(x.abs().max()),'activation_float32_recast_diff_max':float((y-yf).abs().max()),'e039_vs_c57_first_diff':[[i,j,float(y[i,j]),float(c57[i,j]),float(delta[i,j])] for i,j in dix[:20]],'oracle':rows},indent=2))
finally:L.release(fd)
