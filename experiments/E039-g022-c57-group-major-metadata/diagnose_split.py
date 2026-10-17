#!/usr/bin/env python3
"""Split-K partial comparison for E039 L43/down M4x4 strict counterexample."""
import hashlib,importlib.util,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
E=Path(__file__).resolve().parent;LOCK=Path('/home/charles/Workspaces/3rdparty/sglang-V100/experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py');sp=importlib.util.spec_from_file_location('e039_split_lock',LOCK);L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L);UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';L.assert_target_idle(UUID);fd=L.acquire(UUID)
try:
 L.assert_cuda_uuid(torch,UUID);d=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures/43-mlp-down_proj-call0000');m=json.loads((d/'manifest.json').read_text());t=m['tensors']
 def ld(n,dt):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(t[n]['shape']).cuda().contiguous()
 x0=ld('x',torch.float16)[:4].contiguous();x=(x0*4).half().contiguous();qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).view(-1);K,N=x.shape[1],qw.shape[1];G=qz.shape[0]
 code=((qw.to(torch.int64).repeat_interleave(8,0)>>((torch.arange(K,device='cuda',dtype=torch.int64)%8).view(K,1)*4))&15).int();zero=torch.empty((G,N),device='cuda',dtype=torch.int32)
 for j in range(8):zero[:,j::8]=(qz>>(4*j))&15
 w=(code.double()-(zero[g.long()].double()+1))*sc.double()[g.long()];ref=(x.double()@w).float();tol=1e-4+5e-5*ref.abs();perm=torch.argsort(g,stable=True).int();xp=x[:,perm.long()].contiguous()
 mod=load(name='e039_c57_groupmajor',sources=[str(E/'actorder_groupmajor_c57.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E039/build',verbose=False);qmajor,p=mod.stable_groupmajor_pack(qw,g,128)
 # Rebuild temporary partial-only op so each exact 128-K output partial is observed without changing production path.
 import torch.utils.cpp_extension as ce
 source=(E/'actorder_groupmajor_c57.cu').read_text()
 stub='''#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_fp16.h>\n#include <cstdint>\nconstexpr int KTILE=128,NTILE=128,MROWS=4;\n'''
 start=source.index('__global__ void groupmajor_partial');end=source.index('// Exact-c57 reducer association')
 kern=source[start:end]
 wrap='''torch::Tensor partial_only(torch::Tensor x,torch::Tensor qw,torch::Tensor qz,torch::Tensor sc){int M=x.size(0),K=x.size(1),N=qw.size(1),G=K/128;auto out=at::empty({G,M,N},x.options().dtype(at::kFloat));auto stream=at::cuda::getCurrentCUDAStream(x.get_device());dim3 grid(N/128,(M+3)/4,G);groupmajor_partial<<<grid,32,0,stream>>>((half*)x.data_ptr<at::Half>(),(uint32_t*)qw.data_ptr<int32_t>(),(uint32_t*)qz.data_ptr<int32_t>(),(half*)sc.data_ptr<at::Half>(),out.data_ptr<float>(),M,N,K,G);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;} PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("partial_only",&partial_only);}'''
 tmp=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E039/partial_diag.cu');tmp.write_text(stub+kern+wrap);build=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E039/partial-build');build.mkdir(parents=True,exist_ok=True)
 pm=ce.load(name='e039_partial_diag',sources=[str(tmp)],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory=str(build),verbose=False)
 part=pm.partial_only(xp,qmajor,qz,sc);torch.cuda.synchronize();pref=part.sum(0) # diagnostic only; each chunk below is compared against independent original K-range FP64 partial.
 # FP64 original-order per group reference and FP32 candidate partials.
 refs=[]
 for g0 in range(G):
  ks=torch.where(g==g0)[0];rpart=(x.double()[:,ks]@w[ks]).float();refs.append(rpart)
 # Compute each original group partial from the same activation slice in original K order.

 r=torch.stack(refs);diff=(part-r).abs();rat=diff/(1e-4+5e-5*r.abs());worst=torch.argmax(rat);ii=torch.unravel_index(worst,rat.shape);group,m0,n0=map(int,ii)
 print(json.dumps({'fixture':m['prefix'],'binary_source_sha256':hashlib.sha256((E/'actorder_groupmajor_c57.cu').read_bytes()).hexdigest(),'partial_diag_binary':str(Path(pm.__file__).resolve()),'partial_shape':list(part.shape),'max_partial_abs':float(diff.max()),'max_partial_tolerance_ratio':float(rat.max()),'worst_partial':{'group':group,'m':m0,'n':n0,'candidate':float(part[group,m0,n0]),'reference':float(r[group,m0,n0]),'abs_error':float(diff[group,m0,n0]),'allowed':float((1e-4+5e-5*r.abs())[group,m0,n0])},'final_total_e039_strict_bad':int(((pref-ref).abs()>tol).sum()),'partial_all_finite':bool(torch.isfinite(part).all())},indent=2))
finally:L.release(fd)
