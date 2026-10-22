#!/usr/bin/env python3
"""Experimental large-M SM70 Tensor Core GPTQ Prefill kernel and packer.

No integration/default dispatch: T012 implementation under qualification.
Weights stay Q4-packed; the kernel centers each nibble against the original
GPTQ-v1 zero+1 in registers before Volta HMMA, scales each group in FP32, and
owns an 8x32 output tile per CTA with bounded shared FP32 split-group sums.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.cpp_extension import load

CUDA_SRC = r'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>
#define MMA(C,A0,A1,B0,B1) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n" : "+f"(C[0]),"+f"(C[1]),"+f"(C[2]),"+f"(C[3]),"+f"(C[4]),"+f"(C[5]),"+f"(C[6]),"+f"(C[7]) : "r"(A0),"r"(A1),"r"(B0),"r"(B1))
__device__ __forceinline__ unsigned h2(half a,half b){return (unsigned)__half_as_ushort(a)|((unsigned)__half_as_ushort(b)<<16);}
__device__ __forceinline__ int nib(const int* p,size_t index){unsigned v=(unsigned)p[index>>3];return (v>>((index&7)*4))&15;}
__global__ void gather(const half*x,half*gx,const int*p,int M,int K){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<(size_t)M*K)gx[i]=x[(i/K)*(size_t)K+p[i%K]];}
template<int SPLITK> __global__ void kern(const uint2*q,const int*qzeros,const half*x,const half*s,float*out,int M,int N,int K,int G){
 __shared__ float partial[SPLITK][8][32];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0),nt=blockIdx.x%(N/32),mt=blockIdx.x/(N/32),m0=mt*8;
 int groups_per_warp=G/SPLITK,g0=warp*groups_per_warp;const uint2*base=q+(size_t)nt*(K/16)*32+lane;
 float acc[8]={0};
 for(int g=g0;g<g0+groups_per_warp;++g){float c[8]={0};
  int col0=nt*32+qp*8;
  int z0=nib(qzeros,(size_t)g*N+col0+0),z1=nib(qzeros,(size_t)g*N+col0+1),z2=nib(qzeros,(size_t)g*N+col0+2),z3=nib(qzeros,(size_t)g*N+col0+3),z4=nib(qzeros,(size_t)g*N+col0+4),z5=nib(qzeros,(size_t)g*N+col0+5),z6=nib(qzeros,(size_t)g*N+col0+6),z7=nib(qzeros,(size_t)g*N+col0+7);
  int zs[8]={z0+1,z1+1,z2+1,z3+1,z4+1,z5+1,z6+1,z7+1};
  for(int kg=0;kg<8;++kg){uint2 z=*(base+(size_t)(g*8+kg)*32);uint32_t w[2]={z.x,z.y};unsigned B[8];for(int wi=0;wi<2;++wi)for(int bi=0;bi<4;++bi){int j=wi*4+bi;int p0=(w[wi]>>(bi*4))&15,p1=(w[wi]>>((bi+4)*4))&15;B[j]=h2(__int2half_rn(p0-zs[j]),__int2half_rn(p1-zs[j]));}
   int kk=kg*16;const half*xr=x+(size_t)(m0+r)*K+(size_t)g*128+kk;uint4 u=make_uint4(0,0,0,0),v=make_uint4(0,0,0,0);if(m0+r<M){u=*reinterpret_cast<const uint4*>(xr);v=*reinterpret_cast<const uint4*>(xr+8);}unsigned A[4],V[4];*reinterpret_cast<uint4*>(A)=u;*reinterpret_cast<uint4*>(V)=v;MMA(c,A[0],A[1],B[0],B[1]);MMA(c,A[2],A[3],B[2],B[3]);MMA(c,V[0],V[1],B[4],B[5]);MMA(c,V[2],V[3],B[6],B[7]);}
  #pragma unroll
  for(int i=0;i<8;++i){int row=(i&2)|((lane&16)?4:0)|(lane&1),col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2),m=row,n=nt*32+qp*8+col;if(m<8&&m0+m<M)acc[i]+=c[i]*__half2float(s[(size_t)g*N+n]);}
 }
 #pragma unroll
 for(int i=0;i<8;++i){int row=(i&2)|((lane&16)?4:0)|(lane&1),col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2);partial[warp][row][qp*8+col]=acc[i];}
 __syncthreads();
 for(int e=threadIdx.x;e<8*32;e+=blockDim.x){int row=e>>5,col=e&31;float v=0.f;
  #pragma unroll
  for(int w=0;w<SPLITK;++w)v=v+partial[w][row][col];
  if(m0+row<M)out[(size_t)(m0+row)*N+(size_t)nt*32+col]=v;
 }
}
torch::Tensor run(torch::Tensor x,torch::Tensor q,torch::Tensor qz,torch::Tensor s,torch::Tensor p,torch::Tensor gx,torch::Tensor o,int split){TORCH_CHECK(x.is_cuda()&&q.is_cuda()&&qz.is_cuda()&&s.is_cuda()&&p.is_cuda()&&gx.is_cuda()&&o.is_cuda(),"T012 CUDA tensors required");TORCH_CHECK(x.scalar_type()==at::kHalf&&qz.scalar_type()==at::kInt&&s.scalar_type()==at::kHalf&&gx.scalar_type()==at::kHalf&&o.scalar_type()==at::kFloat,"T012 dtype mismatch");int M=x.size(0),K=x.size(1),N=s.size(1),G=s.size(0);TORCH_CHECK(M>=1&&K%128==0&&N%32==0&&G==K/128&&G%split==0,"T012 unsupported shape/split");TORCH_CHECK(q.scalar_type()==at::kInt&&p.scalar_type()==at::kInt,"T012 packed Q4 and permutation must be int32");TORCH_CHECK(q.dim()==4&&q.size(0)==N/32&&q.size(1)==K/16&&q.size(2)==32&&q.size(3)==2&&qz.dim()==2&&qz.size(0)==G&&qz.size(1)==N/8&&p.numel()==K&&gx.sizes()==x.sizes()&&o.sizes()==at::IntArrayRef({M,N}),"T012 tensor shape mismatch");auto st=at::cuda::getCurrentCUDAStream();gather<<<(M*K+255)/256,256,0,st>>>(reinterpret_cast<half*>(x.data_ptr<at::Half>()),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),p.data_ptr<int>(),M,K);
 #define LAUNCH(S) kern<S><<<(N/32)*((M+7)/8),32*S,0,st>>>(reinterpret_cast<uint2*>(q.data_ptr<int>()),qz.data_ptr<int>(),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),reinterpret_cast<half*>(s.data_ptr<at::Half>()),o.data_ptr<float>(),M,N,K,G)
 if(split==8)LAUNCH(8);else if(split==16)LAUNCH(16);else if(split==32)LAUNCH(32);else TORCH_CHECK(false,"T012 split must be 8,16,32");
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
'''


def prepack(qweight: torch.Tensor, g_idx: torch.Tensor, n: int):
    """Pack original nibble codes in T011 donor quadpair/K order; retain Q4."""
    raw = qweight.detach().contiguous().cpu().numpy().view(np.uint32)
    k8, ncols = raw.shape
    k = k8 * 8
    groups = k // 128
    gi = g_idx.detach().contiguous().cpu().numpy().astype(np.int32, copy=False)
    if ncols != n or k % 128 or n % 32 or gi.shape != (k,):
        raise ValueError("unsupported GPTQ dimensions")
    if not np.array_equal(np.sort(gi), np.repeat(np.arange(groups, dtype=np.int32), 128)):
        raise ValueError("g_idx is not a group128 permutation")
    perm = np.argsort(gi, kind="stable").astype(np.int32)
    unpacked = np.empty((k, n), dtype=np.uint8)
    for bit in range(8):
        unpacked[bit::8] = ((raw >> (4 * bit)) & 15).astype(np.uint8)
    col = np.array([((lane >> 2) & 3) * 8 + (lane & 3) + (4 if lane & 16 else 0) for lane in range(32)], dtype=np.int32)
    ko = np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15], dtype=np.int32)
    packed_bytes = np.empty((n // 32, k // 16, 32, 8), dtype=np.uint8)
    for tile in range(n // 32):
        cols = tile * 32 + col
        for kg in range(k // 16):
            ks = perm[(kg // 8) * 128 + (kg % 8) * 16 + ko]
            vals = unpacked[ks[:, None], cols[None, :]].T
            packed_bytes[tile, kg] = vals[:, 0::2] | (vals[:, 1::2] << 4)
    # Kernel ABI is uint2 per lane (two uint32 words, four packed bytes each).
    # This is a bitwise view only: retain the eight Q4 nibbles unchanged.
    packed = packed_bytes.copy().view(np.uint32).reshape(n // 32, k // 16, 32, 2)
    return torch.from_numpy(packed.copy().view(np.int32)), torch.from_numpy(perm)


def load(build_dir: str, cuda_home: str | None = None):
    from pathlib import Path
    build = Path(build_dir); build.mkdir(parents=True, exist_ok=True)
    src = build / "prefill_q4_tc.cu"; src.write_text(CUDA_SRC)
    extra = ["-O3", "-lineinfo", "-gencode=arch=compute_70,code=sm_70"]
    return load_ext(src, build, extra)


def load_ext(src, build, extra):
    return load_extension(name="t011_prefill_q4_tc", source=str(src), build=str(build), extra=extra)


def load_extension(**kwargs):
    # Late import keeps CPU-only prepack tests independent of the CUDA build toolchain.
    from torch.utils.cpp_extension import load as cpp_load
    return cpp_load(name=kwargs["name"], sources=[kwargs["source"]], build_directory=str(kwargs["build"]), extra_cuda_cflags=kwargs["extra"], extra_cflags=["-O2"], with_cuda=True, verbose=True)
