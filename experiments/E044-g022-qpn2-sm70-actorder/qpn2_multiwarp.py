#!/usr/bin/env python3
"""T011 multiwarp QPN2 experiment generator and strict fixture qualifier."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'E035-g022-actorder-q4-wmma'))
from gpu_uuid_lock import assert_cuda_uuid

SRC=r'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>
#define MMA(C,A0,A1,B0,B1) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n" : "+f"(C[0]),"+f"(C[1]),"+f"(C[2]),"+f"(C[3]),"+f"(C[4]),"+f"(C[5]),"+f"(C[6]),"+f"(C[7]) : "r"(A0),"r"(A1),"r"(B0),"r"(B1))
__device__ __forceinline__ unsigned h2(half a,half b){return (unsigned)__half_as_ushort(a)|((unsigned)__half_as_ushort(b)<<16);}
__global__ void gather(const half*x,half*gx,const int*p,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<M*K)gx[i]=x[(size_t)(i/K)*K+p[i%K]];}
template<int SPLITK> __global__ void kern(const uint2*q,const half*x,const half*s,float*out,int M,int N,int K,int G){
 __shared__ float partial[SPLITK][8][32];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0),tile=blockIdx.x;
 int groups_per_warp=G/SPLITK,g0=warp*groups_per_warp; const uint2*base=q+(size_t)tile*(K/16)*32+lane;
 float acc[8]={0};
 for(int g=g0;g<g0+groups_per_warp;++g){float c[8]={0};
  for(int kg=0;kg<8;++kg){uint2 z=*(base+(size_t)(g*8+kg)*32);uint32_t w[2]={z.x,z.y};unsigned B[8];for(int wi=0;wi<2;++wi)for(int bi=0;bi<4;++bi){int p0=(w[wi]>>(bi*4))&15,p1=(w[wi]>>((bi+4)*4))&15;B[wi*4+bi]=h2(__int2half_rn(p0-8),__int2half_rn(p1-8));}
   int kk=kg*16;const half*xr=x+(size_t)r*K+(size_t)g*128+kk;uint4 u=make_uint4(0,0,0,0),v=make_uint4(0,0,0,0);if(r<M){u=*reinterpret_cast<const uint4*>(xr);v=*reinterpret_cast<const uint4*>(xr+8);}unsigned A[4],V[4];*reinterpret_cast<uint4*>(A)=u;*reinterpret_cast<uint4*>(V)=v;MMA(c,A[0],A[1],B[0],B[1]);MMA(c,A[2],A[3],B[2],B[3]);MMA(c,V[0],V[1],B[4],B[5]);MMA(c,V[2],V[3],B[6],B[7]);}
  #pragma unroll
  for(int i=0;i<8;++i){int row=(i&2)|((lane&16)?4:0)|(lane&1),col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2),m=row,n=tile*32+qp*8+col;if(m<M)acc[i]+=c[i]*__half2float(s[(size_t)g*N+n]);}
 }
 #pragma unroll
 for(int i=0;i<8;++i){int row=(i&2)|((lane&16)?4:0)|(lane&1),col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2);partial[warp][row][qp*8+col]=acc[i];}
 __syncthreads();
 for(int e=threadIdx.x;e<8*32;e+=blockDim.x){int row=e>>5,col=e&31;float v=0.f;
  #pragma unroll
  for(int w=0;w<SPLITK;++w)v=v+partial[w][row][col];
  if(row<M)out[(size_t)row*N+(size_t)tile*32+col]=v;
 }
}
torch::Tensor run(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor p,torch::Tensor gx,torch::Tensor o,int split){int M=x.size(0),K=x.size(1),N=s.size(1),G=s.size(0);auto st=at::cuda::getCurrentCUDAStream();gather<<<(M*K+255)/256,256,0,st>>>(reinterpret_cast<half*>(x.data_ptr<at::Half>()),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),p.data_ptr<int>(),M,K);
 #define LAUNCH(S) kern<S><<<N/32,32*S,0,st>>>(reinterpret_cast<uint2*>(q.data_ptr<int>()),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),reinterpret_cast<half*>(s.data_ptr<at::Half>()),o.data_ptr<float>(),M,N,K,G)
 if(split==8)LAUNCH(8);else if(split==16)LAUNCH(16);else if(split==32)LAUNCH(32);else TORCH_CHECK(false,"split must be 8,16,32");
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixtures',default='/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');p.add_argument('--source-fixtures',default='/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures');p.add_argument('--build',default=str(HERE/'build-t011'));p.add_argument('--out',default=str(HERE/'results/t011-correctness.jsonl'));a=p.parse_args()
 if torch.cuda.device_count()!=1: raise SystemExit('require CUDA_VISIBLE_DEVICES set to only target UUID')
 assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189')
 build=Path(a.build);build.mkdir(parents=True,exist_ok=True);sf=build/'qpn2_multiwarp.cu';sf.write_text(SRC)
 mod=load(name='qpn2_multiwarp',sources=[str(sf)],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory=str(build),verbose=True)
 root=Path(a.fixtures);source=Path(a.source_fixtures);rows=[]
 for copied in sorted(root.iterdir()):
  if not copied.is_dir() or copied.name.startswith('.'):continue
  cm=json.loads((copied/'manifest.json').read_text());froot=copied;fm=cm;T=fm['tensors'];K=T['g_idx']['shape'][0];N=T['qweight']['shape'][1];G=T['scales']['shape'][0]
  def raw(path,dtype,shape):return np.fromfile(path,dtype=dtype).reshape(shape)
  def read(n,dtype):return raw(froot/(n+'.bin'),dtype,T[n]['shape'])
  xfull=read('x','<f2');qw=read('qweight','<u4');qz=read('qzeros','<u4');sc=read('scales','<f2');g=read('g_idx','<i4')
  q=np.empty((K,N),np.uint8);z=np.empty((G,N),np.uint8)
  for b in range(8):q[b::8]=((qw>>(4*b))&15).astype(np.uint8);z[:,b::8]=((qz>>(4*b))&15).astype(np.uint8)
  if not np.all(z==7):raise RuntimeError('unsupported qzero')
  perm=np.concatenate([np.flatnonzero(g==gi) for gi in range(G)]).astype(np.int32)
  if not np.array_equal(g[perm],np.repeat(np.arange(G),128)):raise RuntimeError('non-group128 permutation')
  col=np.array([((l>>2)&3)*8+(l&3)+(4 if l&16 else 0) for l in range(32)]);ko=np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);packed=np.empty((N//32,K//16,32,8),np.uint8)
  for tile in range(N//32):
   for kg in range(K//16):
    sk=perm[(kg//8)*128+(kg%8)*16+ko];v=q[sk[:,None],(tile*32+col)[None,:]].T;packed[tile,kg]=v[:,0::2]|(v[:,1::2]<<4)
  qt=torch.from_numpy(packed.view(np.int32).copy()).cuda();st=torch.from_numpy(sc.copy()).cuda();pt=torch.from_numpy(perm).cuda();w64=(q.astype(np.float64)-(z[g].astype(np.float64)+1))*sc.astype(np.float64)[g]
  for M in (1,2,3,4):
   tests=[('ordinary',xfull[:M])]
   if cm['prefix']=='43.mlp.down_proj' and M==4:tests.append(('4x_saved_stress',raw(Path('/mnt/data/pc01/sglang-v100-migration/experiments/E035/packed-word-fixture-package/c57_stress_x_fp16.bin'),'<f2',(4,K))))
   for label,xnp in tests:
    xt=torch.from_numpy(np.ascontiguousarray(xnp)).cuda();gx=torch.empty_like(xt);o=torch.empty((M,N),device='cuda',dtype=torch.float32);ref=torch.from_numpy((xnp.astype(np.float64)@w64).astype(np.float32)).cuda()
    for split in (8,16,32):
     if G%split:continue
     y=mod.run(xt,qt,st,pt,gx,o,split);torch.cuda.synchronize();tol=1e-4+5e-5*ref.abs();err=(y-ref).abs();bad=int((err>tol).sum().item());again=mod.run(xt,qt,st,pt,gx,o,split).clone();torch.cuda.synchronize()
     fresh=torch.empty_like(o);y2=mod.run(xt,qt,st,pt,gx,fresh,split);torch.cuda.synchronize();independent=torch.equal(y,y2)
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph,stream=stream):gy=mod.run(xt,qt,st,pt,gx,o,split)
      graph.replay();torch.cuda.synchronize()
     torch.cuda.current_stream().wait_stream(stream);gb=int(((o-ref).abs()>tol).sum().item());cast=torch.equal(gy.half(),y.half())
     row={'fixture':cm['prefix'],'manifest_sha256':hashlib.sha256((froot/'manifest.json').read_bytes()).hexdigest(),'M':M,'input':label,'splitk':split,'strict_bad':bad,'graph_strict_bad':gb,'max_ratio':float((err/tol).max()),'repeat_equal':bool(torch.equal(y,again)),'independent_scratch_equal':independent,'graph_cast_equal_to_eager':cast,'binary_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(sf.read_bytes()).hexdigest()};rows.append(row);print(json.dumps(row),flush=True)
     if bad or gb or not row['repeat_equal'] or not independent:raise RuntimeError(f'strict/state failure: {row}')

 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
if __name__=='__main__':main()
