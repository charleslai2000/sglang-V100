#!/usr/bin/env python3
"""E044 QPN2 quadpair GPTQ prototype, donor-layout weight prepack + exact FP32 dot check."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--fixture',default=str(HERE/'fixtures/l43-down/packed-word-fixture-package'));p.add_argument('--build',default=str(HERE/'build-qpn'));p.add_argument('--out',default=str(HERE/'results/qpn-prototype.json'));p.add_argument('--stress',action='store_true');a=p.parse_args()
D=Path(a.fixture);mf=json.loads((D/'manifest.json').read_text());T=mf['tensors'];K=T['g_idx']['shape'][0];N=T['qweight']['shape'][1];G=T['scales']['shape'][0];M=T['x']['shape'][0]
def raw(n,d,s):return np.fromfile(D/(n+'.bin'),dtype=d).reshape(s)
qw=raw('qweight','<u4',(K//8,N));qz=raw('qzeros','<u4',(G,N//8));sc=raw('scales','<f2',(G,N));g=raw('g_idx','<i4',(K,));x=raw('x','<f2',(M,K))
q=np.empty((K,N),np.uint8);z=np.empty((G,N),np.uint8)
for b in range(8):q[b::8]=((qw>>(4*b))&15).astype(np.uint8);z[:,b::8]=((qz>>(4*b))&15).astype(np.uint8)
assert np.all(z==7) and K==G*128 and N%32==0 and len(np.unique(g))==G
perm=np.concatenate([np.flatnonzero(g==gi) for gi in range(G)]).astype(np.int32)
col=np.array([((l>>2)&3)*8+(l&3)+(4 if l&16 else 0) for l in range(32)]);korder=np.array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);tiles=N//32;groups16=K//16
packed=np.empty((tiles,groups16,32,8),np.uint8)
for tile in range(tiles):
 for kg in range(groups16):
  sk=perm[(kg//8)*128+(kg%8)*16+korder];v=q[sk[:,None],(tile*32+col)[None,:]].T;packed[tile,kg]=v[:,0::2]|(v[:,1::2]<<4)
packed=np.ascontiguousarray(packed);xg=np.ascontiguousarray(x[:,perm])
w=(q.astype(np.float64)-(z[g].astype(np.float64)+1))*sc.astype(np.float64)[g]
def tt(a,d):return torch.from_numpy(np.ascontiguousarray(a)).to(device='cuda',dtype=d)
q_t=torch.from_numpy(packed.view(np.int32).copy()).cuda();s_t=tt(sc,torch.float16);p_t=tt(perm,torch.int32)
src=r'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>
#define MMA(C,A0,A1,B0,B1) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n" : "+f"(C[0]),"+f"(C[1]),"+f"(C[2]),"+f"(C[3]),"+f"(C[4]),"+f"(C[5]),"+f"(C[6]),"+f"(C[7]) : "r"(A0),"r"(A1),"r"(B0),"r"(B1))
__device__ __forceinline__ unsigned h2(half a,half b){return (unsigned)__half_as_ushort(a)|((unsigned)__half_as_ushort(b)<<16);}
__global__ void gather(const half*x,half*gx,const int*p,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<M*K)gx[i]=x[(size_t)(i/K)*K+p[i%K]];}
__global__ void kern(const uint2*q,const half*x,const half*s,float*out,int M,int N,int K,int G){int lane=threadIdx.x,qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0),tile=blockIdx.x;float acc[8]={0},comp[8]={0};const uint2*base=q+(size_t)tile*(K/16)*32+lane;
 for(int g=0;g<G;++g){float c[8]={0};for(int kg=0;kg<8;++kg){uint2 z=*(base+(size_t)(g*8+kg)*32);uint32_t w[2]={z.x,z.y};unsigned B[8];for(int wi=0;wi<2;++wi){for(int bi=0;bi<4;++bi){int p0=(w[wi]>>(bi*4))&15,p1=(w[wi]>>((bi+4)*4))&15;B[wi*4+bi]=h2(__int2half_rn(p0-8),__int2half_rn(p1-8));}}int kk=kg*16;const half*xr=x+(size_t)r*K+(size_t)g*128+kk;uint4 u=make_uint4(0,0,0,0),v=make_uint4(0,0,0,0);if(r<M){u=*reinterpret_cast<const uint4*>(xr);v=*reinterpret_cast<const uint4*>(xr+8);}unsigned A[4],V[4];*reinterpret_cast<uint4*>(A)=u;*reinterpret_cast<uint4*>(V)=v;MMA(c,A[0],A[1],B[0],B[1]);MMA(c,A[2],A[3],B[2],B[3]);MMA(c,V[0],V[1],B[4],B[5]);MMA(c,V[2],V[3],B[6],B[7]);}
 #pragma unroll
 for(int i=0;i<8;++i){int row=(i&2)|((lane&16)?4:0)|(lane&1),col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2),m=row,n=tile*32+qp*8+col;if(m<M){float val=c[i]*__half2float(s[(size_t)g*N+n]);float yy=val-comp[i],tt=acc[i]+yy;comp[i]=(tt-acc[i])-yy;acc[i]=tt;}}}
 #pragma unroll
 for(int i=0;i<8;++i){int row=(i&2)|((lane&16)?4:0)|(lane&1),col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2),m=row,n=tile*32+qp*8+col;if(m<M)out[(size_t)m*N+n]=acc[i];}}
torch::Tensor run(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor p,torch::Tensor gx,torch::Tensor o){int M=x.size(0),K=x.size(1),N=s.size(1),G=s.size(0);auto st=at::cuda::getCurrentCUDAStream();gather<<<(M*K+255)/256,256,0,st>>>(reinterpret_cast<half*>(x.data_ptr<at::Half>()),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),p.data_ptr<int>(),M,K);kern<<<N/32,32,0,st>>>(reinterpret_cast<uint2*>(q.data_ptr<int>()),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),reinterpret_cast<half*>(s.data_ptr<at::Half>()),o.data_ptr<float>(),M,N,K,G);return o;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}'''
Path(a.build).mkdir(parents=True,exist_ok=True);srcfile=Path(a.build)/'e044_qpn2_gptq.cu';srcfile.write_text(src);mod=load(name='e044_qpn2_gptq',sources=[str(srcfile)],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory=a.build,verbose=True)
res=[]
def check(xin,label):
 xt=tt(xin,torch.float16);gx=torch.empty_like(xt);o=torch.empty((M,N),device='cuda',dtype=torch.float32);ref=torch.from_numpy((xin.astype(np.float64)@w).astype(np.float32)).cuda();y=mod.run(xt,q_t,s_t,p_t,gx,o);torch.cuda.synchronize();tol=1e-4+5e-5*ref.abs();err=(y-ref).abs();bad=err>tol;again=mod.run(xt,q_t,s_t,p_t,gx,o).clone();torch.cuda.synchronize();st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(st):
  gr=torch.cuda.CUDAGraph()
  with torch.cuda.graph(gr,stream=st):gy=mod.run(xt,q_t,s_t,p_t,gx,o)
  gr.replay();torch.cuda.synchronize()
 torch.cuda.current_stream().wait_stream(st);go=o.clone();gb=int(((go-ref).abs()>tol).sum());row={'input':label,'strict_bad':int(bad.sum()),'graph_strict_bad':gb,'max_abs_error':float(err.max()),'max_tolerance_ratio':float((err/tol).max()),'repeat_equal':bool(torch.equal(y,again)),'graph_cast_equal':bool(torch.equal(gy.half(),y.half())),'finite':bool(torch.isfinite(y).all()),'sha256':hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest()}
 if bad.any():row['bad_points']=[{'m':i,'n':j,'actual':float(y[i,j]),'ref':float(ref[i,j]),'err':float(err[i,j]),'allowed':float(tol[i,j])} for i,j in bad.nonzero()[:20].tolist()]
 return row
res.append(check(x,'ordinary'))
if a.stress:res.append(check(raw('c57_stress_x_fp16','<f2',(M,K)),'4x_saved_stress'))
out={'fixture':mf['prefix'],'manifest_sha256':hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(srcfile.read_bytes()).hexdigest(),'prepack_sha256':hashlib.sha256(packed.tobytes()).hexdigest(),'binary_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),'results':res}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
