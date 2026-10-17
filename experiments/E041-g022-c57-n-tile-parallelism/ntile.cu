#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include <tuple>
namespace {
constexpr int KTILE=128,MROWS=4;
template<int NT> __global__ void partial_nt(const half* x,const uint32_t* qw,const uint32_t* qz,const half* sc,const int32_t* gi,float* p,int M,int N,int K){
 constexpr int PER=NT==128?4:NT==64?2:1;int lane=threadIdx.x,n0=blockIdx.x*NT+lane*PER,m0=blockIdx.y*MROWS,kp=blockIdx.z;if(n0+PER>N||m0>=M)return;
 float acc[MROWS][PER]={};int kend=min((kp+1)*KTILE,K);
 for(int k=kp*KTILE;k<kend;++k){int g=gi[k],shift=(k&7)*4,row=k>>3;uint32_t zw=qz[g*(N/8)+(n0>>3)];
  #pragma unroll
  for(int mi=0;mi<MROWS;++mi)if(m0+mi<M){float a=__half2float(x[(m0+mi)*K+k]);
   #pragma unroll
   for(int j=0;j<PER;++j){int n=n0+j;uint32_t w=qw[row*N+n];int q=int((w>>shift)&15u),z=int((zw>>((n&7)*4))&15u)+1;float s=__half2float(sc[g*N+n]);float wt=float(q-z)*s;acc[mi][j]=fmaf(a,wt,acc[mi][j]);}}
 }
 #pragma unroll
 for(int mi=0;mi<MROWS;++mi)if(m0+mi<M)
  #pragma unroll
  for(int j=0;j<PER;++j)p[(kp*M+m0+mi)*N+n0+j]=acc[mi][j];
}
template<typename Out> __global__ void reduce(const float* p,Out* out,int total,int parts){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;float s=0.f;
 #pragma unroll 1
 for(int k=0;k<parts;++k)s+=p[k*total+i];if constexpr(std::is_same_v<Out,half>)out[i]=__float2half_rn(s);else out[i]=s;}
template<int NT,typename Out> at::Tensor run(at::Tensor x,at::Tensor qw,at::Tensor qz,at::Tensor sc,at::Tensor gi){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.is_contiguous());TORCH_CHECK(qw.is_cuda()&&qw.scalar_type()==at::kInt&&qw.is_contiguous());TORCH_CHECK(qz.is_cuda()&&qz.scalar_type()==at::kInt&&qz.is_contiguous());TORCH_CHECK(sc.is_cuda()&&sc.scalar_type()==at::kHalf&&sc.is_contiguous());TORCH_CHECK(gi.is_cuda()&&gi.scalar_type()==at::kInt&&gi.is_contiguous());
 int M=x.size(0),K=x.size(1),N=qw.size(1),parts=(K+KTILE-1)/KTILE;TORCH_CHECK(M>=1&&M<=MROWS&&K%KTILE==0&&N%NT==0&&qw.size(0)==K/8);TORCH_CHECK(gi.numel()==K&&qz.size(0)==K/KTILE&&qz.size(1)==N/8&&sc.size(0)==K/KTILE&&sc.size(1)==N);
 auto p=at::empty({parts,M,N},x.options().dtype(at::kFloat));auto out=at::empty({M,N},x.options().dtype(std::is_same_v<Out, half> ? at::kHalf : at::kFloat));auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
 partial_nt<NT><<<dim3(N/NT,(M+MROWS-1)/MROWS,parts),32,0,stream>>>((const half*)x.data_ptr<at::Half>(),(const uint32_t*)qw.data_ptr<int32_t>(),(const uint32_t*)qz.data_ptr<int32_t>(),(const half*)sc.data_ptr<at::Half>(),gi.data_ptr<int32_t>(),p.data_ptr<float>(),M,N,K);C10_CUDA_KERNEL_LAUNCH_CHECK();
 reduce<Out><<<(M*N+255)/256,256,0,stream>>>(p.data_ptr<float>(),(Out*)out.data_ptr(),M*N,parts);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
}
at::Tensor ntile128(at::Tensor x,at::Tensor q,at::Tensor z,at::Tensor s,at::Tensor g){return run<128,half>(x,q,z,s,g);}at::Tensor ntile64(at::Tensor x,at::Tensor q,at::Tensor z,at::Tensor s,at::Tensor g){return run<64,half>(x,q,z,s,g);}at::Tensor ntile32(at::Tensor x,at::Tensor q,at::Tensor z,at::Tensor s,at::Tensor g){return run<32,half>(x,q,z,s,g);}
at::Tensor ntile128_fp32(at::Tensor x,at::Tensor q,at::Tensor z,at::Tensor s,at::Tensor g){return run<128,float>(x,q,z,s,g);}at::Tensor ntile64_fp32(at::Tensor x,at::Tensor q,at::Tensor z,at::Tensor s,at::Tensor g){return run<64,float>(x,q,z,s,g);}at::Tensor ntile32_fp32(at::Tensor x,at::Tensor q,at::Tensor z,at::Tensor s,at::Tensor g){return run<32,float>(x,q,z,s,g);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("ntile128",&ntile128);m.def("ntile64",&ntile64);m.def("ntile32",&ntile32);m.def("ntile128_fp32",&ntile128_fp32);m.def("ntile64_fp32",&ntile64_fp32);m.def("ntile32_fp32",&ntile32_fp32);}
