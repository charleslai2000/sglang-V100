#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Experimental dataflow: packed GPTQ-v1 Q4 is consumed in place. CTA grid
// partitions N, M and K; each CTA writes one FP32 partial for each output in a
// 128-K chunk. A deterministic ordered reducer writes each output once, so no
// FP16 atomic or dense dequantized B materialization occurs.
constexpr int KTILE=128;
constexpr int NTILE=128;
constexpr int MROWS=4;
__global__ void q4_fp32_partial(
    const half* __restrict__ x,const uint32_t* __restrict__ qw,
    const uint32_t* __restrict__ qz,const half* __restrict__ scales,
    const int32_t* __restrict__ gidx,float* __restrict__ partial,
    int M,int N,int K,int parts) {
  int n0=blockIdx.x*NTILE+threadIdx.x*4;
  int m0=blockIdx.y*MROWS,kp=blockIdx.z;
  if(n0+3>=N || m0+3>=M) return;
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
  float b0=0.f,b1=0.f,b2=0.f,b3=0.f;
  float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
  float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
  int k0=kp*KTILE,kend=min(k0+KTILE,K);
  #pragma unroll 1
  for(int k=k0;k<kend;k++) {
    int g=gidx[k],pack_row=k>>3,shift=(k&7)*4;
    uint32_t wp0=qw[pack_row*N+n0],wp1=qw[pack_row*N+n0+1];
    uint32_t wp2=qw[pack_row*N+n0+2],wp3=qw[pack_row*N+n0+3];
    int q0=(wp0>>shift)&15,q1=(wp1>>shift)&15,q2=(wp2>>shift)&15,q3=(wp3>>shift)&15;
    uint32_t zword=qz[g*(N/8)+(n0>>3)];
    int zp0=((zword>>((n0&7)*4))&15)+1,zp1=((zword>>(((n0+1)&7)*4))&15)+1;
    int zp2=((zword>>(((n0+2)&7)*4))&15)+1,zp3=((zword>>(((n0+3)&7)*4))&15)+1;
    float s0=__half2float(scales[g*N+n0]),s1=__half2float(scales[g*N+n0+1]);
    float s2=__half2float(scales[g*N+n0+2]),s3=__half2float(scales[g*N+n0+3]);
    float x0=__half2float(x[(m0+0)*K+k]),x1=__half2float(x[(m0+1)*K+k]);
    float x2=__half2float(x[(m0+2)*K+k]),x3=__half2float(x[(m0+3)*K+k]);
    a0=fmaf(x0,float(q0-zp0)*s0,a0);a1=fmaf(x0,float(q1-zp1)*s1,a1);a2=fmaf(x0,float(q2-zp2)*s2,a2);a3=fmaf(x0,float(q3-zp3)*s3,a3);
    b0=fmaf(x1,float(q0-zp0)*s0,b0);b1=fmaf(x1,float(q1-zp1)*s1,b1);b2=fmaf(x1,float(q2-zp2)*s2,b2);b3=fmaf(x1,float(q3-zp3)*s3,b3);
    c0=fmaf(x2,float(q0-zp0)*s0,c0);c1=fmaf(x2,float(q1-zp1)*s1,c1);c2=fmaf(x2,float(q2-zp2)*s2,c2);c3=fmaf(x2,float(q3-zp3)*s3,c3);
    d0=fmaf(x3,float(q0-zp0)*s0,d0);d1=fmaf(x3,float(q1-zp1)*s1,d1);d2=fmaf(x3,float(q2-zp2)*s2,d2);d3=fmaf(x3,float(q3-zp3)*s3,d3);
  }
  partial[(kp*M+m0+0)*N+n0]=a0;partial[(kp*M+m0+0)*N+n0+1]=a1;partial[(kp*M+m0+0)*N+n0+2]=a2;partial[(kp*M+m0+0)*N+n0+3]=a3;
  partial[(kp*M+m0+1)*N+n0]=b0;partial[(kp*M+m0+1)*N+n0+1]=b1;partial[(kp*M+m0+1)*N+n0+2]=b2;partial[(kp*M+m0+1)*N+n0+3]=b3;
  partial[(kp*M+m0+2)*N+n0]=c0;partial[(kp*M+m0+2)*N+n0+1]=c1;partial[(kp*M+m0+2)*N+n0+2]=c2;partial[(kp*M+m0+2)*N+n0+3]=c3;
  partial[(kp*M+m0+3)*N+n0]=d0;partial[(kp*M+m0+3)*N+n0+1]=d1;partial[(kp*M+m0+3)*N+n0+2]=d2;partial[(kp*M+m0+3)*N+n0+3]=d3;
}
__global__ void q4_fp32_reduce(
    const float* __restrict__ partial,float* __restrict__ out,
    int M,int N,int parts) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x,total=M*N;
  if(idx>=total) return;
  float acc=0.f;
  #pragma unroll 1
  for(int p=0;p<parts;p++) acc+=partial[p*total+idx];
  out[idx]=acc;
}
torch::Tensor packed_fp32_gptq(torch::Tensor x,torch::Tensor qw,torch::Tensor qz,
                               torch::Tensor scales,torch::Tensor gidx,bool fp32_output) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kHalf,"x must be CUDA fp16");
  TORCH_CHECK(qw.scalar_type()==at::kInt && qz.scalar_type()==at::kInt && gidx.scalar_type()==at::kInt,"packed tensors must be int32");
  int M=x.size(0),K=x.size(1),N=qw.size(1),parts=(K+KTILE-1)/KTILE;
  TORCH_CHECK(K%128==0 && N%128==0 && K%8==0 && N%8==0,"geometry alignment");
  auto out_fp32=torch::empty({M,N},x.options().dtype(at::kFloat));
  auto tmp=torch::empty({parts,M,N},x.options().dtype(at::kFloat));
  auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
  q4_fp32_partial<<<dim3(N/NTILE,M/MROWS,parts),NTILE/4,0,stream>>>(
    reinterpret_cast<const half*>(x.data_ptr<at::Half>()),reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
    reinterpret_cast<const uint32_t*>(qz.data_ptr<int32_t>()),reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
    gidx.data_ptr<int32_t>(),tmp.data_ptr<float>(),M,N,K,parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  q4_fp32_reduce<<<(M*N+255)/256,256,0,stream>>>(tmp.data_ptr<float>(),out_fp32.data_ptr<float>(),M,N,parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return fp32_output?out_fp32:out_fp32.to(at::kHalf);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("packed_fp32_gptq",&packed_fp32_gptq,py::arg("x"),py::arg("qw"),py::arg("qz"),py::arg("scales"),py::arg("gidx"),py::arg("fp32_output")=false);}
