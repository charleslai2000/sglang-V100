#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace {
constexpr int KTILE=128, NTILE=128, MROWS=4;

template<int KW>
__global__ void packed_fp32_partial_warpsliced(
    const half* __restrict__ x,const uint32_t* __restrict__ qw,
    const uint32_t* __restrict__ qz,const half* __restrict__ scales,
    const int32_t* __restrict__ gidx,float* __restrict__ partial,
    int M,int N,int K) {
  const int lane=threadIdx.x;
  const int kw=threadIdx.y;
  const int n0=blockIdx.x*NTILE+lane*4;
  const int m0=blockIdx.y*MROWS;
  const int kp=blockIdx.z;
  if(n0+3>=N || m0>=M) return;
  const int k0=kp*KTILE+kw*(KTILE/KW);
  float acc[4][4]={{0.f}},lo[4][4]={{0.f}};
  #pragma unroll 1
  for(int ki=0;ki<KTILE/KW;ki++) {
    const int k=k0+ki;
    const int g=gidx[k];
    const int row=k>>3,shift=(k&7)*4;
    const uint32_t w0=qw[row*N+n0],w1=qw[row*N+n0+1];
    const uint32_t w2=qw[row*N+n0+2],w3=qw[row*N+n0+3];
    const int q0=int((w0>>shift)&15u),q1=int((w1>>shift)&15u);
    const int q2=int((w2>>shift)&15u),q3=int((w3>>shift)&15u);
    const uint32_t zw=qz[g*(N/8)+(n0>>3)];
    const int z0=int((zw>>((n0&7)*4))&15u)+1;
    const int z1=int((zw>>(((n0+1)&7)*4))&15u)+1;
    const int z2=int((zw>>(((n0+2)&7)*4))&15u)+1;
    const int z3=int((zw>>(((n0+3)&7)*4))&15u)+1;
    const float s0=__half2float(scales[g*N+n0]);
    const float s1=__half2float(scales[g*N+n0+1]);
    const float s2=__half2float(scales[g*N+n0+2]);
    const float s3=__half2float(scales[g*N+n0+3]);
    const float wt0=float(q0-z0)*s0,wt1=float(q1-z1)*s1;
    const float wt2=float(q2-z2)*s2,wt3=float(q3-z3)*s3;
    #pragma unroll
    for(int mi=0;mi<MROWS;mi++) if(m0+mi<M) {
      const float xv=__half2float(x[(m0+mi)*K+k]);
      const float wt[4]={wt0,wt1,wt2,wt3};
      const int delta[4]={q0-z0,q1-z1,q2-z2,q3-z3};
      const float scale[4]={s0,s1,s2,s3};
      #pragma unroll
      for(int j=0;j<4;j++) {
        const double exact_product=double(xv)*double(delta[j])*double(scale[j]);
        const float p=float(exact_product);
        const float pe=float(exact_product-double(p));
        float sum,bb,smb,ae,be,err;
        asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(sum) : "f"(acc[mi][j]), "f"(p));
        asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(bb) : "f"(sum), "f"(-acc[mi][j]));
        asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(smb) : "f"(sum), "f"(-bb));
        asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(ae) : "f"(acc[mi][j]), "f"(-smb));
        asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(be) : "f"(p), "f"(-bb));
        asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(err) : "f"(ae), "f"(be));
        lo[mi][j]=lo[mi][j]+(err+pe);
        acc[mi][j]=sum;
      }
    }
  }
  __shared__ float warp_acc[KW][MROWS][NTILE];
  __shared__ float warp_lo[KW][MROWS][NTILE];
  #pragma unroll
  for(int mi=0;mi<MROWS;mi++) for(int j=0;j<4;j++) {
    warp_acc[kw][mi][lane*4+j]=acc[mi][j];
    warp_lo[kw][mi][lane*4+j]=lo[mi][j];
  }
  __syncthreads();
  if(kw==0) {
    #pragma unroll
    for(int mi=0;mi<MROWS;mi++) if(m0+mi<M) {
      #pragma unroll
      for(int j=0;j<4;j++) {
        float sum=warp_acc[0][mi][lane*4+j],comp=warp_lo[0][mi][lane*4+j];
        #pragma unroll
        for(int w=1;w<KW;w++) {
          const float value=warp_acc[w][mi][lane*4+j];
          float next,bb,smb,ae,be,err;
          asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(next) : "f"(sum), "f"(value));
          asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(bb) : "f"(next), "f"(-sum));
          asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(smb) : "f"(next), "f"(-bb));
          asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(ae) : "f"(sum), "f"(-smb));
          asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(be) : "f"(value), "f"(-bb));
          asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(err) : "f"(ae), "f"(be));
          comp=comp+err+warp_lo[w][mi][lane*4+j];
          sum=next;
        }
        partial[(kp*M+m0+mi)*N+n0+j]=sum+comp;
      }
    }
  }
}

template<typename Out>
__global__ void packed_fp32_reduce_warpsliced(const float* __restrict__ partial,
    Out* __restrict__ out,int total,int parts) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total)return;
  float sum=0.f;
  #pragma unroll 1
  for(int p=0;p<parts;p++) sum+=partial[p*total+i];
  if constexpr(std::is_same_v<Out,half>) out[i]=__float2half_rn(sum);
  else out[i]=sum;
}

template<int KW>
at::Tensor run(at::Tensor x,at::Tensor qw,at::Tensor qz,at::Tensor scales,
               at::Tensor gidx,bool return_fp32) {
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.is_contiguous(),"x must be contiguous CUDA FP16");
  TORCH_CHECK(qw.is_cuda()&&qw.scalar_type()==at::kInt&&qw.is_contiguous(),"qw must be contiguous CUDA int32");
  TORCH_CHECK(qz.is_cuda()&&qz.scalar_type()==at::kInt&&qz.is_contiguous(),"qz must be contiguous CUDA int32");
  TORCH_CHECK(scales.is_cuda()&&scales.scalar_type()==at::kHalf&&scales.is_contiguous(),"scales must be contiguous CUDA FP16");
  TORCH_CHECK(gidx.is_cuda()&&gidx.scalar_type()==at::kInt&&gidx.is_contiguous(),"g_idx must be contiguous CUDA int32");
  const int M=x.size(0),K=x.size(1),N=qw.size(1),parts=(K+KTILE-1)/KTILE;
  TORCH_CHECK(M>=1&&M<=MROWS&&K%KTILE==0&&N%NTILE==0&&qw.size(0)==K/8,"unsupported geometry");
  TORCH_CHECK(gidx.numel()==K&&qz.size(0)==K/KTILE&&qz.size(1)==N/8&&scales.size(0)==K/KTILE&&scales.size(1)==N,"metadata mismatch");
  auto partial=at::empty({parts,M,N},x.options().dtype(at::kFloat));
  auto out=at::empty({M,N},x.options().dtype(at::kFloat));
  auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
  dim3 block(32,KW,1),grid(N/NTILE,(M+MROWS-1)/MROWS,parts);
  packed_fp32_partial_warpsliced<KW><<<grid,block,0,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(qz.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      gidx.data_ptr<int32_t>(),partial.data_ptr<float>(),M,N,K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  packed_fp32_reduce_warpsliced<<<(M*N+255)/256,256,0,stream>>>(partial.data_ptr<float>(),out.data_ptr<float>(),M*N,parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(return_fp32) return out;
  return out.to(at::kHalf);
}
}

at::Tensor gptq_q4_fp32_m4_warp2(at::Tensor x,at::Tensor qw,at::Tensor qz,at::Tensor scales,at::Tensor gidx,bool fp32){return run<2>(x,qw,qz,scales,gidx,fp32);}
at::Tensor gptq_q4_fp32_m4_warp4(at::Tensor x,at::Tensor qw,at::Tensor qz,at::Tensor scales,at::Tensor gidx,bool fp32){return run<4>(x,qw,qz,scales,gidx,fp32);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("warp2",&gptq_q4_fp32_m4_warp2);m.def("warp4",&gptq_q4_fp32_m4_warp4);}
