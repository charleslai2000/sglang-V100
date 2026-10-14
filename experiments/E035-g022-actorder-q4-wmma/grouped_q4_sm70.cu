#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;
constexpr int TM=16, TN=16, TK=16;

__global__ void gather_act(const half* x, half* gx, const int32_t* perm,
                           int M, int K) {
  int i=blockIdx.x*blockDim.x+threadIdx.x, total=M*K;
  if(i<total) { int m=i/K,k=i%K; gx[i]=x[m*K+perm[k]]; }
}

// One CTA computes one MxN tile for one GPTQ group. B is reconstructed as
// exact small-integer FP16 values in shared memory, then fed directly to
// SM70 WMMA; accumulator remains FP32. No scale is applied until the reducer.
__global__ void group_wmma_fused(const half* __restrict__ gx,
                                 const uint32_t* __restrict__ qw,
                                 const half* __restrict__ scales,
                                 float* __restrict__ out,
                                 int M,int N,int K,int G) {
  int nt=blockIdx.x, mt=blockIdx.y;
  int m0=mt*TM,n0=nt*TN;
  __shared__ half a[TM*TK];
  __shared__ half b[TK*TN];
  __shared__ float tile[TM*TN];
  float acc[TM*TN];
  #pragma unroll
  for(int t=threadIdx.x;t<TM*TN;t+=blockDim.x) acc[t]=0.f;
  wmma::fragment<wmma::matrix_a,TM,TN,TK,half,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,TM,TN,TK,half,wmma::row_major> bf;
  wmma::fragment<wmma::accumulator,TM,TN,TK,float> cf;
  for(int g=0;g<G;g++) {
    wmma::fill_fragment(cf,0.f);
    int k0=g*128;
    for(int kk=0;kk<128;kk+=TK) {
      for(int t=threadIdx.x;t<TM*TK;t+=blockDim.x) {
        int r=t/TK,c=t%TK,m=m0+r,k=k0+kk+c;
        a[t]=(m<M)?gx[m*K+k]:__float2half(0.f);
      }
      for(int t=threadIdx.x;t<TK*TN;t+=blockDim.x) {
        int r=t/TN,c=t%TN,k=k0+kk+r,n=n0+c;
        int packed_row=k>>3,shift=(k&7)*4;
        uint32_t word=qw[packed_row*N+n];
        int q=(word>>shift)&15;
        b[t]=__int2half_rn(q-8);
      }
      __syncthreads();
      wmma::load_matrix_sync(af,a,TK);
      wmma::load_matrix_sync(bf,b,TN);
      wmma::mma_sync(cf,af,bf,cf);
      __syncthreads();
    }
    wmma::store_matrix_sync(tile,cf,TN,wmma::mem_row_major);
    __syncthreads();
    for(int t=threadIdx.x;t<TM*TN;t+=blockDim.x) {
      int n=n0+t%TN;
      if(n<N) acc[t]=fmaf(tile[t],__half2float(scales[g*N+n]),acc[t]);
    }
    __syncthreads();
  }
  for(int t=threadIdx.x;t<TM*TN;t+=blockDim.x) {
    int m=m0+t/TN,n=n0+t%TN;
    if(m<M && n<N) out[m*N+n]=acc[t];
  }
}

torch::Tensor grouped_q4_sm70(torch::Tensor x,torch::Tensor qw,
                              torch::Tensor scales,torch::Tensor perm,
                              torch::Tensor scratch,torch::Tensor gx,
                              torch::Tensor out,bool fp32_output) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kHalf,"x must be CUDA fp16");
  TORCH_CHECK(qw.is_cuda() && qw.scalar_type()==at::kInt,"qw must be int32 CUDA");
  TORCH_CHECK(scales.scalar_type()==at::kHalf && perm.scalar_type()==at::kInt,"metadata dtype");
  TORCH_CHECK(x.is_contiguous() && qw.is_contiguous() && scales.is_contiguous() && perm.is_contiguous(),"contiguous inputs required");
  int M=x.size(0),K=x.size(1),N=qw.size(1),G=scales.size(0);
  TORCH_CHECK(K%128==0 && G==K/128 && N%16==0 && qw.size(0)==K/8,"shape/alignment");
  TORCH_CHECK(scratch.sizes()==torch::IntArrayRef({M,N}) && scratch.scalar_type()==at::kFloat,"output scratch shape/dtype");
  TORCH_CHECK(gx.sizes()==x.sizes() && gx.scalar_type()==at::kHalf,"gather scratch shape/dtype");
  TORCH_CHECK(out.sizes()==torch::IntArrayRef({M,N}) && out.scalar_type()==at::kFloat,"output scratch shape/dtype");
  TORCH_CHECK(scratch.data_ptr()!=out.data_ptr(),"scratch and output must not alias");
  auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
  gather_act<<<(M*K+255)/256,256,0,stream>>>(reinterpret_cast<half const*>(x.data_ptr<at::Half>()),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),perm.data_ptr<int32_t>(),M,K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  group_wmma_fused<<<dim3(N/TN,(M+TM-1)/TM),32,0,stream>>>(reinterpret_cast<half const*>(gx.data_ptr<at::Half>()),reinterpret_cast<uint32_t const*>(qw.data_ptr<int32_t>()),reinterpret_cast<half const*>(scales.data_ptr<at::Half>()),out.data_ptr<float>(),M,N,K,G);
  return fp32_output?out:out.to(at::kHalf);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("grouped_q4_sm70",&grouped_q4_sm70);}
