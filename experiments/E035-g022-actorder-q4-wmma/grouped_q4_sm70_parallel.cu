#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;
constexpr int TM=16,TN=16,TK=16;

__global__ void gather_act_parallel(const half* x,half* gx,const int32_t* perm,int M,int K){
  int i=blockIdx.x*blockDim.x+threadIdx.x,total=M*K;
  if(i<total){int m=i/K,k=i%K;gx[i]=x[m*K+perm[k]];}
}

// One CTA owns one (group, M-tile, N-tile). It performs exactly one GPTQ
// group's WMMA dot products and writes unscaled FP32 tile elements.
__global__ void group_wmma_partial(const half* __restrict__ gx,
                                  const uint32_t* __restrict__ qw,
                                  float* __restrict__ partial,
                                  int M,int N,int K,int G){
  int nt=blockIdx.x,mt=blockIdx.y,g=blockIdx.z,lane=threadIdx.x;
  int m0=mt*TM,n0=nt*TN,k0=g*128;
  __shared__ half a[TM*TK];
  __shared__ half b[TK*TN];
  __shared__ float tile[TM*TN];
  wmma::fragment<wmma::matrix_a,TM,TN,TK,half,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,TM,TN,TK,half,wmma::row_major> bf;
  wmma::fragment<wmma::accumulator,TM,TN,TK,float> cf;
  wmma::fill_fragment(cf,0.f);
  for(int kk=0;kk<128;kk+=TK){
    for(int t=lane;t<TM*TK;t+=32){
      int r=t/TK,c=t%TK,m=m0+r,k=k0+kk+c;
      a[t]=(m<M)?gx[m*K+k]:__float2half(0.f);
    }
    for(int t=lane;t<TK*TN;t+=32){
      int r=t/TN,c=t%TN,k=k0+kk+r,n=n0+c;
      uint32_t word=qw[(k>>3)*N+n];
      b[t]=__int2half_rn(int((word>>((k&7)*4))&15)-8);
    }
    __syncthreads();
    wmma::load_matrix_sync(af,a,TK);
    wmma::load_matrix_sync(bf,b,TN);
    wmma::mma_sync(cf,af,bf,cf);
    __syncthreads();
  }
  wmma::store_matrix_sync(tile,cf,TN,wmma::mem_row_major);
  __syncthreads();
  for(int i=lane;i<TM*TN;i+=32){
    int m=m0+i/TN,n=n0+i%TN;
    if(m<M&&n<N)partial[(g*M+m)*N+n]=tile[i];
  }
}

// Per-output thread; preserve original strict ascending-group FP32 FMA order.
__global__ void reduce_groups_ordered(const float* __restrict__ partial,
                                      const half* __restrict__ scales,
                                      float* __restrict__ out,int M,int N,int G){
  int i=blockIdx.x*blockDim.x+threadIdx.x,total=M*N;
  if(i>=total)return;
  int n=i%N;float acc=0.f;
  for(int g=0;g<G;g++)acc=fmaf(partial[g*total+i],__half2float(scales[g*N+n]),acc);
  out[i]=acc;
}

torch::Tensor grouped_q4_sm70_parallel(torch::Tensor x,torch::Tensor qw,
                                      torch::Tensor scales,torch::Tensor perm,
                                      torch::Tensor scratch,torch::Tensor gx,
                                      torch::Tensor out,bool fp32_output){
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf,"x must be CUDA fp16");
  TORCH_CHECK(qw.is_cuda()&&qw.scalar_type()==at::kInt,"qw must be int32 CUDA");
  TORCH_CHECK(scales.is_cuda()&&scales.scalar_type()==at::kHalf&&perm.is_cuda()&&perm.scalar_type()==at::kInt,"metadata dtype/device");
  TORCH_CHECK(x.is_contiguous()&&qw.is_contiguous()&&scales.is_contiguous()&&perm.is_contiguous(),"contiguous inputs required");
  int M=x.size(0),K=x.size(1),N=qw.size(1),G=scales.size(0);
  TORCH_CHECK(K%128==0&&G==K/128&&N%16==0&&qw.size(0)==K/8,"shape/alignment");
  TORCH_CHECK(scratch.is_cuda()&&scratch.scalar_type()==at::kFloat&&scratch.sizes()==torch::IntArrayRef({G,M,N}),"partial scratch must be [G,M,N] FP32");
  TORCH_CHECK(gx.is_cuda()&&gx.sizes()==x.sizes()&&gx.scalar_type()==at::kHalf,"gather scratch mismatch");
  TORCH_CHECK(out.is_cuda()&&out.sizes()==torch::IntArrayRef({M,N})&&out.scalar_type()==at::kFloat,"output scratch mismatch");
  TORCH_CHECK(scratch.data_ptr()!=out.data_ptr(),"scratch and output must not alias");
  auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
  gather_act_parallel<<<(M*K+255)/256,256,0,stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),reinterpret_cast<half*>(gx.data_ptr<at::Half>()),perm.data_ptr<int32_t>(),M,K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  group_wmma_partial<<<dim3(N/TN,(M+TM-1)/TM,G),32,0,stream>>>(reinterpret_cast<const half*>(gx.data_ptr<at::Half>()),reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),scratch.data_ptr<float>(),M,N,K,G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  reduce_groups_ordered<<<(M*N+255)/256,256,0,stream>>>(scratch.data_ptr<float>(),reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),out.data_ptr<float>(),M,N,G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return fp32_output?out:out.to(at::kHalf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("grouped_q4_sm70_parallel",&grouped_q4_sm70_parallel);}
