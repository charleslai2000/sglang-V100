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

// R1: each lane owns eight statically indexed tile elements, eliminating the
// dynamically-indexed per-thread 256-float accumulator/local stack traffic.
__global__ void group_wmma_fused(const half* __restrict__ gx,
                                 const uint32_t* __restrict__ qw,
                                 const half* __restrict__ scales,
                                 float* __restrict__ out,
                                 int M,int N,int K,int G) {
  int nt=blockIdx.x, mt=blockIdx.y;
  int m0=mt*TM,n0=nt*TN,lane=threadIdx.x;
  __shared__ half a[TM*TK];
  __shared__ half b[TK*TN];
  __shared__ float tile[TM*TN];
  float acc0=0.f,acc1=0.f,acc2=0.f,acc3=0.f;
  float acc4=0.f,acc5=0.f,acc6=0.f,acc7=0.f;
  wmma::fragment<wmma::matrix_a,TM,TN,TK,half,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,TM,TN,TK,half,wmma::row_major> bf;
  wmma::fragment<wmma::accumulator,TM,TN,TK,float> cf;
  for(int g=0;g<G;g++) {
    wmma::fill_fragment(cf,0.f);
    int k0=g*128;
    for(int kk=0;kk<128;kk+=TK) {
      for(int t=lane;t<TM*TK;t+=blockDim.x) {
        int r=t/TK,c=t%TK,m=m0+r,k=k0+kk+c;
        a[t]=(m<M)?gx[m*K+k]:__float2half(0.f);
      }
      for(int t=lane;t<TK*TN;t+=blockDim.x) {
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
    int i0=lane, i1=lane+32, i2=lane+64, i3=lane+96;
    int i4=lane+128,i5=lane+160,i6=lane+192,i7=lane+224;
    int n0i=n0+i0%TN,n1i=n0+i1%TN,n2i=n0+i2%TN,n3i=n0+i3%TN;
    int n4i=n0+i4%TN,n5i=n0+i5%TN,n6i=n0+i6%TN,n7i=n0+i7%TN;
    if(n0i<N) acc0=fmaf(tile[i0],__half2float(scales[g*N+n0i]),acc0);
    if(n1i<N) acc1=fmaf(tile[i1],__half2float(scales[g*N+n1i]),acc1);
    if(n2i<N) acc2=fmaf(tile[i2],__half2float(scales[g*N+n2i]),acc2);
    if(n3i<N) acc3=fmaf(tile[i3],__half2float(scales[g*N+n3i]),acc3);
    if(n4i<N) acc4=fmaf(tile[i4],__half2float(scales[g*N+n4i]),acc4);
    if(n5i<N) acc5=fmaf(tile[i5],__half2float(scales[g*N+n5i]),acc5);
    if(n6i<N) acc6=fmaf(tile[i6],__half2float(scales[g*N+n6i]),acc6);
    if(n7i<N) acc7=fmaf(tile[i7],__half2float(scales[g*N+n7i]),acc7);
    __syncthreads();
  }
  int i0=lane, i1=lane+32, i2=lane+64, i3=lane+96;
  int i4=lane+128,i5=lane+160,i6=lane+192,i7=lane+224;
  int m0i=m0+i0/TN,n0i=n0+i0%TN;
  int m1i=m0+i1/TN,n1i=n0+i1%TN;
  int m2i=m0+i2/TN,n2i=n0+i2%TN;
  int m3i=m0+i3/TN,n3i=n0+i3%TN;
  int m4i=m0+i4/TN,n4i=n0+i4%TN;
  int m5i=m0+i5/TN,n5i=n0+i5%TN;
  int m6i=m0+i6/TN,n6i=n0+i6%TN;
  int m7i=m0+i7/TN,n7i=n0+i7%TN;
  if(m0i<M && n0i<N) out[m0i*N+n0i]=acc0;
  if(m1i<M && n1i<N) out[m1i*N+n1i]=acc1;
  if(m2i<M && n2i<N) out[m2i*N+n2i]=acc2;
  if(m3i<M && n3i<N) out[m3i*N+n3i]=acc3;
  if(m4i<M && n4i<N) out[m4i*N+n4i]=acc4;
  if(m5i<M && n5i<N) out[m5i*N+n5i]=acc5;
  if(m6i<M && n6i<N) out[m6i*N+n6i]=acc6;
  if(m7i<M && n7i<N) out[m7i*N+n7i]=acc7;
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
