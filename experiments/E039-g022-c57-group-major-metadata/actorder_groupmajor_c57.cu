#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace {
constexpr int KTILE=128, NTILE=128, MROWS=4;

// c57-compatible CTA/N mapping and four adjacent output channels/thread.
// qmajor preserves GPTQ [K/8,N] and N-contiguous word layout. Since the
// activation and packed rows are stable-group-major, each split-K tile owns
// one exact GPTQ group; metadata is loaded once outside its K loop.
__global__ void groupmajor_partial(const half* __restrict__ x,
    const uint32_t* __restrict__ qw,const uint32_t* __restrict__ qz,
    const half* __restrict__ scales,float* __restrict__ partial,
    int M,int N,int K,int groups) {
  const int n0=blockIdx.x*NTILE+threadIdx.x*4;
  const int m0=blockIdx.y*MROWS;
  const int kp=blockIdx.z;
  if(n0+3>=N||m0>=M)return;
  const int group=kp;
  const int zrow=group*(N/8)+(n0>>3);
  const uint32_t zw=qz[zrow];
  const int zp[4]={int((zw>>((n0&7)*4))&15u)+1,
    int((zw>>(((n0+1)&7)*4))&15u)+1,
    int((zw>>(((n0+2)&7)*4))&15u)+1,
    int((zw>>(((n0+3)&7)*4))&15u)+1};
  const float s[4]={__half2float(scales[group*N+n0]),
    __half2float(scales[group*N+n0+1]),__half2float(scales[group*N+n0+2]),
    __half2float(scales[group*N+n0+3])};
  float acc[4][4]={{0.f}},comp[4][4]={{0.f}};
  const int k0=kp*KTILE;
  #pragma unroll 1
  for(int kl=0;kl<KTILE;kl++) {
    const int k=k0+kl;
    const int row=k>>3,shift=(k&7)*4;
    const uint32_t w0=qw[row*N+n0],w1=qw[row*N+n0+1];
    const uint32_t w2=qw[row*N+n0+2],w3=qw[row*N+n0+3];
    const int q[4]={int((w0>>shift)&15u),int((w1>>shift)&15u),
                    int((w2>>shift)&15u),int((w3>>shift)&15u)};
    #pragma unroll
    for(int mi=0;mi<4;mi++) if(m0+mi<M) {
      const float xv=__half2float(x[(m0+mi)*K+k]);
      #pragma unroll
      for(int j=0;j<4;j++) {
        const float wt=float(q[j]-zp[j])*s[j];
        const float product=__fmul_rn(xv,wt);
        const float product_error=__fmaf_rn(xv,wt,-product);
        const float adjusted=product-comp[mi][j];
        const float next=acc[mi][j]+adjusted;
        comp[mi][j]=((next-acc[mi][j])-adjusted)-product_error;
        acc[mi][j]=next;
      }
    }
  }
  #pragma unroll
  for(int mi=0;mi<4;mi++) if(m0+mi<M)
    #pragma unroll
    for(int j=0;j<4;j++) partial[(kp*M+m0+mi)*N+n0+j]=acc[mi][j];
}

// Exact-c57 reducer association: ascending partition, plain FP32 addition.
template<typename Out>
__global__ void ordered_fp32_sum(const float* __restrict__ partial,
    Out* __restrict__ out,int total,int parts) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total)return;
  float sum=0.f,compensation=0.f;
  #pragma unroll 1
  for(int p=0;p<parts;p++) {
    const float adjusted=partial[p*total+i]-compensation;
    const float next=sum+adjusted;
    compensation=(next-sum)-adjusted;
    sum=next;
  }
  if constexpr (std::is_same_v<Out,half>) out[i]=__float2half_rn(sum);
  else out[i]=sum;
}
}

std::vector<torch::Tensor> stable_groupmajor_pack(torch::Tensor qweight,
    torch::Tensor gidx,int group_size) {
  TORCH_CHECK(qweight.is_cuda()&&qweight.scalar_type()==at::kInt&&qweight.is_contiguous(),"qweight must be CUDA contiguous int32");
  TORCH_CHECK(gidx.is_cuda()&&gidx.scalar_type()==at::kInt&&gidx.is_contiguous(),"g_idx must be CUDA contiguous int32");
  const int K=gidx.numel(),N=qweight.size(1);
  TORCH_CHECK(group_size==KTILE&&K%KTILE==0&&N%NTILE==0&&qweight.size(0)==K/8,"unsupported Q4 geometry");
  const int G=K/KTILE;
  // CPU mapping/packing is load-time-only; use int64 storage to avoid signed shifts.
  auto gi=gidx.to(at::kCPU).contiguous();
  auto ga=gi.accessor<int32_t,1>();
  std::vector<int> cursor(G,0),perm(K);
  for(int k=0;k<K;k++) {const int g=ga[k];TORCH_CHECK(g>=0&&g<G,"g_idx out of range");perm[g*KTILE+cursor[g]++]=k;}
  for(int g=0;g<G;g++)TORCH_CHECK(cursor[g]==KTILE,"each group must contain exactly 128 K entries");
  auto qw=qweight.to(at::kCPU).contiguous();auto qa=qw.accessor<int32_t,2>();
  auto out=at::zeros({K/8,N},at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  auto oa=out.accessor<int32_t,2>();
  for(int kp=0;kp<G;kp++) for(int kl=0;kl<KTILE;kl++) {
    const int dst=kp*KTILE+kl,src=perm[dst];
    for(int n=0;n<N;n++) {
      const uint32_t word=static_cast<uint32_t>(qa[src/8][n]);
      const uint32_t code=(word>>((src&7)*4))&15u;
      const uint32_t old=static_cast<uint32_t>(oa[dst/8][n]);
      oa[dst/8][n]=static_cast<int32_t>(old|(code<<((dst&7)*4)));
    }
  }
  auto p=at::empty({K},at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  auto pa=p.accessor<int32_t,1>();for(int k=0;k<K;k++)pa[k]=perm[k];
  return {out.to(qweight.device()),p.to(qweight.device())};
}

torch::Tensor groupmajor_activation(torch::Tensor x,torch::Tensor perm) {
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.is_contiguous(),"x must be contiguous CUDA FP16");
  TORCH_CHECK(perm.is_cuda()&&perm.scalar_type()==at::kInt&&perm.numel()==x.size(1),"bad perm");
  auto index=perm.to(at::kLong).view({1,-1}).expand({x.size(0),x.size(1)});
  return at::gather(x,1,index).contiguous();
}

torch::Tensor c57_groupmajor_gptq(torch::Tensor x,torch::Tensor qmajor,
    torch::Tensor qzeros,torch::Tensor scales,int group_size,bool return_fp32) {
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.is_contiguous(),"x must be contiguous CUDA FP16");
  TORCH_CHECK(qmajor.is_cuda()&&qmajor.scalar_type()==at::kInt&&qmajor.is_contiguous(),"qmajor must be contiguous CUDA int32");
  TORCH_CHECK(qzeros.is_cuda()&&qzeros.scalar_type()==at::kInt&&qzeros.is_contiguous(),"qzeros must be contiguous CUDA int32");
  TORCH_CHECK(scales.is_cuda()&&scales.scalar_type()==at::kHalf&&scales.is_contiguous(),"scales must be contiguous CUDA FP16");
  const int M=x.size(0),K=x.size(1),N=qmajor.size(1),G=K/group_size;
  TORCH_CHECK(M>=1&&M<=4&&group_size==KTILE&&K%KTILE==0&&N%NTILE==0,"unsupported shape");
  TORCH_CHECK(qmajor.size(0)==K/8,"qmajor K storage shape mismatch");
  TORCH_CHECK(qmajor.size(0)==K/8&&qzeros.size(0)==G&&qzeros.size(1)==N/8&&scales.size(0)==G&&scales.size(1)==N,"metadata shape mismatch");
  auto partial=at::empty({G,M,N},x.options().dtype(at::kFloat));
  auto out=at::empty({M,N},x.options().dtype(return_fp32?at::kFloat:at::kHalf));
  const auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
  dim3 grid(N/NTILE,(M+MROWS-1)/MROWS,G);
  groupmajor_partial<<<grid,NTILE/4,0,stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),reinterpret_cast<const uint32_t*>(qmajor.data_ptr<int32_t>()),reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),partial.data_ptr<float>(),M,N,K,G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(return_fp32) ordered_fp32_sum<float><<<(M*N+255)/256,256,0,stream>>>(partial.data_ptr<float>(),out.data_ptr<float>(),M*N,G);
  else ordered_fp32_sum<half><<<(M*N+255)/256,256,0,stream>>>(partial.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr<at::Half>()),M*N,G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("stable_groupmajor_pack",&stable_groupmajor_pack);m.def("groupmajor_activation",&groupmajor_activation);m.def("c57_groupmajor_gptq",&c57_groupmajor_gptq,py::arg("x"),py::arg("qmajor"),py::arg("qzeros"),py::arg("scales"),py::arg("group_size"),py::arg("return_fp32")=false);}
