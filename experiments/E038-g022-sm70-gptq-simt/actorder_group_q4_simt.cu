#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/ops/gather.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace {
constexpr int GROUP_K=128;
constexpr int N_TILE=32;
constexpr int K_WARPS=4;
constexpr int PARTS=4;
constexpr int THREADS=32*K_WARPS;

// Block shape [K-warp=4, N-lane=32]. Each warp computes a distinct 32-K
// segment for 32 contiguous output channels. Thus scale/qzero reads are
// contiguous in N; group-major packed Q4 words are shared by each 8 lanes.
__global__ void grouped_simt_partials(
    const half* __restrict__ xperm,
    const uint32_t* __restrict__ qmajor,
    const uint32_t* __restrict__ qzeros,
    const half* __restrict__ scales,
    float* __restrict__ partial,
    int M,int N,int G,int groups_per_part) {
  const int lane=threadIdx.x;
  const int kwarp=threadIdx.y;
  const int n=blockIdx.x*N_TILE+lane;
  const int m=blockIdx.y;
  const int part=blockIdx.z;
  __shared__ float group_partial[K_WARPS][N_TILE];
  float part_acc=0.0f;
  const int g0=part*groups_per_part;
  const int g1=min(G,g0+groups_per_part);
  for(int g=g0;g<g1;g++) {
    float dot=0.0f;
    if(n<N) {
      #pragma unroll 1
      for(int ki=0;ki<GROUP_K/K_WARPS;ki++) {
        const int kg=kwarp*(GROUP_K/K_WARPS)+ki;
        const uint32_t zw=qzeros[g*(N/8)+(n>>3)];
        const int z=int((zw>>((n&7)*4))&15u)+1;
        const uint32_t w=qmajor[(g*GROUP_K+kg)*(N/8)+(n>>3)];
        const int q=int((w>>((n&7)*4))&15u);
        const float xv=__half2float(xperm[m*(G*GROUP_K)+g*GROUP_K+kg]);
        dot=fmaf(xv,float(q-z),dot);
      }
    }
    group_partial[kwarp][lane]=dot;
    __syncthreads();
    if(kwarp==0 && n<N) {
      float group_dot=group_partial[0][lane];
      group_dot+=group_partial[1][lane];
      group_dot+=group_partial[2][lane];
      group_dot+=group_partial[3][lane];
      part_acc=fmaf(group_dot,__half2float(scales[g*N+n]),part_acc);
    }
    __syncthreads();
  }
  if(kwarp==0 && n<N) partial[(part*M+m)*N+n]=part_acc;
}

__global__ void ordered_fp32_reduce(const float* __restrict__ partial,float* __restrict__ out,int total) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total)return;
  float sum=0.0f;
  #pragma unroll
  for(int p=0;p<PARTS;p++) sum+=partial[p*total+i];
  out[i]=sum;
}
}

std::vector<torch::Tensor> group_major_pack(torch::Tensor qweight,torch::Tensor qzeros,torch::Tensor scales,torch::Tensor gidx,int group_size) {
  TORCH_CHECK(qweight.device().is_cuda()&&qweight.scalar_type()==at::kInt&&qweight.is_contiguous(),"qweight must be contiguous CUDA int32");
  TORCH_CHECK(gidx.device().is_cuda()&&gidx.scalar_type()==at::kInt&&gidx.is_contiguous(),"g_idx must be contiguous CUDA int32");
  const int K=gidx.numel(),N=qweight.size(1);
  TORCH_CHECK(group_size==GROUP_K&&K%group_size==0&&N%8==0&&qweight.size(0)==K/8,"unsupported Q4 geometry/group");
  TORCH_CHECK(qzeros.is_cuda()&&qzeros.scalar_type()==at::kInt&&qzeros.is_contiguous()&&qzeros.size(0)==K/group_size&&qzeros.size(1)==N/8,"qzeros must be [G,N/8] int32");
  TORCH_CHECK(scales.is_cuda()&&scales.scalar_type()==at::kHalf&&scales.is_contiguous()&&scales.size(0)==K/group_size&&scales.size(1)==N,"scales must be [G,N] FP16");
  const int G=K/group_size;
  auto qw_cpu=qweight.to(at::kCPU).contiguous();auto gi_cpu=gidx.to(at::kCPU).contiguous();
  auto qa=qw_cpu.accessor<int32_t,2>();auto ga=gi_cpu.accessor<int32_t,1>();
  std::vector<int> cursor(G,0),perm(K);
  for(int k=0;k<K;k++){const int g=ga[k];TORCH_CHECK(g>=0&&g<G,"g_idx out of range");perm[g*group_size+cursor[g]++]=k;}
  for(int g=0;g<G;g++)TORCH_CHECK(cursor[g]==group_size,"each group must contain 128 K entries");
  auto qcpu=at::empty({G,group_size,N/8},at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  auto pcpu=at::empty({K},at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  auto qo=qcpu.accessor<int32_t,3>();auto po=pcpu.accessor<int32_t,1>();
  for(int g=0;g<G;g++)for(int kg=0;kg<group_size;kg++) {
    const int k=perm[g*group_size+kg];po[g*group_size+kg]=k;
    for(int nw=0;nw<N/8;nw++) {
      uint32_t packed=0;
      for(int ni=0;ni<8;ni++) {
        const uint32_t src=static_cast<uint32_t>(qa[k/8][nw*8+ni]);
        packed|=((src>>((k&7)*4))&15u)<<(4*ni);
      }
      qo[g][kg][nw]=static_cast<int32_t>(packed);
    }
  }
  // Group-major row order is canonical group*128+offset. Preserve group-indexed
  // qzeros/scales byte-exact; this is an experimental startup-only transform.
  return {qcpu.to(qweight.device()),pcpu.to(qweight.device()),qzeros.clone(),scales.clone()};
}

torch::Tensor permute_activation_group_major(torch::Tensor x,torch::Tensor perm,int group_size) {
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.is_contiguous(),"x must be contiguous CUDA FP16");
  TORCH_CHECK(group_size==GROUP_K&&perm.is_cuda()&&perm.scalar_type()==at::kInt&&perm.numel()==x.size(1),"bad permutation/group");
  const int M=x.size(0),K=x.size(1);
  TORCH_CHECK(K%group_size==0,"K must be group aligned");
  auto out=at::empty({M,K},x.options());
  auto idx=perm.to(at::kLong).view({1,K}).expand({M,K});
  out.copy_(at::gather(x,1,idx));
  return out;
}

torch::Tensor group_major_fp32(torch::Tensor xperm,torch::Tensor qmajor,torch::Tensor qzeros,torch::Tensor scales,int group_size,int partitions) {
  TORCH_CHECK(xperm.is_cuda()&&xperm.scalar_type()==at::kHalf&&xperm.is_contiguous(),"xperm must be contiguous CUDA FP16");
  TORCH_CHECK(qmajor.is_cuda()&&qmajor.scalar_type()==at::kInt&&qzeros.is_cuda()&&qzeros.scalar_type()==at::kInt&&scales.is_cuda()&&scales.scalar_type()==at::kHalf,"bad GPTQ metadata tensors");
  TORCH_CHECK(group_size==GROUP_K&&partitions==PARTS,"candidate is fixed to group128/four partial partitions");
  const int M=xperm.size(0),K=xperm.size(1),N=scales.size(1),G=K/group_size;
  TORCH_CHECK(K==G*group_size&&qmajor.sizes()==torch::IntArrayRef({G,group_size,N/8})&&qzeros.sizes()==torch::IntArrayRef({G,N/8}),"invalid group-major shapes");
  auto partial=at::empty({PARTS,M,N},xperm.options().dtype(at::kFloat));
  auto out=at::empty({M,N},xperm.options().dtype(at::kFloat));
  const int groups_per_part=(G+PARTS-1)/PARTS;
  dim3 grid((N+N_TILE-1)/N_TILE,M,PARTS),block(N_TILE,K_WARPS);
  const auto stream=at::cuda::getCurrentCUDAStream(xperm.get_device());
  grouped_simt_partials<<<grid,block,0,stream>>>(reinterpret_cast<const half*>(xperm.data_ptr<at::Half>()),reinterpret_cast<const uint32_t*>(qmajor.data_ptr<int32_t>()),reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),partial.data_ptr<float>(),M,N,G,groups_per_part);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ordered_fp32_reduce<<<(M*N+255)/256,256,0,stream>>>(partial.data_ptr<float>(),out.data_ptr<float>(),M*N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("group_major_pack",&group_major_pack);m.def("permute_activation_group_major",&permute_activation_group_major);m.def("group_major_fp32",&group_major_fp32);}
