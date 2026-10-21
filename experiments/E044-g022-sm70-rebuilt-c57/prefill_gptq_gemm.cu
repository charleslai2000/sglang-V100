#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace {
__global__ void reconstruct_q4_fp32_kernel(
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const half* __restrict__ scales,
    const int32_t* __restrict__ gidx,
    float* __restrict__ weight,
    int K, int N) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (k >= K || n >= N) return;
  const uint32_t packed_w = reinterpret_cast<const uint32_t*>(qweight)[(k >> 3) * N + n];
  const int q = (packed_w >> ((k & 7) * 4)) & 0xf;
  const int g = gidx[k];
  const uint32_t packed_z = reinterpret_cast<const uint32_t*>(qzeros)[g * (N >> 3) + (n >> 3)];
  const int z = ((packed_z >> ((n & 7) * 4)) & 0xf) + 1;
  const float scale = __half2float(scales[g * N + n]);
  weight[k * N + n] = static_cast<float>(q - z) * scale;
}
}

at::Tensor gptq_prefill_fp32_gemm(
    at::Tensor x, at::Tensor qweight, at::Tensor qzeros,
    at::Tensor scales, at::Tensor gidx, double multiplier, bool apply_multiplier) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2,
              "x must be CUDA FP16 [M,K]");
  TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == at::kInt && qweight.dim() == 2,
              "qweight must be CUDA int32 [K/8,N]");
  TORCH_CHECK(qzeros.is_cuda() && qzeros.scalar_type() == at::kInt && qzeros.dim() == 2,
              "qzeros must be CUDA int32 [G,N/8]");
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kHalf && scales.dim() == 2,
              "scales must be CUDA FP16 [G,N]");
  TORCH_CHECK(gidx.is_cuda() && gidx.scalar_type() == at::kInt && gidx.dim() == 1,
              "g_idx must be CUDA int32 [K]");
  const int M=x.size(0), K=x.size(1), N=qweight.size(1), G=scales.size(0);
  TORCH_CHECK(M > 0 && M <= 4096 && K == gidx.numel() && K == qweight.size(0)*8 &&
              N % 8 == 0 && scales.size(1)==N && qzeros.size(0)==G && qzeros.size(1)==N/8 && G==K/128,
              "incompatible GPTQ geometry");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && qzeros.is_contiguous() &&
              scales.is_contiguous() && gidx.is_contiguous(), "inputs must be contiguous");
  c10::cuda::CUDAGuard guard(x.device());
  auto weight=at::empty({K,N},x.options().dtype(at::kFloat));
  auto activation=x.to(at::kFloat);
  auto out=at::empty({M,N},x.options().dtype(at::kFloat));
  dim3 block(32,8), grid((N+31)/32,(K+7)/8);
  reconstruct_q4_fp32_kernel<<<grid,block,0,at::cuda::getCurrentCUDAStream()>>>(
      qweight.data_ptr<int32_t>(),qzeros.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),gidx.data_ptr<int32_t>(),
      weight.data_ptr<float>(),K,N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cublasHandle_t handle=at::cuda::getCurrentCUDABlasHandle();
  const float alpha=apply_multiplier ? static_cast<float>(multiplier) : 1.0f;
  const float beta=0.0f;
  // Row-major x[M,K], w[K,N], out[M,N] via column-major reinterpretation.
  // Explicit FP32 inputs/output/compute; no global handle math-mode mutation.
  const cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  cublasStatus_t status=cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
      N,M,K,&alpha,weight.data_ptr<float>(),CUDA_R_32F,N,
      activation.data_ptr<float>(),CUDA_R_32F,K,&beta,out.data_ptr<float>(),CUDA_R_32F,N,
      CUBLAS_COMPUTE_32F,algo);
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"cublasGemmEx FP32 GPTQ Prefill failed: ",static_cast<int>(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
