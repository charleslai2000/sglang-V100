#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int KTILE = 128;
constexpr int NTILE = 128;
constexpr int MROWS = 4;

// Isolated T009 candidate: the only compute change from exact-c57 is to
// issue one naturally aligned 128-bit Q4 load for the same 4 adjacent words.
__global__ void packed_fp32_partial_uint4(
    const half* __restrict__ x, const uint32_t* __restrict__ qw,
    const uint32_t* __restrict__ qz, const half* __restrict__ scales,
    const int32_t* __restrict__ gidx, float* __restrict__ partial,
    int M, int N, int K) {
  const int n0 = blockIdx.x * NTILE + threadIdx.x * 4;
  const int m0 = blockIdx.y * MROWS;
  const int kp = blockIdx.z;
  if (n0 + 3 >= N || m0 >= M) return;
  float acc[4][4] = {};
  const int k0 = kp * KTILE;
  const int kend = min(k0 + KTILE, K);
#pragma unroll 1
  for (int k = k0; k < kend; ++k) {
    const int g = gidx[k];
    const int shift = (k & 7) * 4;
    const int row = k >> 3;
    const uint32_t* qrow = qw + row * N + n0;
    // n0 is 4-aligned by lane mapping and N is a multiple of 128.
    // Require runtime alignment instead of silently issuing an unsafe vector access.
    if ((reinterpret_cast<uintptr_t>(qrow) & 0xf) != 0) return;
    const uint4 packed = *reinterpret_cast<const uint4*>(qrow);
    const uint32_t w0 = packed.x, w1 = packed.y;
    const uint32_t w2 = packed.z, w3 = packed.w;
    const int q[4] = {int((w0 >> shift) & 15), int((w1 >> shift) & 15),
                      int((w2 >> shift) & 15), int((w3 >> shift) & 15)};
    const uint32_t zw = qz[g * (N / 8) + (n0 >> 3)];
    const int zp[4] = {int((zw >> ((n0 & 7) * 4)) & 15) + 1,
                       int((zw >> (((n0 + 1) & 7) * 4)) & 15) + 1,
                       int((zw >> (((n0 + 2) & 7) * 4)) & 15) + 1,
                       int((zw >> (((n0 + 3) & 7) * 4)) & 15) + 1};
    float wt[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wt[j] = float(q[j] - zp[j]) * __half2float(scales[g * N + n0 + j]);
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
      if (m0 + mi < M) {
        const float xv = __half2float(x[(m0 + mi) * K + k]);
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[mi][j] = fmaf(xv, wt[j], acc[mi][j]);
      }
    }
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    if (m0 + mi < M) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        partial[(kp * M + m0 + mi) * N + n0 + j] = acc[mi][j];
    }
  }
}

template <typename Out>
__global__ void packed_fp32_reduce_uint4(
    const float* __restrict__ partial, Out* __restrict__ out,
    int total, int parts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float sum = 0.f;
#pragma unroll 1
  for (int p = 0; p < parts; ++p) sum += partial[p * total + i];
  if constexpr (std::is_same_v<Out, half>) out[i] = __float2half_rn(sum);
  else out[i] = sum;
}
}

template <typename Out>
at::Tensor run_uint4(at::Tensor x, at::Tensor qw, at::Tensor qz,
                     at::Tensor scales, at::Tensor gidx) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous());
  TORCH_CHECK(qw.is_cuda() && qw.scalar_type() == at::kInt && qw.is_contiguous());
  TORCH_CHECK(qz.is_cuda() && qz.scalar_type() == at::kInt && qz.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kHalf && scales.is_contiguous());
  TORCH_CHECK(gidx.is_cuda() && gidx.scalar_type() == at::kInt && gidx.is_contiguous());
  const int M = x.size(0), K = x.size(1), N = qw.size(1);
  const int parts = (K + KTILE - 1) / KTILE;
  TORCH_CHECK(M >= 1 && M <= MROWS && K % KTILE == 0 && N % NTILE == 0);
  TORCH_CHECK(gidx.numel() == K && qz.size(1) == N / 8 && scales.size(1) == N);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(qw.data_ptr<int32_t>()) & 0xf) == 0,
              "qweight base must be 16-byte aligned");
  auto tmp = at::empty({parts, M, N}, x.options().dtype(at::kFloat));
  auto out = at::empty({M, N}, x.options().dtype(std::is_same_v<Out, half> ? at::kHalf : at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  packed_fp32_partial_uint4<<<dim3(N / NTILE, (M + MROWS - 1) / MROWS, parts), NTILE / 4, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(qz.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      gidx.data_ptr<int32_t>(), tmp.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  packed_fp32_reduce_uint4<Out><<<(M * N + 255) / 256, 256, 0, stream>>>(
      tmp.data_ptr<float>(), reinterpret_cast<Out*>(out.data_ptr()), M * N, parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor q4_uint4_fp16(at::Tensor x, at::Tensor qw, at::Tensor qz,
                          at::Tensor scales, at::Tensor gidx) {
  return run_uint4<half>(x, qw, qz, scales, gidx);
}
at::Tensor q4_uint4_fp32(at::Tensor x, at::Tensor qw, at::Tensor qz,
                          at::Tensor scales, at::Tensor gidx) {
  return run_uint4<float>(x, qw, qz, scales, gidx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("q4_uint4_fp16", &q4_uint4_fp16);
  m.def("q4_uint4_fp32", &q4_uint4_fp32);
}
