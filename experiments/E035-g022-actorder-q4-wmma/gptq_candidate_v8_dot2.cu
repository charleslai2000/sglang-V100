#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <cmath>

namespace {
constexpr int KTILE = 128;
constexpr int NTILE = 128;
constexpr int MROWS = 4;
#ifndef ACCUM_FP64
#define ACCUM_FP64 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
#error "FP64 accumulation route requires native device double precision"
#endif

__device__ __forceinline__ void two_sum(float a, float b, float& s, float& e) {
  asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(a), "f"(b));
  float bb, s_minus_bb, a_err, b_err;
  asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(bb) : "f"(s), "f"(-a));
  asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(s_minus_bb) : "f"(s), "f"(-bb));
  asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(a_err) : "f"(a), "f"(-s_minus_bb));
  asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(b_err) : "f"(b), "f"(-bb));
  asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(a_err), "f"(b_err));
}
__device__ __forceinline__ void two_prod(float a, float b, float& p, float& e) {
  p = __fmul_rn(a, b);
  e = __fmaf_rn(a, b, -p);
}

__global__ void packed_fp32_partial(
    const half* __restrict__ x, const uint32_t* __restrict__ qw,
    const uint32_t* __restrict__ qz, const half* __restrict__ scales,
    const int32_t* __restrict__ gidx, float* __restrict__ partial,
    int M, int N, int K, int64_t call_id, int do_probe) {
  const int n0 = blockIdx.x * NTILE + threadIdx.x * 4;
  const int m0 = blockIdx.y * MROWS;
  const int kp = blockIdx.z;
  if (n0 + 3 >= N || m0 >= M) return;
#if ACCUM_FP64
  double acc64[4][4] = {};
#else
  float acc_hi[4][4] = {}, acc_lo[4][4] = {};
#endif
  const int k0 = kp * KTILE;
  const int kend = min(k0 + KTILE, K);
  // Q4 stores eight consecutive K values in each packed word. Load the four
  // N words once per K octet, then consume K in the original ascending order.
  // Keep this experiment opt-in; dispatcher/runtime integration is separate.
#pragma unroll 1
  for (int kb = k0; kb < kend; kb += 8) {
    const int row = kb >> 3;
    const uint32_t w0 = qw[row * N + n0];
    const uint32_t w1 = qw[row * N + n0 + 1];
    const uint32_t w2 = qw[row * N + n0 + 2];
    const uint32_t w3 = qw[row * N + n0 + 3];
#pragma unroll 1
    for (int ki = 0; ki < 8; ++ki) {
      const int k = kb + ki;
      const int g = gidx[k];
      const int shift = ki * 4;
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
          for (int j = 0; j < 4; ++j) {
#if ACCUM_FP64
            acc64[mi][j] = fma(double(xv), double(wt[j]), acc64[mi][j]);
#else
            float p, pe, s, se;
            two_prod(xv, wt[j], p, pe);
            two_sum(acc_hi[mi][j], p, s, se);
            acc_lo[mi][j] = acc_lo[mi][j] + (se + pe);
            acc_hi[mi][j] = s;
#endif
          }
        }
      }
    }
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    if (m0 + mi < M) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#if ACCUM_FP64
        partial[(kp * M + m0 + mi) * N + n0 + j] = float(acc64[mi][j]);
#else
        partial[(kp * M + m0 + mi) * N + n0 + j] = acc_hi[mi][j] + acc_lo[mi][j];
#endif
      }
    }
  }
}

template <typename Out>
__global__ void packed_fp32_reduce(
    const float* __restrict__ partial, Out* __restrict__ out,
    int total, int parts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float sum = 0.f, compensation = 0.f;
#pragma unroll 1
  for (int p = 0; p < parts; ++p) {
    const float value = partial[p * total + i];
    const float adjusted = value - compensation;
    const float next = sum + adjusted;
    compensation = (next - sum) - adjusted;
    sum = next;
  }
  if constexpr (std::is_same_v<Out, half>) {
    out[i] = __float2half_rn(sum);
  } else {
    out[i] = sum;
  }
}
}  // namespace

at::Tensor packed_q4_fp32_candidate(
    at::Tensor x, at::Tensor qw, at::Tensor qz,
    at::Tensor scales, at::Tensor gidx, bool return_fp32) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "x must be CUDA FP16");
  TORCH_CHECK(qw.scalar_type() == at::kInt && qz.scalar_type() == at::kInt &&
                  gidx.scalar_type() == at::kInt && scales.scalar_type() == at::kHalf,
              "GPTQ tensors must be int32/FP16");
  TORCH_CHECK(x.dim() == 2 && qw.dim() == 2 && qz.dim() == 2 && scales.dim() == 2 && gidx.dim() == 1,
              "GPTQ tensors must be rank 2/1");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= MROWS &&
                  x.size(1) % 128 == 0 && qw.size(1) % NTILE == 0,
              "candidate supports aligned M in [1,4] geometries");
  const int M = x.size(0), K = x.size(1), N = qw.size(1);
  const int groups = qz.size(0), parts = (K + KTILE - 1) / KTILE;
  TORCH_CHECK(gidx.numel() == K && scales.size(0) == groups && scales.size(1) == N &&
                  qz.size(1) == N / 8,
              "incompatible GPTQ metadata shapes");
  TORCH_CHECK(x.is_contiguous() && qw.is_contiguous() && qz.is_contiguous() &&
                  scales.is_contiguous() && gidx.is_contiguous(),
              "candidate expects contiguous GPTQ tensors");
  auto accum = at::empty({M, N}, x.options().dtype(at::kFloat));
  auto tmp = at::empty({parts, M, N}, x.options().dtype(at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  packed_fp32_partial<<<dim3(N / NTILE, (M + MROWS - 1) / MROWS, parts), NTILE / 4, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(qz.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      gidx.data_ptr<int32_t>(), tmp.data_ptr<float>(), M, N, K, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (return_fp32) {
    packed_fp32_reduce<float><<<(M * N + 255) / 256, 256, 0, stream>>>(
        tmp.data_ptr<float>(), accum.data_ptr<float>(), M * N, parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return accum;
  }
  auto out = at::empty({M, N}, x.options());
  packed_fp32_reduce<half><<<(M * N + 255) / 256, 256, 0, stream>>>(
      tmp.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      M * N, parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
