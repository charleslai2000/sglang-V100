#include <cuda_fp16.h>
#include <cstdint>

// Isolated SASS probe mirroring the candidate partial kernel's packed Q4 word
// reuse and ordered scalar computation, without touching a production binary.
extern "C" __global__ void packed_word_reuse_probe(
    const half* __restrict__ x, const uint32_t* __restrict__ qw,
    const uint32_t* __restrict__ qz, const half* __restrict__ scales,
    const int32_t* __restrict__ gidx, float* __restrict__ out,
    int M, int N, int K) {
  const int n0 = blockIdx.x * 128 + threadIdx.x * 4;
  const int m0 = blockIdx.y * 4;
  const int part = blockIdx.z;
  float acc[4][4] = {};
  int k0 = part * 128;
  int kend = min(k0 + 128, K);
#pragma unroll 1
  for (int kb = k0; kb < kend; kb += 8) {
    const int row = kb >> 3;
    const uint32_t w0 = qw[row * N + n0];
    const uint32_t w1 = qw[row * N + n0 + 1];
    const uint32_t w2 = qw[row * N + n0 + 2];
    const uint32_t w3 = qw[row * N + n0 + 3];
#pragma unroll
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
          for (int j = 0; j < 4; ++j) acc[mi][j] = fmaf(xv, wt[j], acc[mi][j]);
        }
      }
    }
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
    if (m0 + mi < M)
#pragma unroll
      for (int j = 0; j < 4; ++j) out[(m0 + mi) * N + n0 + j] = acc[mi][j];
}
