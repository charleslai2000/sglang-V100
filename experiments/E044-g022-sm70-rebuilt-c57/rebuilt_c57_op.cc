#include <torch/library.h>
#include <ATen/core/Tensor.h>

at::Tensor packed_q4_fp32_candidate(
    at::Tensor x, at::Tensor qw, at::Tensor qz,
    at::Tensor scales, at::Tensor gidx, bool return_fp32);

TORCH_LIBRARY_FRAGMENT(rebuilt_c57, m) {
  m.def("gptq_q4_fp32_m4_candidate(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool return_fp32=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(rebuilt_c57, CUDA, m) {
  m.impl("gptq_q4_fp32_m4_candidate", &packed_q4_fp32_candidate);
}
