#include <torch/library.h>
#include <ATen/core/Tensor.h>

at::Tensor gptq_prefill_fp32_gemm(
    at::Tensor x, at::Tensor qweight, at::Tensor qzeros,
    at::Tensor scales, at::Tensor gidx, double multiplier, bool apply_multiplier,
    bool use_double_fp16);

TORCH_LIBRARY_FRAGMENT(t011_prefill, m) {
  m.def("gptq_fp32_prefill(Tensor x, Tensor qweight, Tensor qzeros, Tensor scales, Tensor g_idx, float multiplier, bool apply_multiplier, bool use_double_fp16=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(t011_prefill, CUDA, m) {
  m.impl("gptq_fp32_prefill", &gptq_prefill_fp32_gemm);
}
