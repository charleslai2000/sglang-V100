#include <torch/extension.h>
#include <torch/library.h>

at::Tensor packed_q4_fp32_candidate(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool);
at::Tensor packed_q4_fp32_numerical_fix(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool);
TORCH_LIBRARY(numerical_fix, m) {
  m.def("gptq(Tensor x, Tensor qw, Tensor qz, Tensor scales, Tensor gidx, bool return_fp32=False) -> Tensor");
  m.impl("gptq", torch::kCUDA, &packed_q4_fp32_numerical_fix);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
