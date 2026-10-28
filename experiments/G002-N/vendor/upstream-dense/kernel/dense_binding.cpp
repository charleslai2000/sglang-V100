#include <torch/extension.h>
#include <ATen/ATen.h>
#include <optional>
#include <vector>
#include "fused_mha.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &flash_attention_forward,
        "Pinned upstream dense Volta attention; D512 specialization");
}
