#include <torch/extension.h>
#include <ATen/ATen.h>
#include <optional>
#include <string>

at::Tensor flash_attention_prefill_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out, const at::Tensor& block_table,
    const at::Tensor& seq_lens, const float softmax_scale,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const bool is_causal, const int window);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill_paged_fwd", &flash_attention_prefill_paged,
          "Isolated SM70 paged WMMA prefill (upstream BSD-3-Clause)");
    m.def("fwd", &flash_attention_prefill_paged,
          "Compatibility alias for the isolated paged WMMA prefill");
}
