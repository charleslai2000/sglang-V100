# REBUILT-C57 standalone candidate (not qualified)

This directory builds an isolated Torch custom op from the original G017 c57 CUDA source. It is an independent rebuild, not the unavailable historical c57 binary and must not inherit historical qualification.

- CUDA source: `/data/t011/build/rebuilt-c57/source/gptq_candidate_v8.cu`, SHA256 `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`.
- Standalone extension: `/data/t011/build/rebuilt-c57/rebuilt_c57_op.so`, SHA256 `ab5db27bade8cb868faed6ee4b72c9f7047f838b0ac23c4cdfa4c2b895944d95`. This proves prior build/load/small-matrix smoke only.
- Registration schema is `rebuilt_c57::gptq_q4_fp32_m4_candidate`; CUDA implementation is bound in `rebuilt_c57_op.cc`.
- `gptq_kernels.py` REBUILT-C57 dispatch is exploratory and is not model-qualified. Explicit opt-in is intended to fail closed for unsupported layer/input configurations; it must also validate loaded schema/implementation identity before production use.
- Build and route experiment has not yet passed Falcon dispatch, correctness, Graph, capacity, or performance gates. Do not use as READY evidence.
