# REBUILT-C57 standalone candidate (not qualified)

This directory builds an isolated Torch custom op from the original G017 c57 CUDA source. It is an independent rebuild, not the unavailable historical c57 binary and must not inherit historical qualification.

- CUDA source: `/data/t011/build/rebuilt-c57/source/gptq_candidate_v8.cu`, SHA256 `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`.
- Standalone extension: `/data/t011/build/rebuilt-c57/rebuilt_c57_op.so`, SHA256 `ab5db27bade8cb868faed6ee4b72c9f7047f838b0ac23c4cdfa4c2b895944d95`. This proves prior build/load/small-matrix smoke only.
- Registration schema is `rebuilt_c57::gptq_q4_fp32_m4_candidate`; CUDA implementation is bound in `rebuilt_c57_op.cc`.
- `gptq_kernels.py` REBUILT-C57 dispatch is exploratory and is not model-qualified. It checks artifact SHA, schema string, and that a CUDA dispatch-key implementation exists, but worker integration and kernel execution have not passed.
- Current dispatch diff patch SHA256: `2cf308612ba431b56e02731a2ba2a7432851e78216802c840aa40aa84a28a0a6` at `/data/t011/artifacts/rebuilt-c57-dispatch-exploratory.patch`. The patch is uncommitted and not production-qualified.
- Falcon worker/JIT findings and the actual M1/M2/M4 REBUILT-C57 calls following a Prefill fallback are recorded in `worker-jit-findings.md`. These are model-forward smoke only, not oracle/Graph/state/service qualification.
- Build script SHA256 `ba3f00b88d49165382f6e6a91458271fc812d21a20b0489ba0b105d76453fc46`; op registration unit SHA256 `62230be68d3676f3e17b05b409e7bce916b7247eec2d05e12cfa726a30fffbb4`.
- Build/load/small-matrix smoke is not Falcon dispatch, correctness, Graph, capacity, or performance qualification. Do not use as READY evidence.
