# G002-N pinned upstream dense WMMA ABI 状态

## 核实对象

仓库 `1CatAI/1Cat-vLLM`，固定 revision `fb38933cf62f1fdb6e99da15b82f93b930fae076`。从该 revision 的 GitHub raw 获取 `flash-attention-v100/kernel/fused_mha_forward.cu`（43,649 bytes），并读取同 revision 的 `kernel/fused_mha_api.cpp`、`include/fused_mha.h` 与 `setup.py`。未修改上游数学 kernel。审计副本 SHA256：dense CU `c87e49f6a042cf6018f0e5f16ac058a2525d2f2ae565f5c7f99b458c11a96167`；API `c55fcf3c14a0ab2febe7bb37adae98b662f4f05528a88e0f471257ae7bc34999`；setup `b926a26015b31bb8a6f1d135b8a1cbb2013f14a32a2965511a6f2cbc047a59a7`；header `9b125c265693c5a2e1e6a36c8563a45196a9ae0dc1ecd19b48c7fa12852586c7`。

## Dense 实现和注册

`fused_mha_forward.cu:59-62` 定义 D512 tile `(BLOCK_M,BLOCK_N,WARPS)=(16,32,16)`；`:189-198` 定义 `flash_attention_forward_kernel`，`:849-900` 是 launcher，`:902-935` 的 `launcher_flash_attention_forward<512>` 选择 dense specialization。Dense public wrapper `flash_attention_forward` 在 `:1043-1132`，`:1119-1122` dispatch 到 `launcher_flash_attention_forward<512>`。

ABI 是 Q `[B,Hq,M,D]`、K/V `[B,Hkv,N,D]`，要求 CUDA FP16、末维 contiguous、Hq 可被 Hkv 整除；D512 受支持。完整 attention 数学已经存在，不需新写 kernel。

同 revision 的 `kernel/fused_mha_api.cpp:18` 已注册 `fwd -> flash_attention_forward`。`setup.py` 的 CUDAExtension source list 已包含 `kernel/fused_mha_forward.cu`。因此 dense binding 不是缺失，也不需要额外注册。按该 fixed API 可用现有扩展 `flash_attn_v100_cuda.fwd`，而非实验用的 paged-only `g002n_wmma_prefill.prefill_paged_fwd`。

## 当前 checkout 的差异

本工作树 `experiments/G002-N/vendor/flash-attention-v100` 是为隔离 paged kernel 而裁剪的子集：只有 `fused_mha_forward_paged.cu`，`prefill_binding.cpp` 仅导出 `prefill_paged_fwd`/兼容别名，`setup_prefill.py` 只编译 paged source。该裁剪 checkout 不包含 pinned dense `fused_mha_forward.cu` 和完整上游 `fused_mha_api.cpp`；不是上游没有 dense 实现/注册。

继续使用裁剪目录的办法会要求复制/引入完整上游 dense translation unit、`fused_mha_api.cpp` 引用的多类其它 kernel declarations/objects、及其构建源依赖；这不是必要的薄 binding。现有 `/data/t011` 内没有完整 pinned dense checkout 或已安装 `flash_attn_v100_cuda` 模块供直接调用。不要再把 paged wrapper 重新解释成 dense ABI。

## 判定

上游源码不存在用户列出的 dense ABI 限制：Dense D512 kernel 与现有 `fwd` 注册均存在。当前阻断具体是本项目 vendored subset 未携带完整 upstream extension，运行环境也未提供对应构建物。后续应由 G002-N 决策确认是否允许把完整固定上游 extension sources/build 作为 vendor/build dependency；获得授权后可直接 build 上游 `fwd` 并接入 pool gather。当前不继续写算法或另起 dense kernel，不改 production 默认 Triton。