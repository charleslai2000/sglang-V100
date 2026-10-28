# G002-N 上游 WMMA 源码来源

## 固定 Revision

- 仓库：`https://github.com/1CatAI/1Cat-vLLM.git`
- WMMA paged prefill 基础实现：`92a785cb21e2b2acbd4be1dfd87e799c9c632a37` (`Release 1Cat-vLLM 0.0.3`)
- D512 specialization patch：`47b5bbc18f63b2778a46ab3891c92be8e25f6fdf` (`[gemma4][FA] head_dim-512 prefill (split via small blocks) -> fully-FA`)
- D512 dense/paged forward kernel families及 traits/config 以 `47b5bbc...` 为最终源码版本；实验不复制整仓或 vLLM framework
- PR #1271 HEAD：`fb38933cf62f1fdb6e99da15b82f93b930fae076` (`[Lint] Fix ruff violations in Gemma-4 and SM70 files`)
- E2B/E4B KV-sharing ownership 修正所在 PR #59 commit：`68d48fc80ed6a4731bde664d469a4b5a24127ef8` (`[FA_V100] Run gemma E2B/E4B KV-shared layers on FA (read target cache)`)
- 来源代码许可：BSD-3-Clause；保留上游 `flash-attention-v100/LICENSE` 与 SPDX/copyright 标识。

## 实验范围

只引入 paged FP16 D512 attention 所需的 WMMA 实现、Traits/header 和最小 PyTorch CUDA binding；不得引入 vLLM backend、model registry、服务路由、decode/backward 或 FP8 实现。保留 D512 参数：`BLOCK_M=16`、`BLOCK_N=32`、`WARPS=16`。生产默认路径不变，候选必须显式调用实验 binding。

Paged 上游 ABI 的 Q 为 `[B,H,M,D]`、KV cache 为 `[pages,page_size,Hkv,D]`、block table 为物理页索引。Gemma4 E4B Full Attention 是 Hq=8/Hkv=2/D=512，scale=1，FP16；共享 KV 层必须读 owner layer 的 KV cache，而不能使用 shared layer 传入的空 K/V。
