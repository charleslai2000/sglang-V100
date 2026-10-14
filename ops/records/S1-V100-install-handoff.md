# S1 V100 installer and smoke handoff

Date: 2026-10-04

## Repository and authority

- Local working checkout: `/home/ubuntu/workspaces/3rdparty/sglang-V100`, branch `milestone/s1-falcon-h1`, baseline `dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3`, `personal` remote points to `charleslai2000/sglang-V100`.
- Runtime/build checkout: `gpushare-v100:/hy-tmp/sglang-V100`, same branch and baseline. NInfer Stage 2 remains paused and untouched.
- Local worktree is intentionally uncommitted: `scripts/build_sm70_turbomind.py`, `scripts/install_v100.sh`, `scripts/setup_v100_marlin.sh`, `patches/marlin-v100-sm70.patch`, `sgl-kernel/CMakeLists.txt`, plus `sgl-kernel/cmake/empty-repo/.keep`. No changes pushed or committed.

## Completed and directly verified

- Host: Tesla V100-SXM2-32GB, SM70; Torch `2.9.1+cu128`, CUDA runtime `12.8`.
- SGLang editable and FlashInfer installed; FlashInfer sampling source resolved from `/hy-tmp/sglang-v100-deps/flashinfer-sm70`.
- Built a lean SM70-only `sglang-kernel` wheel using locally verified pinned FetchContent sources and CMake source-dir settings. Wheel: `sglang_kernel-0.4.3-cp310-abi3-linux_x86_64.whl`, 4,610,430 bytes, SHA256 `60d4415187d45c64ec65a3deaacf7d1471a5ecc6affc46b70c486dd438f4a8be`. Installed extension: `sgl_kernel/sm70/common_ops.abi3.so`.
- Built and loaded attributed TurboMind SM70 extension `_sm70_turbomind_v100.so` (11,937,136 bytes).
- Built Marlin dense and MoE extension shared objects for SM70 using the preserved `/hy-tmp/sglang-v100-deps/marlin-v100` worktree and CUDA 12.8-compatible BF16 header. Marlin's own smoke loaded the extension and registered `torch.ops._moe_C.moe_wna16_marlin_gemm`.
- Direct `bash scripts/smoke_v100.sh` run, after explicitly setting `CUDA_HOME=/usr/local/cuda-12.8`, `CUDACXX=/usr/local/cuda-12.8/bin/nvcc`, and prepending CUDA 12.8 bin to PATH, exited successfully and reported readiness, FlashInfer SM70 sampling, TileLang attention, SM70 kernel, Marlin repack, TurboMind FP8/FP16/AWQ registration, and NCCL 2.27.5.
- One earlier smoke attempt without CUDA overrides selected pip's CUDA 13 `nvcc` and failed with “Unsupported gpu architecture 'sm_70'”. This was an environment selection issue; rerun with the explicit CUDA 12.8 environment passed.

## Preserved shared dependency edits

- Marlin checkout remains uncommitted at original base `6d72a49939701d26b15b617a4cd2423174adb2d1`. Existing tuning edits in `sm70_marlin_gemm.cuh` were preserved; overlapping optional patch was skipped instead of forced. The BF16 compatibility shim needed a targeted fix because CUDA 12.8 already supplies those API functions. Final diff bundle checksum: `/hy-tmp/marlin-v100-uncommitted-final.patch`, SHA256 `23b825b7181ce92e50e35b883cc9a4df93190dbb1668ff3fc48a342b76a769a0`.
- FlashInfer SM70 checkout's four tracked source modifications and `.sglang-v100-patches` untracked stamp were left intact. Final tracked diff bundle checksum: `/hy-tmp/flashinfer-sm70-uncommitted-final.patch`, SHA256 `c5d0986940a4291aef7e199b3bc75cb24f0f425414f74757236bfdc85b3effb3`.

## S1 serving acceptance status

End-to-end Falcon-H1 serving acceptance was subsequently run and recorded in [`S1-V100-GPTQ-serving-acceptance.md`](S1-V100-GPTQ-serving-acceptance.md). That record covers the legacy GPTQ route, numerical investigations, deterministic generation, prefill/decode, concurrency 2, and factual QA at approximately 1K/4K tokens. This installation handoff's build evidence remains valid, but its earlier statement that serving acceptance had not run is superseded.

## Important execution detail

`install_v100.sh` was repeatedly modified while diagnosing stale remote scripts; although direct kernel build and Marlin build passed, an installer retry still failed at the overlapping Marlin patch before the direct follow-up correction. Do not use its current end-to-end installer success as evidence. Run the direct final smoke with the explicit CUDA 12.8 environment above, then continue to model-serving tests. Audit local and server repo status/diffs before any commit or push.
