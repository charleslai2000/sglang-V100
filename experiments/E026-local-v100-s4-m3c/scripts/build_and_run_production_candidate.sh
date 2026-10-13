#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration
BUILD="$ART/experiments/E026/gptq-kernel-build"
OPS="$ART/experiments/E026/common_ops-sm70.so"
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=7.0
cd "$ROOT/sgl-kernel"
python -m pip install --no-deps --no-build-isolation -e . -Cbuild-dir="$BUILD" \
  -Cbuild.targets=common_ops_sm100_build -Cbuild.tool-args=-j1 \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=70 \
  -Ccmake.define.SGL_KERNEL_V100_ONLY=ON \
  -Ccmake.define.SGL_KERNEL_GPTQ_ONLY=ON \
  -Ccmake.define.SGL_KERNEL_GPTQ_ONLY=ON \
  -Ccmake.define.SGL_KERNEL_COMPILE_THREADS=1 \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_CUTLASS=/mnt/data/pc01/sglang-v100-migration/deps-isolated/cutlass-turbomind-sglang-v100-57e3cfb47a2d \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FMT=/mnt/data/pc01/sglang-v100-migration/deps-isolated/sgl-kernel-repo-fmt \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_TRITON=/mnt/data/pc01/sglang-v100-migration/deps-isolated/sgl-kernel-repo-triton \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FLASHINFER=/mnt/data/pc01/sglang-v100-migration/deps-isolated/flashinfer-sm70-sglang-v100-bc29697ba20b \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FLASH_ATTENTION=/mnt/data/pc01/sglang-v100-migration/deps-isolated/sgl-kernel-repo-flash-attention \
  -Ceditable.rebuild=false -Cinstall.strip=false
cp -f "$BUILD/sm70/common_ops.abi3.so" "$OPS"
sha256sum "$OPS"
