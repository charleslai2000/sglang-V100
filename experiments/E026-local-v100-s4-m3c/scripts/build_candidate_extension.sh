#!/usr/bin/env bash
# Reproducible isolated SM70 candidate extension build from this worktree.
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)"
ART=/mnt/data/pc01/sglang-v100-migration/experiments/E026
BUILD="$ART/g017-close-build"
WHEELS="$ART/g017-close-wheelhouse"
DEPS=/mnt/data/pc01/sglang-v100-migration/deps-isolated
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
mkdir -p "$BUILD" "$WHEELS"
python -m pip wheel --no-deps --no-build-isolation -w "$WHEELS" "$ROOT/sgl-kernel" \
  -Cbuild-dir="$BUILD" -Cbuild.targets=common_ops_sm100_build -Cbuild.tool-args=-j1 \
  -Ccmake.define.SGL_KERNEL_V100_ONLY=ON \
  -Ccmake.define.SGL_KERNEL_COMPILE_THREADS=1 -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=70 \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_CUTLASS="$DEPS/cutlass-turbomind-sglang-v100-57e3cfb47a2d" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FMT="$DEPS/sgl-kernel-repo-fmt" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_TRITON="$DEPS/sgl-kernel-repo-triton" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FLASHINFER="$DEPS/flashinfer-sm70-sglang-v100-bc29697ba20b" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FLASH_ATTENTION="$DEPS/sgl-kernel-repo-flash-attention"
WHEEL=$(find "$WHEELS" -maxdepth 1 -name 'sglang_kernel-*.whl' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
python - "$WHEEL" "$ART/g017-close-final-sm70.so" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    matches = [n for n in z.namelist() if n.endswith('/sm70/common_ops.abi3.so')]
    if len(matches) != 1:
        raise SystemExit(f'expected one SM70 common_ops, got {matches}')
    with z.open(matches[0]) as src, open(sys.argv[2], 'wb') as dst:
        dst.write(src.read())
PY
sha256sum "$WHEEL" "$ART/g017-close-final-sm70.so"
printf 'source base=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf 'source tree sha256 manifest follows\n'
find "$ROOT/python/sglang/srt/hardware_backend/gpu/quantization" "$ROOT/sgl-kernel/csrc/gemm/gptq" "$ROOT/sgl-kernel/python/sgl_kernel" -maxdepth 2 -type f \( -name 'gptq_kernels.py' -o -name 'gptq_candidate_v8.cu' -o -name 'load_utils.py' -o -name '__init__.py' \) -print0 | sort -z | xargs -0 sha256sum
nvcc --version | tail -4
g++-12 --version | head -1
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "abi", torch._C._GLIBCXX_USE_CXX11_ABI)'
