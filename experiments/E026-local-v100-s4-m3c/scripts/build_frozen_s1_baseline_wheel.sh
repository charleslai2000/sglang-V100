#!/usr/bin/env bash
# Rebuild the frozen full production extension as an isolated wheel artifact.
# This never installs into site-packages and never touches the editable symlink.
set -euo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration/experiments/E026
BUILD="$ART/frozen-s1-sm70-build"
WHEELS="$ART/frozen-s1-wheelhouse"
DEPS=/mnt/data/pc01/sglang-v100-migration/deps-isolated
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
mkdir -p "$BUILD" "$WHEELS"
cd "$ROOT"
python -m pip wheel --no-deps --no-build-isolation \
  -w "$WHEELS" "$ROOT/sgl-kernel" \
  -Cbuild-dir="$BUILD" \
  -Cbuild.targets=common_ops_sm100_build \
  -Cbuild.tool-args=-j1 \
  -Ccmake.define.SGL_KERNEL_V100_ONLY=ON \
  -Ccmake.define.SGL_KERNEL_COMPILE_THREADS=1 \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=70 \\
  -Ccmake.define.CMAKE_CUDA_FLAGS=-Xptxas=-v \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_CUTLASS="$DEPS/cutlass-turbomind-sglang-v100-57e3cfb47a2d" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FMT="$DEPS/sgl-kernel-repo-fmt" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_TRITON="$DEPS/sgl-kernel-repo-triton" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FLASHINFER="$DEPS/flashinfer-sm70-sglang-v100-bc29697ba20b" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO_FLASH_ATTENTION="$DEPS/sgl-kernel-repo-flash-attention"
WHEEL=$(find "$WHEELS" -maxdepth 1 -name 'sglang_kernel-*.whl' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
test -n "$WHEEL"
python - "$WHEEL" "$ART/frozen-s1-common_ops-sm70.so" <<'PY'
import sys, zipfile
wheel, out=sys.argv[1:]
with zipfile.ZipFile(wheel) as z:
    matches=[n for n in z.namelist() if n.endswith('/sm70/common_ops.abi3.so')]
    if len(matches)!=1: raise SystemExit(f'expected one SM70 common_ops, got {matches}')
    with z.open(matches[0]) as src, open(out,'wb') as dst: dst.write(src.read())
PY
sha256sum "$WHEEL" "$ART/frozen-s1-common_ops-sm70.so"
