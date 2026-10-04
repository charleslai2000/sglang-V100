#!/usr/bin/env bash
# Reproducible host installer for SGLang on NVIDIA V100 (SM70).
# Run this script as a child process; do not source it from an SSH login shell.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[install_v100] Do not source this file; run: bash %q\n' \
    "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail

on_error() {
  local rc=$?
  printf '\n[install_v100] FAILED (exit %d) at line %d: %s\n' \
    "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND" >&2
  printf '[install_v100] Your login shell is still active; fix the error and rerun this script.\n' >&2
  exit "$rc"
}
trap on_error ERR

log() { printf '\n\033[1;34m[install_v100]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[install_v100] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}"
FLASHINFER_REV="c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833"
# Pinned source revisions for the temporarily retained TurboMind component.
TURBOMIND_SOURCE_REV="6ada86ed64af6d1a7b3cb0f34df237fd86f06d48"
TURBOMIND_CUTLASS_REV="da5e086dab31d63815acafdac9a9c5893b1c69e2"

[[ -d "$REPO_ROOT/.git" ]] || die "$REPO_ROOT is not an SGLang-V100 checkout."

if [[ ${EUID} -eq 0 ]]; then
  SUDO=()
else
  command -v sudo >/dev/null || die "sudo is required to install system packages."
  SUDO=(sudo)
fi

log "Installing host compiler and CUDA 12.8 prerequisites"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential ca-certificates cmake curl git g++-12 ninja-build \
  pkg-config wget

if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  CUDA_REPO="ubuntu${VERSION_ID//./}"
  wget -q \
    "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    -O /tmp/cuda-keyring.deb
  "${SUDO[@]}" dpkg -i /tmp/cuda-keyring.deb
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y cuda-toolkit-12-8
fi

if ! command -v conda >/dev/null 2>&1; then
  log "Installing Miniconda"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

CONDA_EXE="$(command -v conda || true)"
[[ -n "$CONDA_EXE" ]] || CONDA_EXE="$HOME/miniconda3/bin/conda"
[[ -x "$CONDA_EXE" ]] || die "conda was not found after installation."
# shellcheck disable=SC1090
. "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx sglang-v100; then
  conda create -y -n sglang-v100 python=3.12 pip
fi
conda activate sglang-v100

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0

# Use every CPU only when RAM can sustain that many compiler processes.  The
# previous unconditional nproc (88 jobs on the reference host, with no swap)
# could invoke the global OOM killer.  This is computed, never hard-coded to 8.
CPU_JOBS="$(nproc)"
MEM_AVAILABLE_KIB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
MEM_JOBS=$(( (MEM_AVAILABLE_KIB - 16 * 1024 * 1024) / (4 * 1024 * 1024) ))
(( MEM_JOBS < 1 )) && MEM_JOBS=1
SAFE_JOBS="$CPU_JOBS"
(( MEM_JOBS < SAFE_JOBS )) && SAFE_JOBS="$MEM_JOBS"
export MAX_JOBS="${MAX_JOBS:-$SAFE_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
log "Build parallelism: MAX_JOBS=$MAX_JOBS (CPU=$CPU_JOBS, RAM-safe=$SAFE_JOBS), NVCC_THREADS=$NVCC_THREADS"

python -m pip install --upgrade pip setuptools wheel scikit-build-core ninja psutil
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  grpcio==1.81.1 grpcio-health-checking==1.81.1 \
  grpcio-reflection==1.81.1 protobuf==6.33.6 tilelang==0.1.8 \
  cuda-tile==1.5.0
python -m pip install -e "$REPO_ROOT/python[diffusion-v100]"

prepare_patched_repo() {
  local name=$1 url=$2 rev=$3 destination=$4
  shift 4
  local backup_destination old_fingerprint patch_file patch_fingerprint stamp_file

  if (( $# > 0 )); then
    patch_fingerprint="$({ sha256sum "$@"; } | sha256sum | awk '{print $1}')"
  else
    patch_fingerprint="$(printf '%s' "$rev" | sha256sum | awk '{print $1}')"
  fi
  stamp_file="$destination/.sglang-v100-patches"

  # An installer-managed checkout is expected to be dirty because patches are
  # applied without creating commits. If those patches change, retain that
  # checkout verbatim and build a clean replacement rather than either
  # rejecting a normal upgrade or discarding possible local edits.
  if [[ -d "$destination/.git" ]] && [[ -f "$stamp_file" ]] && \
      [[ "$(<"$stamp_file")" != "$patch_fingerprint" ]] && \
      ! git -C "$destination" diff --quiet; then
    old_fingerprint="$(<"$stamp_file")"
    backup_destination="${destination}.sglang-v100-backup-${old_fingerprint:0:12}"
    [[ ! -e "$backup_destination" ]] || \
      die "managed dependency backup already exists: $backup_destination"
    log "Preserving previous patched $name checkout at $backup_destination"
    mv "$destination" "$backup_destination"
    stamp_file="$destination/.sglang-v100-patches"
  fi

  if [[ ! -d "$destination/.git" ]]; then
    log "Cloning $name at $rev"
    mkdir -p "$(dirname "$destination")"
    git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$rev"
  elif [[ "$(git -C "$destination" rev-parse HEAD)" != "$rev" ]]; then
    git -C "$destination" diff --quiet && \
      git -C "$destination" diff --cached --quiet || \
      die "$destination has local changes; move it aside or set SGLANG_V100_DEPS_DIR."
    git -C "$destination" fetch origin "$rev"
    git -C "$destination" checkout --detach "$rev"
  fi

  if [[ -f "$stamp_file" ]] && [[ "$(<"$stamp_file")" == "$patch_fingerprint" ]]; then
    return
  fi
  git -C "$destination" diff --quiet && \
    git -C "$destination" diff --cached --quiet || \
    die "$destination has untracked installer patch state; remove it and rerun."

  for patch_file in "$@"; do
    git -C "$destination" apply --check "$patch_file" || \
      die "$name patch does not apply cleanly: $patch_file"
    git -C "$destination" apply "$patch_file"
  done
  printf '%s\n' "$patch_fingerprint" >"$stamp_file"
}

prepare_sparse_repo() {
  local name=$1 url=$2 rev=$3 destination=$4
  shift 4

  if [[ ! -d "$destination/.git" ]]; then
    log "Fetching the attributed $name source subset at $rev"
    mkdir -p "$(dirname "$destination")"
    git clone --filter=blob:none --sparse --no-checkout "$url" "$destination"
  elif [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    die "$destination has local changes; move it aside or set SGLANG_V100_DEPS_DIR."
  fi

  git -C "$destination" sparse-checkout set "$@"
  if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$rev" ]]; then
    git -C "$destination" fetch origin "$rev"
    git -C "$destination" checkout --detach "$rev"
  fi
}

FLASHINFER_DIR="$DEPS_ROOT/flashinfer-sm70"
prepare_patched_repo \
  FlashInfer https://github.com/haohervchb/flashinfer.git \
  "$FLASHINFER_REV" "$FLASHINFER_DIR" \
  "$REPO_ROOT/patches/flashinfer-sm70.patch"
log "Installing the proven FlashInfer SM70 source"
python -m pip uninstall -y flashinfer-python flashinfer-cubin || true
python -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_DIR"

# Attention is implemented in this repository's TileLang package. Uninstall a
# legacy external attention wheel from this environment if an older installer
# put one there; do not delete any user's source checkout.
python -m pip uninstall -y flash-attn-v100 flash_attn_v100 || true

TURBOMIND_SOURCE_DIR="$DEPS_ROOT/turbomind-sm70-source"
prepare_sparse_repo \
  "LMDeploy/1Cat TurboMind" https://github.com/1CatAI/1Cat-vLLM.git \
  "$TURBOMIND_SOURCE_REV" "$TURBOMIND_SOURCE_DIR" \
  LICENSE csrc/core csrc/sm70_turbomind csrc/moe

TURBOMIND_CUTLASS_DIR="$DEPS_ROOT/cutlass-turbomind"
prepare_patched_repo \
  CUTLASS https://github.com/NVIDIA/cutlass.git \
  "$TURBOMIND_CUTLASS_REV" "$TURBOMIND_CUTLASS_DIR"

log "Building the attributed TurboMind SM70 block-FP8 and FP16 MoE backend"
SGLANG_TURBOMIND_SM70_ROOT="$TURBOMIND_SOURCE_DIR" \
SGLANG_TURBOMIND_CUTLASS_ROOT="$TURBOMIND_CUTLASS_DIR" \
  python "$REPO_ROOT/scripts/build_sm70_turbomind.py"

if [[ ! -d "$HOME/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.2.1 \
    https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
fi
export CUTLASS_DIR="$HOME/cutlass"

log "Building lean SM70-only sglang-kernel"
# Remove the newer-GPU wheel first so pip cannot leave an orphaned common_ops
# filename beside the locally built ABI3 module.
python -m pip uninstall -y sglang-kernel || true
# A prior in-place build can leave an ignored CPython .so inside the source
# package. pip then silently bundles it beside the fresh ABI3 SM70 extension.
find "$REPO_ROOT/sgl-kernel/python/sgl_kernel" \
  -type f -name 'common_ops*.so' -delete
python - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    for artifact in (Path(root) / "sgl_kernel").glob("*/common_ops*.so"):
        artifact.unlink()
PY
# The kernel build fetches several pinned upstream repositories. A failed
# `git clone` of any of them aborts the whole wheel. The helper below verifies
# the exact CMake pin before reusing a local checkout or preparing an isolated
# pinned source when that checkout contains unrelated modifications.
kernel_dep_pairs() {
  python3 - "$REPO_ROOT/sgl-kernel/CMakeLists.txt" <<'PY'
import re, sys

text = open(sys.argv[1]).read()
for name, body in re.findall(
    r"FetchContent_Declare\(\s*([\w-]+)(.*?)\)\s*\n", text, re.S
):
    repo = re.search(r"GIT_REPOSITORY\s+\"?([^\s\"]+)", body)
    tag = re.search(r"GIT_TAG\s+([^\s]+)", body)
    if repo and tag:
        print(f"{name.upper()}|{repo.group(1)}|{tag.group(1)}")
PY
}

prepare_kernel_dep() {
  local name=$1 url=$2 rev=$3 destination=$4 attempt checked_out seed
  # Reuse previously prepared pinned checkouts by their explicit name. Never
  # reset or otherwise mutate those shared dependency repositories.
  case "$name" in
    REPO-CUTLASS) seed="$DEPS_ROOT/cutlass-turbomind" ;;
    REPO-FMT) seed="$DEPS_ROOT/sgl-kernel-repo-fmt" ;;
    REPO-TRITON) seed="$DEPS_ROOT/sgl-kernel-repo-triton" ;;
    REPO-FLASHINFER) seed="$DEPS_ROOT/flashinfer-sm70" ;;
    REPO-FLASH-ATTENTION) seed="$DEPS_ROOT/sgl-kernel-repo-flash-attention" ;;
    REPO-MSCCLPP) seed="$DEPS_ROOT/sgl-kernel-repo-mscclpp" ;;
    *) seed="" ;;
  esac
  if [[ -n "$seed" ]] && [[ -d "$seed/.git" ]]; then
    checked_out="$(git -C "$seed" rev-parse "${rev}^{commit}" 2>/dev/null || true)"
    if [[ -n "$checked_out" ]] &&
      git -C "$seed" cat-file -e "${checked_out}^{tree}" 2>/dev/null; then
      destination="$seed"
    fi
  fi
  if [[ -d "$destination/.git" ]]; then
    checked_out="$(git -C "$destination" rev-parse "${rev}^{commit}" 2>/dev/null || true)"
    if [[ -n "$checked_out" ]] &&
      git -C "$destination" cat-file -e "${checked_out}^{tree}" 2>/dev/null; then
      if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$checked_out" ]]; then
        if [[ -n "$(git -C "$destination" status --porcelain 2>/dev/null || true)" ]]; then
          log "Preserving $name worktree changes; preparing an isolated pinned source copy"
          destination="${destination}-sglang-v100-${rev:0:12}"
          local isolated_head
          isolated_head="$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)"
          if [[ -n "$isolated_head" ]]; then
            if [[ -n "$(git -C "$destination" status --porcelain 2>/dev/null || true)" ]]; then
              warn "$name isolated source copy is dirty; refusing to overwrite it"
              return 1
            fi
            checked_out="$(git -C "$destination" rev-parse "${rev}^{commit}" 2>/dev/null || true)"
            if [[ -z "$checked_out" ]] || [[ "$isolated_head" != "$checked_out" ]]; then
              warn "$name isolated checkout is not at required pin $rev"
              return 1
            fi
          else
            git -C "$seed" worktree add --detach --no-checkout "$destination" "$rev" || return 1
            git -C "$destination" checkout --detach "$rev" || return 1
            checked_out="$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)"
          fi
          log "Using isolated pinned $name source at $destination"
        else
          git -C "$destination" checkout --detach "$checked_out" || return 1
        fi
      fi
      log "Reusing $name checkout at $rev from $destination"
      PREPARED_KERNEL_DEP_DIR="$destination"
      return 0
    fi
  fi
  for attempt in 1 2 3 4 5; do
    log "Preparing $name at $rev (attempt $attempt)"
    rm -rf "$destination"
    if git clone --filter=blob:none --no-checkout "$url" "$destination" &&
      git -C "$destination" fetch --depth 1 origin "$rev" &&
      git -C "$destination" checkout --detach FETCH_HEAD; then
      git -C "$destination" submodule update --init --recursive ||
        warn "$name submodule update did not complete; continuing"
      PREPARED_KERNEL_DEP_DIR="$destination"
      return 0
    fi
    sleep 5
  done
  return 1
}

log "Restoring the CUDA 12 NCCL required by torch 2.9.1"
python -m pip uninstall -y nvidia-nccl-cu13 || true
python -m pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.27.5

if [[ "${SGLANG_V100_SKIP_KERNEL_BUILD:-0}" != "1" ]]; then
  log "Building lean SM70-only sglang-kernel wheel"
  python -m pip uninstall -y sglang-kernel || true
  CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=$NVCC_THREADS"
  while IFS='|' read -r kernel_dep_name kernel_dep_url kernel_dep_rev; do
    [[ -n "$kernel_dep_name" ]] || continue
    [[ "$kernel_dep_name" == REPO-MSCCLPP ]] && continue
    kernel_dep_dir="$DEPS_ROOT/sgl-kernel-$(printf '%s' "$kernel_dep_name" | tr '[:upper:]' '[:lower:]')"
    if prepare_kernel_dep "$kernel_dep_name" "$kernel_dep_url" "$kernel_dep_rev" "$kernel_dep_dir"; then
      kernel_cmake_name="$(printf '%s' "$kernel_dep_name" | tr '[:lower:]-' '[:upper:]_')"
      CMAKE_ARGS="${CMAKE_ARGS} -DFETCHCONTENT_SOURCE_DIR_${kernel_cmake_name}=$PREPARED_KERNEL_DEP_DIR"
    else
      die "could not prepare $kernel_dep_name at $kernel_dep_rev"
    fi
  done < <(kernel_dep_pairs)
  export CMAKE_ARGS CUDA_HOME CUDACXX=/usr/local/cuda-12.8/bin/nvcc
  export TORCH_CUDA_ARCH_LIST=7.0
  python -m pip install --no-deps --no-build-isolation \\
    -C "cmake.args=$(printf '%s' "$CMAKE_ARGS" | sed 's/ /;/g')" \\
    "$REPO_ROOT/sgl-kernel"
else
  log "Reusing the previously built SM70-only sglang-kernel wheel"
fi

log "Building V100 Marlin GPTQ/AWQ kernels"
export MARLIN_V100_REPO="${MARLIN_V100_REPO:-$DEPS_ROOT/marlin-v100}"
export MARLIN_V100_REF="${MARLIN_V100_REF:-6d72a49939701d26b15b617a4cd2423174adb2d1}"
# CUDA 12.8 now declares the BF16 vector helpers used by the V100 shim; do not
# apply the older duplicate-declaration compatibility patch on this toolchain.
export MARLIN_V100_SKIP_BF16_COMPAT=1
bash "$REPO_ROOT/scripts/setup_v100_marlin.sh"

log "Running SM70 smoke checks and precompiling first-chat sampling"
if ! SGLANG_V100_FLASHINFER_DIR="$FLASHINFER_DIR" \
  bash "$REPO_ROOT/scripts/smoke_v100.sh"; then
  printf '\n[install_v100] All expensive builds completed successfully.\n' >&2
  printf '[install_v100] Rerun only validation (no rebuild) with:\n' >&2
  printf '  bash %q\n' "$REPO_ROOT/scripts/smoke_v100.sh" >&2
  exit 1
fi

log "Complete. Run: conda activate sglang-v100"
