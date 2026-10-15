#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration
export PYTHONPATH="$ROOT/sgl-kernel/python:$ROOT/python"
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
export SGLANG_KERNEL_G017_S1_SHA256=fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME="$ART/cache/hf" HF_HUB_CACHE="$ART/cache/hf/hub"
export TRITON_CACHE_DIR="$ART/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ART/cache/inductor"
export TMPDIR="$ART/cache/tmp" TMP="$ART/cache/tmp" TEMP="$ART/cache/tmp"
export SGLANG_G022_GPTQ_NUMERICAL_FIX=fp64
export SGLANG_KERNEL_G022_NUMERICAL_FIX_PATH="$ART/experiments/E035/numerical-fix-fp64-optin-build/g022_fp64_optin.so"
export SGLANG_G022_NUMERICAL_FIX_SHA256=35998ff295f0a4fe27417886ff2fd86bc9a1c1030975435e8b34aa42fb26b5f6
unset SGLANG_G017_Q4_FP32_CANDIDATE SGLANG_KERNEL_G017_CANDIDATE_PATH SGLANG_G017_CANDIDATE_SHA256
exec "$@"
