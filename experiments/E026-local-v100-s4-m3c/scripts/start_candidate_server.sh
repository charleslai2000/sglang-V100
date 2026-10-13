#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
# Prefer the isolated candidate-only sgl_kernel package so loader cannot pick S1.
# Keep the source tree after it for SGLang Python imports.
export PYTHONPATH="$ART/experiments/E026/candidate-runtime:$ROOT/python"
export HF_HOME="$ART/cache/hf" HF_HUB_CACHE="$ART/cache/hf/hub"
if [[ "${SGLANG_ENABLE_METRICS_DEVICE_TIMER:-0}" == "1" ]]; then
  export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
  FPM_FLAG=--enable-forward-pass-metrics
else
  unset SGLANG_ENABLE_METRICS_DEVICE_TIMER
  FPM_FLAG=
fi
export TRITON_CACHE_DIR="$ART/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ART/cache/inductor"
export TMPDIR="$ART/cache/tmp" TMP="$ART/cache/tmp" TEMP="$ART/cache/tmp"
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
MODE=${G017_MODE:-candidate}
if [[ "$MODE" == candidate ]]; then
  export SGLANG_G017_Q4_FP32_CANDIDATE=1
  export SGL_KERNEL_G017_CANDIDATE_PATH="$ART/experiments/E026/candidate-runtime/sgl_kernel/sm70/common_ops.abi3.so"
  export SGLANG_G017_CANDIDATE_SHA256=02f923ea7329912a4413f179f292acb444a5e001caebf110b24f531cabf70d8b
  export G017_EXPECTED_CANDIDATE_PROBE=${G017_EXPECTED_CANDIDATE_PROBE:-1}
  export SGLANG_G017_MEASURED_EXTENSION="$SGL_KERNEL_G017_CANDIDATE_PATH"
  export SGLANG_G017_Q4_FP32_CANDIDATE_PROBE=${SGLANG_G017_Q4_FP32_CANDIDATE_PROBE:-0}
elif [[ "$MODE" == s1 ]]; then
  unset SGLANG_G017_Q4_FP32_CANDIDATE SGL_KERNEL_G017_CANDIDATE_PATH SGLANG_G017_CANDIDATE_SHA256 SGLANG_G017_Q4_FP32_CANDIDATE_PROBE
  export PYTHONPATH="$ART/experiments/E026/frozen-s1-runtime:$ROOT/python"
  export SGLANG_G017_MEASURED_EXTENSION="$ART/experiments/E026/frozen-s1-runtime/sgl_kernel/sm70/common_ops.abi3.so"
else
  echo "G017_MODE must be candidate or s1" >&2; exit 2
fi
if [[ "${SGLANG_G017_Q4_FP32_CANDIDATE_PROBE:-0}" == "1" && "$MODE" == "candidate" ]]; then
  export G017_EXPECTED_CANDIDATE_PROBE=1
else
  unset G017_EXPECTED_CANDIDATE_PROBE
fi
python -m sglang.launch_server \
  --model-path /mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 \
  --mem-fraction-static 0.88 --max-running-requests 4 --max-total-tokens 16384 \
  --cuda-graph-max-bs 4 --cuda-graph-bs 1 2 4 \
  --attention-backend tilelang_fa_v100 --disable-radix-cache --disable-piecewise-cuda-graph $FPM_FLAG --trust-remote-code
