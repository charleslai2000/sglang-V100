#!/usr/bin/env bash
# Explicit opt-in G017 candidate launcher. Without G017_MODE=candidate, runs frozen S1.
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)"
ART=/mnt/data/pc01/sglang-v100-migration
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME="$ART/cache/hf" HF_HUB_CACHE="$ART/cache/hf/hub"
export TRITON_CACHE_DIR="$ART/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ART/cache/inductor"
export TMPDIR="$ART/cache/tmp" TMP="$ART/cache/tmp" TEMP="$ART/cache/tmp"
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
MODE=${G017_MODE:-s1}
if [[ "${SGLANG_ENABLE_METRICS_DEVICE_TIMER:-0}" == "1" ]]; then
  export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1; FPM_FLAG=--enable-forward-pass-metrics
else
  unset SGLANG_ENABLE_METRICS_DEVICE_TIMER; FPM_FLAG=
fi
if [[ "$MODE" == candidate ]]; then
  export SGLANG_G017_Q4_FP32_CANDIDATE=1
  export SGLANG_KERNEL_G017_CANDIDATE_PATH="$ART/experiments/E026/g017-close-final-sm70.so"
  unset SGLANG_KERNEL_G017_S1_PATH SGLANG_KERNEL_G017_S1_SHA256
  export SGLANG_G017_CANDIDATE_SHA256=c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9
  export SGLANG_G017_CANDIDATE_PROBE=${SGLANG_G017_CANDIDATE_PROBE:-0}
  export SGLANG_G017_MEASURED_EXTENSION="$SGLANG_KERNEL_G017_CANDIDATE_PATH"
  export PYTHONPATH="/tmp/g017-ep-site:$ROOT/sgl-kernel/python:$ROOT/python"
elif [[ "$MODE" == s1 ]]; then
  unset SGLANG_G017_Q4_FP32_CANDIDATE SGLANG_KERNEL_G017_CANDIDATE_PATH SGLANG_G017_CANDIDATE_SHA256 SGLANG_G017_CANDIDATE_PROBE SGLANG_KERNEL_G017_S1_PATH
  export SGLANG_KERNEL_G017_S1_SHA256=fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842
  export SGLANG_G017_MEASURED_EXTENSION="$ART/experiments/E026/frozen-s1-runtime/sgl_kernel/sm70/common_ops.abi3.so"
    export PYTHONPATH="/tmp/g017-ep-site:$ROOT/sgl-kernel/python:$ROOT/python"
else
  echo "G017_MODE must be s1 or candidate (got: $MODE)" >&2; exit 2
fi
python -m sglang.launch_server \
  --model-path /mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 \
  --mem-fraction-static 0.88 --max-running-requests 4 --max-total-tokens 16384 \
  --cuda-graph-max-bs 4 --cuda-graph-bs 1 2 4 \
  --attention-backend tilelang_fa_v100 --disable-radix-cache --disable-piecewise-cuda-graph $FPM_FLAG --trust-remote-code
