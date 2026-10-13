#!/usr/bin/env bash
# Host-only clean-source runtime shim for this host's pre-existing editable finder.
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)"
ART=/mnt/data/pc01/sglang-v100-migration
export PYTHONPATH="/tmp/g017-ep-site:$ROOT/sgl-kernel/python:$ROOT/python"
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME="$ART/cache/hf" HF_HUB_CACHE="$ART/cache/hf/hub"
export TRITON_CACHE_DIR="$ART/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ART/cache/inductor"
export TMPDIR="$ART/cache/tmp" TMP="$ART/cache/tmp" TEMP="$ART/cache/tmp"
if [[ "${G017_MODE:-s1}" == candidate ]]; then
  unset SGLANG_KERNEL_G017_S1_PATH
  export SGLANG_G017_Q4_FP32_CANDIDATE=1
  export SGLANG_KERNEL_G017_CANDIDATE_PATH="$ART/experiments/E026/candidate-runtime/sgl_kernel/sm70/common_ops.abi3.so"
  export SGLANG_G017_CANDIDATE_SHA256=02f923ea7329912a4413f179f292acb444a5e001caebf110b24f531cabf70d8b
  export G017_EXPECTED_CANDIDATE_PROBE=0
elif [[ "${G017_MODE:-s1}" == s1 ]]; then
  unset SGLANG_G017_Q4_FP32_CANDIDATE SGL_KERNEL_G017_CANDIDATE_PATH SGLANG_KERNEL_G017_CANDIDATE_PATH SGLANG_G017_CANDIDATE_SHA256 G017_EXPECTED_CANDIDATE_PROBE
  export SGLANG_KERNEL_G017_S1_SHA256=fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842
else
  echo "G017_MODE must be s1 or candidate" >&2; exit 2
fi
if [[ "${G017_SERVER:-0}" == 1 ]]; then
  exec python -m sglang.launch_server --model-path /mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4 \
    --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 --host 127.0.0.1 --port 30003 \
    --tp-size 1 --dtype float16 --mem-fraction-static 0.88 --max-running-requests 4 \
    --max-total-tokens 16384 --cuda-graph-max-bs 4 --cuda-graph-bs 1 2 4 \
    --attention-backend tilelang_fa_v100 --disable-radix-cache --disable-piecewise-cuda-graph --trust-remote-code
else
  exec "$@"
fi
