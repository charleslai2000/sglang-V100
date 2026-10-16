#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTHONPATH="$ART/experiments/E026/candidate-c57-runtime:$ROOT/python:$ROOT/sgl-kernel/python"
export HF_HOME="$ART/cache/hf" HF_HUB_CACHE="$ART/cache/hf/hub" TRITON_CACHE_DIR="$ART/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ART/cache/inductor" TMPDIR="$ART/cache/tmp" TMP="$ART/cache/tmp" TEMP="$ART/cache/tmp"
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
export SGLANG_G017_Q4_FP32_CANDIDATE=1
export SGLANG_KERNEL_G017_CANDIDATE_PATH="$ART/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so"
export SGLANG_G017_CANDIDATE_SHA256=c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9
exec python -m sglang.launch_server \
  --model-path /mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 \
  --mem-fraction-static 0.80 --max-running-requests 4 --max-total-tokens 16384 \
  --chunked-prefill-size 2048 --mamba-scheduler-strategy extra_buffer --mamba-track-interval 256 --page-size 1 \
  --cuda-graph-max-bs 4 --cuda-graph-bs 1 2 4 --attention-backend tilelang_fa_v100 \
  --enable-forward-pass-metrics --forward-pass-metrics-ipc-name "ipc://$ART/experiments/E036/fpm" \
  --disable-piecewise-cuda-graph --trust-remote-code
