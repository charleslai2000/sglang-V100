#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration
export PYTHONPATH="$ROOT/experiments/E035-g022-actorder-q4-wmma:$ROOT/sgl-kernel/python:$ROOT/python"
export E035_RUNTIME_MODE=fp64
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME="$ART/cache/hf" HF_HUB_CACHE="$ART/cache/hf/hub" TRITON_CACHE_DIR="$ART/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ART/cache/inductor" TMPDIR="$ART/cache/tmp" TMP="$ART/cache/tmp" TEMP="$ART/cache/tmp"
exec python -m sglang.launch_server --model-path /mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4 --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 --mem-fraction-static 0.88 --max-running-requests 4 --max-total-tokens 16384 --cuda-graph-max-bs 4 --cuda-graph-bs 1 2 4 --attention-backend tilelang_fa_v100 --disable-radix-cache --disable-piecewise-cuda-graph --enable-forward-pass-metrics --forward-pass-metrics-ipc-name "ipc://$ART/experiments/E035/fp64-fpm-endpoint"
