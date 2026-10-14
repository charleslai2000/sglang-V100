#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTHONPATH=/hy-tmp/sglang-V100-s2-07343bd165/python
export HF_HOME=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub
export TRITON_CACHE_DIR=/hy-tmp/triton-cache TORCHINDUCTOR_CACHE_DIR=/hy-tmp/torchinductor-cache
export TMPDIR=/hy-tmp/sglang-tmp TMP=/hy-tmp/sglang-tmp TEMP=/hy-tmp/sglang-tmp
LOG=/hy-tmp/sglang-logs/s3-v100-traced-server.log
OUT=${S3_OUT:?must set dedicated S3_OUT}
cd /hy-tmp/sglang-V100-s2-07343bd165
python -m sglang.launch_server --model-path /data/models/Falcon-H1-7B-Instruct-GPTQ-Int4 --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 --mem-fraction-static 0.88 --max-running-requests 4 --max-total-tokens 16384 --cuda-graph-max-bs 4 --cuda-graph-bs 1 2 4 --disable-radix-cache --disable-piecewise-cuda-graph --trust-remote-code >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup(){ kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for i in $(seq 1 720); do
 if curl -fsS http://127.0.0.1:30003/health >/dev/null 2>&1; then break; fi
 if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -80 "$LOG"; exit 2; fi
 sleep 5
done
curl -fsS http://127.0.0.1:30003/health >/dev/null
sleep 15
export S2_BASE=http://127.0.0.1:30003 S3_OUT="$OUT"
python /hy-tmp/sglang-tmp/s3-v100-trace-workload.py
