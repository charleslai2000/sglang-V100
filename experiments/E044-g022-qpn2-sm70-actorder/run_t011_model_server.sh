#!/usr/bin/env bash
# Isolated T011 opt-in server wrapper. Defaults/S1/c57 are unchanged.
set -euo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration
mkdir -p "$ART/experiments/E044" "$ART/cache/tmp"
source /home/charles/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export PYTHONPATH="$ROOT/experiments/E044-g022-qpn2-sm70-actorder:$ROOT/experiments/E035-g022-actorder-q4-wmma:$ROOT/sgl-kernel/python:$ROOT/python"
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
export CUDA_HOME=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc CXX=/usr/bin/g++ MAX_JOBS=1 NINJA_STATUS='[%f/%t] '
export HF_HOME="$ART/cache/hf" HF_HUB_CACHE="$ART/cache/hf/hub" TRITON_CACHE_DIR="$ART/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ART/cache/inductor" TMPDIR="$ART/cache/tmp" TMP="$ART/cache/tmp" TEMP="$ART/cache/tmp"
export SGLANG_G017_CANDIDATE_SHA256=c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9
export SGLANG_KERNEL_G017_CANDIDATE_PATH=/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so
if [[ "${T011_ROUTE:-t011}" == "c57" ]]; then
  export E035_RUNTIME_MODE=c57
  unset SGLANG_G022_QPN2_BINDING
  export SGLANG_G017_Q4_FP32_CANDIDATE=1
else
  export E035_RUNTIME_MODE=fp64
  unset SGLANG_G017_Q4_FP32_CANDIDATE
  export SGLANG_G022_QPN2_BINDING=1
fi
export SGLANG_G022_QPN2_BUILD=/mnt/data/sglang-V100/t011-build
export SGLANG_G021_FPM_SIDECAR=${T011_TIMELINE_SIDECAR:-}
export SGLANG_G022_FPM_GPU_SIDECAR=${T011_GPU_SIDECAR:-"$ROOT/experiments/E044-g022-qpn2-sm70-actorder/results/gpu-${T011_ROUTE:-t011}-matched.jsonl"}
export SGLANG_G021_FPM_DIAG=1
EXT_INFO=$(NINJA_STATUS='' PYTHONPATH="$ROOT/experiments/E044-g022-qpn2-sm70-actorder:$ROOT/experiments/E035-g022-actorder-q4-wmma" /home/charles/miniconda3/envs/sglang-v100/bin/python -c 'import hashlib; from qpn2_multiwarp import load_module; _,b=load_module("/mnt/data/sglang-V100/t011-build"); print(str(b)+" "+hashlib.sha256(b.read_bytes()).hexdigest())' 2>/dev/null | tail -1)
export SGLANG_G022_QPN2_EXTENSION=${EXT_INFO% *}
export SGLANG_G022_QPN2_EXTENSION_SHA256=${EXT_INFO##* }
printf 'T011 extension %s sha256=%s\n' "$SGLANG_G022_QPN2_EXTENSION" "$SGLANG_G022_QPN2_EXTENSION_SHA256"
exec python -m sglang.launch_server --model-path /mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4 --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 --mem-fraction-static "${T011_MEM_FRACTION:-0.88}" --max-running-requests 4 --max-total-tokens "${T011_MAX_TOTAL_TOKENS:-16384}" --cuda-graph-max-bs 4 --cuda-graph-bs 1 2 3 4 --attention-backend tilelang_fa_v100 --disable-radix-cache --disable-piecewise-cuda-graph --enable-forward-pass-metrics --forward-pass-metrics-ipc-name "ipc://$ART/experiments/E044/t011-fpm-endpoint"
