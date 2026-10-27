#!/usr/bin/env bash
set -euo pipefail
cd /data/workspace/3rdparty/sglang-V100
D=/data/t011/artifacts/g002m
mkdir -p "$D"
export HOME=/home/charles PATH=/data/t011/toolkit/cuda-12.8.1/bin:$PATH CUDACXX=/data/t011/toolkit/cuda-12.8.1/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-12 CXX=/usr/bin/g++-12 CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189 CUDA_HOME=/data/t011/toolkit/cuda-12.8.1 LD_LIBRARY_PATH=/data/t011/toolkit/cuda-12.8.1/lib64:/data/t011/venv/sglang/lib/python3.12/site-packages/torch/lib PYTHONPATH=$PWD/python SGLANG_SM70_MARLIN_DENSE_LIBRARY=$PWD/python/sglang/jit_kernel/_sm70_marlin_v100_dense.abi3.so SM70_MARLIN_DENSE_CTA_GEOMETRY=32x128x32x4x32x32x32 SM70_MARLIN_DENSE_SPLIT_K=1

for length in 1024 4096; do
  for variant in base w4; do
    if [[ "$variant" == w4 ]]; then export SGLANG_G002M_D512_NUM_WARPS=4; else unset SGLANG_G002M_D512_NUM_WARPS || true; fi
    reps=(1 2 3)
    if [[ "$length" == 1024 && "$variant" == base ]]; then reps=(1 2 3); fi
    if [[ "$length" == 4096 && "$variant" == base ]]; then reps=(1 2 3); fi
    for rep in "${reps[@]}"; do
      host_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
      (( host_kb >= 8388608 )) || { echo "host memory floor violated: $host_kb kB" >&2; exit 2; }
      gpu_line=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.free --format=csv,noheader)
      [[ "$gpu_line" == "$CUDA_VISIBLE_DEVICES,"* ]] || { echo "unexpected GPU: $gpu_line" >&2; exit 2; }
      echo "RUN length=$length variant=$variant rep=$rep host_kB=$host_kb gpu='$gpu_line'"
      export SGLANG_G002M_INPUT_SEED=$((20261027 + length * 10 + rep))
      /data/t011/venv/sglang/bin/python -m sglang.bench_one_batch \
        --model-path /data/t011/models/gemma4-w4a16-ct --dtype float16 --quantization compressed-tensors \
        --trust-remote-code --mem-fraction-static 0.76 --context-length 8192 --max-running-requests 1 \
        --disable-cuda-graph --disable-radix-cache --swa-full-tokens-ratio 0.8 \
        --batch-size 1 --input-len "$length" --output-len 1 --profile-stage prefill \
        --prefill-diagnostic-id "g002m-${variant}-${length}-r${rep}" \
        --prefill-diagnostic-layer-ids 5,11,17,23,29,35,41 \
        --prefill-diagnostic-output-file "$D/${variant}-${length}-r${rep}.pt" \
        --run-name "g002m-${variant}-${length}-r${rep}" \
        --result-filename "$D/${variant}-${length}-r${rep}.jsonl" \
        2>&1 | tee "$D/${variant}-${length}-r${rep}.log"
    done
  done
done
