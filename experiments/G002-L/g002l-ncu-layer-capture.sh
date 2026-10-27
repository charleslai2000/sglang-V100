#!/usr/bin/env bash
set -euo pipefail
cd /data/workspace/3rdparty/sglang-V100
D=/data/t011/artifacts/g002m
mkdir -p "$D"
export HOME=/home/charles
export PATH=/data/t011/toolkit/cuda-12.8.1/bin:$PATH
export CUDACXX=/data/t011/toolkit/cuda-12.8.1/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-12 CXX=/usr/bin/g++-12
export CUDA_VISIBLE_DEVICES=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
export CUDA_HOME=/data/t011/toolkit/cuda-12.8.1
export LD_LIBRARY_PATH=/data/t011/toolkit/cuda-12.8.1/lib64:/data/t011/venv/sglang/lib/python3.12/site-packages/torch/lib
export PYTHONPATH=$PWD/python
export SGLANG_SM70_MARLIN_DENSE_LIBRARY=$PWD/python/sglang/jit_kernel/_sm70_marlin_v100_dense.abi3.so
export SM70_MARLIN_DENSE_CTA_GEOMETRY=32x128x32x4x32x32x32 SM70_MARLIN_DENSE_SPLIT_K=1
export SGLANG_GEMMA4_ATTENTION_LAYER_NVTX=1
unset SGLANG_G002M_D512_NUM_WARPS

# NCU 2022.4 profiles only kernels nested in the exact Gemma4 layer NVTX range.
# One 4096-token ModelRunner Forward; NCU replay timings are discarded.
sudo -n --preserve-env=HOME,PATH,CUDACXX,CUDAHOSTCXX,CXX,CUDA_VISIBLE_DEVICES,CUDA_HOME,LD_LIBRARY_PATH,PYTHONPATH,SGLANG_SM70_MARLIN_DENSE_LIBRARY,SM70_MARLIN_DENSE_CTA_GEOMETRY,SM70_MARLIN_DENSE_SPLIT_K,SGLANG_GEMMA4_ATTENTION_LAYER_NVTX \
  /data/t011/toolkit/cuda-12.8.1/bin/ncu \
  --force-overwrite --target-processes all --nvtx --nvtx-include 'G002L_ATTENTION_LAYER:5:D=512/' \
  --set full --profile-from-start off --replay-mode kernel --kernel-name 'regex:_fwd_kernel' --launch-count 1 \
  -o "$D/layer5-full" \
  /data/t011/venv/sglang/bin/python -m sglang.bench_one_batch \
  --model-path /data/t011/models/gemma4-w4a16-ct --dtype float16 --quantization compressed-tensors \
  --trust-remote-code --mem-fraction-static 0.76 --context-length 8192 --max-running-requests 1 \
  --disable-cuda-graph --disable-radix-cache --swa-full-tokens-ratio 0.8 \
  --batch-size 1 --input-len 4096 --output-len 1 --profile-stage prefill \
  --profile --profile-activities CUDA_PROFILER --prefill-diagnostic-id g002m-ncu-layer5 \
  --result-filename "$D/layer5-prefill.jsonl" 2>&1 | tee "$D/layer5-ncu.log"

sudo -n --preserve-env=HOME,PATH,CUDACXX,CUDAHOSTCXX,CXX,CUDA_VISIBLE_DEVICES,CUDA_HOME,LD_LIBRARY_PATH,PYTHONPATH,SGLANG_SM70_MARLIN_DENSE_LIBRARY,SM70_MARLIN_DENSE_CTA_GEOMETRY,SM70_MARLIN_DENSE_SPLIT_K,SGLANG_GEMMA4_ATTENTION_LAYER_NVTX \
  /data/t011/toolkit/cuda-12.8.1/bin/ncu \
  --force-overwrite --target-processes all --nvtx --nvtx-include 'G002L_ATTENTION_LAYER:41:D=512/' \
  --set full --profile-from-start off --replay-mode kernel --kernel-name 'regex:_fwd_kernel' --launch-count 1 \
  -o "$D/layer41-full" \
  /data/t011/venv/sglang/bin/python -m sglang.bench_one_batch \
  --model-path /data/t011/models/gemma4-w4a16-ct --dtype float16 --quantization compressed-tensors \
  --trust-remote-code --mem-fraction-static 0.76 --context-length 8192 --max-running-requests 1 \
  --disable-cuda-graph --disable-radix-cache --swa-full-tokens-ratio 0.8 \
  --batch-size 1 --input-len 4096 --output-len 1 --profile-stage prefill \
  --profile --profile-activities CUDA_PROFILER --prefill-diagnostic-id g002m-ncu-layer41 \
  --result-filename "$D/layer41-prefill.jsonl" 2>&1 | tee "$D/layer41-ncu.log"
