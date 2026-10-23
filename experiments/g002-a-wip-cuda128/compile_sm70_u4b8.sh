#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/t011/repos/marlin_v100-g002/source
BUILD=$ROOT/build/temp.linux-x86_64-cpython-312
export CUDA_HOME=/data/t011/toolkit/cuda-12.8.1
export CUTLASS_DIR=/data/t011/deps/cutlass
export PATH=/data/t011/venv/sglang/bin:$CUDA_HOME/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_70,code=sm_70 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_COMPILER=/data/t011/toolkit/cuda-12.8.1/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/data/t011/toolkit/cuda-12.8.1'

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DVLLM_PYTHON_EXECUTABLE=/data/t011/venv/sglang/bin/python \
  -DCMAKE_CUDA_FLAGS='-gencode arch=compute_70,code=sm_70' \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME

ninja -C "$BUILD" -j 1 _C
