#!/usr/bin/env bash
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
PY=/data/t011/venv/sglang/bin/python
CUDA=/data/t011/toolkit/cuda-12.8.1
OUT=/data/t011/build/prefill-gemm
mkdir -p "$OUT"
export CUDA_HOME="$CUDA" CUDACXX="$CUDA/bin/nvcc" CUDAHOSTCXX=/usr/bin/g++-12 PATH="$CUDA/bin:/data/t011/venv/sglang/bin:$PATH"
SRC="$ROOT/experiments/E044-g022-sm70-rebuilt-c57/prefill_gptq_gemm.cu"
REG="$ROOT/experiments/E044-g022-sm70-rebuilt-c57/prefill_gptq_op.cc"
sha256sum "$SRC" "$REG" "$ROOT/experiments/E044-g022-sm70-rebuilt-c57/build_prefill_gemm.sh" | tee "$OUT/source.sha256"
"$PY" - "$SRC" "$REG" "$OUT/t011_prefill_gptq.so" > "$OUT/build.log" 2>&1 <<'PY'
import os,sys,torch
from torch.utils.cpp_extension import load
src,reg,out=sys.argv[1:]
load(name='t011_prefill_gptq',sources=[src,reg],build_directory=os.path.dirname(out),extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-gencode=arch=compute_70,code=sm_70','--threads=1'],with_cuda=True,verbose=True,is_python_module=False)
print('loaded',out)
PY
sha256sum "$OUT/t011_prefill_gptq.so" | tee "$OUT/binary.sha256"
