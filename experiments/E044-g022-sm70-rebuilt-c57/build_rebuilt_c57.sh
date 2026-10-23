#!/usr/bin/env bash
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
CUDA=/data/t011/toolkit/cuda-12.8.1
PY=/data/t011/venv/sglang/bin/python
OUT=/data/t011/build/rebuilt-c57
mkdir -p "$OUT"
export PATH="$CUDA/bin:/data/t011/venv/sglang/bin:$PATH"
export CUDA_HOME="$CUDA" CUDACXX="$CUDA/bin/nvcc" CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0
SRC=/data/t011/build/rebuilt-c57/source/gptq_candidate_v8.cu
REG="$ROOT/experiments/E044-g022-sm70-rebuilt-c57/rebuilt_c57_op.cc"
sha256sum "$SRC" "$REG" "$ROOT/experiments/E044-g022-sm70-rebuilt-c57/build_rebuilt_c57.sh" | tee "$OUT/source.sha256"
test "$(sha256sum "$SRC" | awk '{print $1}')" = fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1
"$PY" - "$SRC" "$REG" "$OUT/rebuilt_c57_op.so" > "$OUT/build.log" 2>&1 <<'PY'
import os,sys,torch
from torch.utils.cpp_extension import load
src,reg,out=sys.argv[1:]
mod=load(name='rebuilt_c57_op',sources=[src,reg],build_directory=os.path.dirname(out),extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-gencode=arch=compute_70,code=sm_70','--threads=1'],with_cuda=True,verbose=True,is_python_module=False)
print('loaded',mod)
PY
sha256sum "$OUT/rebuilt_c57_op.so" | tee "$OUT/binary.sha256"
