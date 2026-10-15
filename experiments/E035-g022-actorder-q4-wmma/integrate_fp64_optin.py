#!/usr/bin/env python3
"""Build an isolated SM70 numerical-fix extension from the measured FP64 source.

This avoids rebuilding the unrelated full sgl-kernel library and never replaces
S1 or the exact-c57 runtime artifact.
"""
import argparse
import hashlib
import os
from pathlib import Path

from torch.utils.cpp_extension import load

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent
source = here / "gptq_candidate_v8_fp64.cu"
registration = a.output.with_suffix(".cpp")
registration.parent.mkdir(parents=True, exist_ok=True)
registration.write_text(r'''#include <torch/extension.h>
#include <torch/library.h>
at::Tensor packed_q4_fp32_numerical_fix(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool);
TORCH_LIBRARY(sgl_kernel, m) {
  m.def("gptq_q4_fp32_m4_numerical_fix(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool return_fp32=False) -> Tensor");
  m.impl("gptq_q4_fp32_m4_numerical_fix", torch::kCUDA, &packed_q4_fp32_numerical_fix);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
''')
build_dir = a.output.parent / "build"
build_dir.mkdir(parents=True, exist_ok=True)
mod = load(
    name="g022_fp64_optin",
    sources=[str(registration), str(source)],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O3", "-lineinfo", "-Xcompiler=-fno-fast-math,-ffp-contract=off", "--fmad=true", "-gencode=arch=compute_70,code=sm_70"],
    build_directory=str(build_dir),
    verbose=True,
)
binary = Path(mod.__file__).resolve()
a.output.parent.mkdir(parents=True, exist_ok=True)
if binary != a.output.resolve():
    import shutil
    shutil.copy2(binary, a.output)
print(f"path={a.output.resolve()}")
print(f"sha256={hashlib.sha256(a.output.read_bytes()).hexdigest()}")
