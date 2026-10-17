#!/usr/bin/env python3
"""Build the isolated E038 CUDA extension; never installs or edits common_ops."""
import hashlib, json, os
from pathlib import Path
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parent
build=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E038/build')
build.mkdir(parents=True,exist_ok=True)
mod=load(name='e038_group_major_simt',sources=[str(root/'actorder_group_q4_simt.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory=str(build),verbose=True)
p=Path(mod.__file__).resolve();print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((root/'actorder_group_q4_simt.cu').read_bytes()).hexdigest(),'cuda':__import__('torch').version.cuda,'torch':__import__('torch').__version__},indent=2))
