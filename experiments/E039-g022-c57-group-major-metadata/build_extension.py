#!/usr/bin/env python3
"""Build E039 isolated SM70 extension; never installs or modifies common_ops."""
import hashlib,json
from pathlib import Path
from torch.utils.cpp_extension import load
src=Path(__file__).with_name('actorder_groupmajor_c57.cu');build=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E039/build');build.mkdir(parents=True,exist_ok=True)
mod=load(name='e039_c57_groupmajor',sources=[str(src)],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory=str(build),verbose=True)
bin=Path(mod.__file__).resolve();print(json.dumps({'binary':str(bin),'binary_sha256':hashlib.sha256(bin.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()},indent=2))
