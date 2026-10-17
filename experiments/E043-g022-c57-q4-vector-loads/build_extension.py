#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
from torch.utils.cpp_extension import load
src=Path(__file__).with_name('q4_vector.cu')
build=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E043/build')
build.mkdir(parents=True,exist_ok=True)
mod=load(name='e043_q4_uint4',sources=[str(src)],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory=str(build),verbose=True)
bin=Path(mod.__file__).resolve()
print(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'binary':str(bin),'binary_sha256':hashlib.sha256(bin.read_bytes()).hexdigest()},sort_keys=True))
