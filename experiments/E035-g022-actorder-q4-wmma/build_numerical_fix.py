#!/usr/bin/env python3
import argparse,hashlib,os
from pathlib import Path
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['fp64','dot2'],required=True);a=p.parse_args()
here=Path(__file__).resolve().parent
source=here/f'gptq_candidate_v8_{a.variant}.cu'
name=f'g022_numerical_fix_{a.variant}'
build=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E035')/f'numerical-fix-{a.variant}-build'
build.mkdir(parents=True,exist_ok=True)
mod=load(name=name,sources=[str(here/'numerical_fix_registration.cpp'),str(source)],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','--fmad=true','-gencode=arch=compute_70,code=sm_70'],build_directory=str(build),verbose=True)
path=Path(mod.__file__).resolve();print(path);print(hashlib.sha256(path.read_bytes()).hexdigest())
