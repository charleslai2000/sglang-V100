from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent
ext = CUDAExtension(
    name="g002n_wmma_prefill",
    sources=[str(root / "kernel/prefill_binding.cpp"), str(root / "kernel/prefill_paged_sm70.cu")],
    include_dirs=[str(root / "kernel")],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17", "-gencode=arch=compute_70,code=sm_70", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
    },
)
setup(name="g002n-wmma-prefill", ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
