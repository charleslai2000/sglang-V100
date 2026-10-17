import sys,importlib.machinery,torch,os
from pathlib import Path
root=Path('/home/charles/Workspaces/3rdparty/sglang-V100');sys.path.insert(0,str(root/'experiments/E043-g022-c57-q4-vector-loads'))
import test_operator as t
proj=sys.argv[1];m=int(sys.argv[2]);arm=sys.argv[3]
_,_,x,qw,qz,sc,g=t.load_fixture(proj);x=x[:m].contiguous()
if arm=='c57':
 mod=importlib.machinery.ExtensionFileLoader('common_ops',str(t.C57)).load_module();fn=lambda:torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,sc,g,False)
else:
 mod=t.load(name='e043_q4_uint4',sources=[str(root/'experiments/E043-g022-c57-q4-vector-loads/q4_vector.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E043/build',verbose=False);fn=lambda:mod.q4_uint4_fp16(x,qw,qz,sc,g)
y=fn();torch.cuda.synchronize();print(f'E043_NCU_READY arm={arm} projection={proj} M={m} qw_mod16={qw.data_ptr()%16} out={y.dtype} shape={tuple(y.shape)}',flush=True)
