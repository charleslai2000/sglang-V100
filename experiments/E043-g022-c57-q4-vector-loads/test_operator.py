#!/usr/bin/env python3
import argparse, hashlib, importlib.machinery, json, os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
ROOT=Path('/home/charles/Workspaces/3rdparty/sglang-V100')
FIXROOT=Path('/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures')
C57=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so')
C57_SHA='c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9'
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_fixture(proj):
 d=FIXROOT/('43-mlp-gate_up_proj-call0000' if proj=='gate_up' else '43-mlp-down_proj-call0000'); m=json.loads((d/'manifest.json').read_text())
 def t(n,dtype):
  meta=m['tensors'][n];p=d/f'{n}.bin';assert sha(p)==meta['sha256'];v=torch.from_file(str(p),shared=False,size=Path(p).stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(*meta['shape']).cuda().contiguous();return v
 return d,m,t('x',torch.float16),t('qweight',torch.int32),t('qzeros',torch.int32),t('scales',torch.float16),t('g_idx',torch.int32).view(-1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('projection',choices=['gate_up','down']);ap.add_argument('M',type=int,choices=[1,2,4]);ap.add_argument('--arm',choices=['c57','candidate'],required=True);ap.add_argument('--mode',choices=['check','bench'],default='check');a=ap.parse_args()
 assert sha(C57)==C57_SHA
 assert torch.cuda.device_count()==1
 prop=torch.cuda.get_device_properties(0); actual=str(prop.uuid).lower();assert actual==UUID.lower().removeprefix('gpu-'),actual
 os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='0';os.environ['SGLANG_G017_Q4_FP32_CANDIDATE_PROBE']='0'
 cm=importlib.machinery.ExtensionFileLoader('common_ops',str(C57)).load_module()
 src=ROOT/'experiments/E043-g022-c57-q4-vector-loads/q4_vector.cu'
 cand=load(name='e043_q4_uint4',sources=[str(src)],extra_cflags=['-O2'],extra_cuda_cflags=['-O3','-lineinfo','-Xcompiler=-fno-fast-math,-ffp-contract=off','-gencode=arch=compute_70,code=sm_70'],build_directory='/mnt/data/pc01/sglang-v100-migration/experiments/E043/build',verbose=False)
 d,m,x0,qw,qz,sc,g=load_fixture(a.projection);x=x0[:a.M].contiguous()
 fn=(lambda:torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,sc,g,False)) if a.arm=='c57' else (lambda:cand.q4_uint4_fp16(x,qw,qz,sc,g))
 y=fn();torch.cuda.synchronize(); assert y.dtype==torch.float16 and torch.isfinite(y).all()
 qptr=qw.data_ptr(); outsha=hashlib.sha256(y.cpu().contiguous().numpy().tobytes()).hexdigest()
 result={'event':'E043_UINT4_OPERATOR','arm':a.arm,'projection':a.projection,'M':a.M,'UUID':UUID,'qweight_ptr_mod16':qptr%16,'fixture_manifest_sha256':sha(d/'manifest.json'),'extension_sha256':C57_SHA,'candidate_source_sha256':sha(src),'candidate_binary_sha256':sha(cand.__file__),'output_sha256':outsha,'dtype':str(y.dtype)}
 if a.mode=='check':
  c57=fn() if a.arm=='c57' else torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,qw,qz,sc,g,False);torch.cuda.synchronize()
  result['exact_equal']=bool(torch.equal(y,c57));result['mismatch_count']=int((y!=c57).sum().item());assert result['exact_equal'],result
 else:
  def bench(f,n=30):
   vals=[]
   for _ in range(n):
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();f();e.record();e.synchronize();vals.append(s.elapsed_time(e)*1000)
   return sorted(vals)
  for _ in range(8):fn()
  torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): gout=fn()
  graph.replay();torch.cuda.synchronize()
  result['eager_us']=bench(fn);result['graph_us']=bench(graph.replay);result['eager_median_us']=result['eager_us'][len(result['eager_us'])//2];result['graph_median_us']=result['graph_us'][len(result['graph_us'])//2]
 print(json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
