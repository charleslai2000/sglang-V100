#!/usr/bin/env python3
"""Measure full-layer T011 vs exact-c57 QPN2 complete-Linear in real Falcon shapes, FP16 output/bias, eager and graph."""
import argparse,hashlib,importlib.util,json,os,sys,time,statistics
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(ROOT/'python'),str(HERE),str(HERE.parent/'E035-g022-actorder-q4-wmma')]
import importlib.util
from gpu_uuid_lock import assert_cuda_uuid

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['t011','c57'],required=True);p.add_argument('--model',default='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4');p.add_argument('--build',default='/mnt/data/sglang-V100/t011-build');p.add_argument('--iters',type=int,default=50);p.add_argument('--out',required=True);a=p.parse_args()
 if torch.cuda.device_count()!=1:raise RuntimeError('single visible GPU required')
 assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');binary=Path(a.build)/'qpn2_multiwarp.so';tsha=hashlib.sha256(binary.read_bytes()).hexdigest()
 from transformers import AutoConfig
 from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig
 from sglang.srt.layers.quantization.gptq.schemes.gptq_linear import GPTQLinearScheme
 from sglang.srt.model_loader.weight_utils import default_weight_loader
 config=GPTQConfig(weight_bits=4,group_size=128,desc_act=True,lm_head_quantized=False,dynamic={});scheme=GPTQLinearScheme(config);scheme.kernel.use_shuffle=False
 os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='1';os.environ['SGLANG_G017_Q4_FP32_CANDIDATE_PROBE']='0'
 # Set the isolated exact-c57 selector before any import can initialize sgl_kernel.
 if a.arm=='c57':
  os.environ['SGLANG_KERNEL_G017_CANDIDATE_PATH']='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so';os.environ['SGLANG_G017_CANDIDATE_SHA256']='c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9'
 T=AutoConfig.from_pretrained(a.model,trust_remote_code=True,local_files_only=True);H=T.hidden_size;I=T.intermediate_size
 layername='gate_up_proj' if H==3072 else 'down_proj';K,N=(3072,24576) if layername=='gate_up_proj' else (12288,3072);layer=torch.nn.Module();scheme.create_weights(layer,input_size_per_partition=K,output_partition_sizes=[N],input_size=K,params_dtype=torch.float16,weight_loader=default_weight_loader);layer.qweight=torch.nn.Parameter(torch.zeros(K//8,N,device='cuda',dtype=torch.int32),requires_grad=False);layer.qzeros=torch.nn.Parameter(torch.full((K//128,N//8),0x77777777,device='cuda',dtype=torch.int32),requires_grad=False);layer.scales=torch.nn.Parameter(torch.ones(K//128,N,device='cuda',dtype=torch.float16),requires_grad=False);layer.g_idx=torch.nn.Parameter(torch.arange(K,device='cuda',dtype=torch.int32)//128,requires_grad=False)
 # Fill data from the real checkpoint module.
 import struct
 idx=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())['weight_map'];projs=('gate_proj','up_proj') if layername=='gate_up_proj' else ('down_proj',);tensors=[]
 for proj in projs:
  needle='feed_forward.'+proj+'.qweight';name=next(k for k in idx if k.endswith(needle));root=name[:-len('qweight')];shard=idx[name]
  with open(Path(a.model)/shard,'rb') as f:
   hlen=struct.unpack('<Q',f.read(8))[0];f.read(hlen);base=8+hlen;hdr=json.loads((Path(a.model)/shard).read_bytes()[8:base])
   def load(key,dtype):
    m=hdr[key];f.seek(base+m['data_offsets'][0]);return np.frombuffer(f.read(m['data_offsets'][1]-m['data_offsets'][0]),dtype=dtype).copy().reshape(m['shape'])
   tensors.append((load(name,'<i4'),load(root+'qzeros','<i4'),load(root+'scales','<f2'),load(root+'g_idx','<i4')))
 qw=np.concatenate([v[0] for v in tensors],axis=1);qz=np.concatenate([v[1] for v in tensors],axis=1);sc=np.concatenate([v[2] for v in tensors],axis=1);gi=tensors[0][3]
 if any(not np.array_equal(gi,v[3]) for v in tensors[1:]):raise RuntimeError('merged gate/up g_idx permutations differ')
 layer.qweight.data.copy_(torch.from_numpy(qw).cuda());layer.qzeros.data.copy_(torch.from_numpy(qz).cuda());layer.scales.data.copy_(torch.from_numpy(sc).cuda());layer.g_idx.data.copy_(torch.from_numpy(gi).cuda());layer.quant_method=type('QM',(),{'apply':lambda self,l,x,b:scheme.apply_weights(l,x,b)})()
 os.environ['SGLANG_G022_QPN2_BINDING']='1' if a.arm=='t011' else '0';os.environ['SGLANG_G022_QPN2_EXTENSION']=str(binary);os.environ['SGLANG_G022_QPN2_EXTENSION_SHA256']=tsha;os.environ['SGLANG_G017_Q4_FP32_CANDIDATE']='1' if a.arm=='c57' else '0';os.environ['SGLANG_KERNEL_G017_CANDIDATE_PATH']='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so';os.environ['SGLANG_G017_CANDIDATE_SHA256']='c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9'
 # c57 candidate direct invocation, exact loaded extension identity.
 c57='/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so'
 if a.arm=='c57':
  if os.path.realpath(__import__('sgl_kernel').common_ops.__file__)!=os.path.realpath(c57):raise RuntimeError(f'exact c57 loaded path mismatch: {__import__("sgl_kernel").common_ops.__file__}')
  cfn=torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate
 else:cfn=None
 if a.arm=='t011':
  from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import GPTQLinearKernel
  method=GPTQLinearKernel(config);method.use_shuffle=False
  method.process_weights_after_loading(layer)
 else:method=None
 rows=[]
 for M in (1,2,4):
  x=torch.randn((M,K),device='cuda',dtype=torch.float16);bias=torch.randn((N,),device='cuda',dtype=torch.float16);qw=layer.qweight; qz=layer.qzeros;scales=layer.scales;gi=layer.g_idx
  def call():
   if a.arm=='t011':return method.apply(layer,x,bias)
   return (cfn(x,qw,qz,scales,gi,False).to(torch.float16)+bias)
  for mode in ('eager','graph'):
   if mode=='eager':
    for _ in range(10):call()
    torch.cuda.synchronize();times=[];last=None
    for _ in range(a.iters):
     st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();last=call();en.record();en.synchronize();times.append(st.elapsed_time(en)*1000)
   else:
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):out=call()
     graph.replay();torch.cuda.synchronize()
    torch.cuda.current_stream().wait_stream(stream);times=[];last=out
    for _ in range(10):graph.replay()
    torch.cuda.synchronize()
    for _ in range(a.iters):
     st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();graph.replay();en.record();en.synchronize();times.append(st.elapsed_time(en)*1000)
   times.sort();rows.append({'arm':a.arm,'layer':layername,'M':M,'mode':mode,'median_us':statistics.median(times),'p10_us':times[int(.1*(len(times)-1))],'p90_us':times[int(.9*(len(times)-1))],'n':len(times),'output_sha256':hashlib.sha256(last.detach().cpu().numpy().tobytes()).hexdigest(),'candidate_binary_sha256':tsha if a.arm=='t011' else hashlib.sha256(Path(c57).read_bytes()).hexdigest()})
 Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
