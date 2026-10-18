#!/usr/bin/env python3
"""Collect SGLang ForwardBatch MLP inputs at graph M4; compare each captured gate/down layer to QPN2 strict oracle."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(ROOT/'python'),str(HERE),str(HERE.parent/'E035-g022-actorder-q4-wmma')]
from qpn2_multiwarp import load_module
from gpu_uuid_lock import assert_cuda_uuid

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4');p.add_argument('--build',default='/mnt/data/sglang-V100/t011-build');p.add_argument('--out',default=str(HERE/'results/t011-fullmodel-dispatch.json'));a=p.parse_args()
 if torch.cuda.device_count()!=1:raise RuntimeError('only target GPU must be visible')
 assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');_,binary=load_module(a.build);sha=hashlib.sha256(binary.read_bytes()).hexdigest();os.environ['SGLANG_G022_QPN2_BINDING']='1';os.environ['SGLANG_G022_QPN2_EXTENSION']=str(binary);os.environ['SGLANG_G022_QPN2_EXTENSION_SHA256']=sha
 from sglang.srt.server_args import ServerArgs,get_global_server_args
 from sglang.srt.configs.model_config import ModelConfig
 from sglang.srt.model_loader.loader import get_model_loader
 from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import GPTQLinearKernel
 ServerArgs(model_path=a.model,trust_remote_code=True,device='cuda',quantization='gptq',dtype='float16');server=get_global_server_args();mc=ModelConfig(a.model,trust_remote_code=True,model_override_args=None,revision=None,context_length=8192,dtype='float16',quantization='gptq');model=get_model_loader(server).load_model(model_config=mc,device_config=server.device_config)
 results=[]
 def hook(layer,x):
  if not isinstance(getattr(layer,'quant_method',None),GPTQLinearKernel) or getattr(layer,'_t011_qpn2_status',None)!='packed':return
  xr=x.reshape(-1,x.shape[-1]);M,K=xr.shape
  if M not in (1,2,3,4):return
  layername=getattr(layer,'prefix','')
  perm=layer._t011_qpn2_perm;gx=torch.empty_like(xr);fp=torch.empty((M,layer.qweight.shape[1]),device=x.device,dtype=torch.float32);actual=layer._t011_qpn2_module.run(xr,layer._t011_qpn2_packed,layer.scales,perm,gx,fp,8 if K==3072 else 16).clone()
  qw=layer.qweight.detach().cpu().numpy().view(np.uint32);N=layer.qweight.shape[1];q=np.empty((K,N),np.uint8)
  for bit in range(8):q[bit::8]=((qw>>(4*bit))&15).astype(np.uint8)
  qz=layer.qzeros.detach().cpu().numpy().view(np.uint32);zgroup=np.empty((layer.scales.shape[0],N),np.uint8)
  for bit in range(8):zgroup[:,bit::8]=((qz>>(4*bit))&15).astype(np.uint8)
  gi=layer.g_idx.detach().cpu().numpy();sc=layer.scales.detach().cpu().numpy();w=(q.astype(np.float64)-(zgroup[gi].astype(np.float64)+1))*sc.astype(np.float64)[gi];ref=torch.from_numpy((xr.detach().cpu().numpy().astype(np.float64)@w).astype(np.float32)).to(x.device);lim=1e-4+5e-5*ref.abs();err=(actual-ref).abs();bad=int((err>lim).sum().item());results.append({'layer':layername,'M':M,'K':K,'N':N,'binary_sha256':sha,'strict_bad':bad,'max_ratio':float((err/lim).max().item()),'fp32_sha256':hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),'fp16_sha256':hashlib.sha256(actual.half().cpu().numpy().tobytes()).hexdigest(),'x_sha256':hashlib.sha256(xr.detach().cpu().numpy().tobytes()).hexdigest()});
  if bad:raise RuntimeError(f'first QPN vs oracle layer failure: {results[-1]}')
 hooks=[]
 for module in model.modules():
  if hasattr(module,'gate_up_proj') and hasattr(module,'down_proj'):
   hooks.append(module.gate_up_proj.register_forward_pre_hook(lambda m,args,h=hook:h(m,args[0])));hooks.append(module.down_proj.register_forward_pre_hook(lambda m,args,h=hook:h(m,args[0])))
 from sglang.srt.model_executor.forward_context import set_forward_context
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch,ForwardMode
 # Real scheduler-owned ForwardBatch setup is required; this harness intentionally does not synthesize state.
 raise RuntimeError('hook setup complete but scheduler-owned ForwardBatch lifecycle required; use serving worker hook rather than direct model forward')
if __name__=='__main__':main()
