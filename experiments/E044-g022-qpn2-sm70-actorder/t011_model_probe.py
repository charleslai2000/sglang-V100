#!/usr/bin/env python3
"""Retained failed direct-loader harness; production evidence is in t011-model-runtime.json."""
import json,os,sys,time
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import GPTQLinearKernel
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from qpn2_multiwarp import load_module

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--model',default='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4');p.add_argument('--build',default='experiments/E044-g022-qpn2-sm70-actorder/build-t011');p.add_argument('--out',required=True);a=p.parse_args()
 if torch.cuda.device_count()!=1:raise RuntimeError('only one explicitly masked GPU is allowed')
 target='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';got=torch.cuda.get_device_properties(0).uuid
 if str(got)!=target:raise RuntimeError(f'wrong GPU UUID {got}')
 mod,so=load_module(a.build);sha=__import__('hashlib').sha256(so.read_bytes()).hexdigest()
 os.environ['SGLANG_G022_QPN2_MULTIWARP']='1';os.environ['SGLANG_G022_QPN2_EXTENSION']=str(so);os.environ['SGLANG_G022_QPN2_EXTENSION_SHA256']=sha;os.environ.pop('SGLANG_G017_Q4_FP32_CANDIDATE',None)
 device='cuda';torch.cuda.reset_peak_memory_stats();before=torch.cuda.mem_get_info();start=time.perf_counter()
 server=ServerArgs(model_path=a.model,trust_remote_code=True,device=device,quantization='gptq',dtype='float16',disable_cuda_graph=True)
 mc=ModelConfig(a.model,trust_remote_code=True,model_override_args=None,revision=None,context_length=16384,dtype='float16',quantization='gptq')
 loader=get_model_loader(server)
 from sglang.srt.model_loader.loader import LoadConfig
 model=loader.load_model(model_config=mc,device_config=server.device_config)
 torch.cuda.synchronize();after=torch.cuda.mem_get_info();layers=[];counts=Counter()
 for name,m in model.named_modules():
  if isinstance(getattr(m,'quant_method',None),GPTQLinearKernel):
   counts['gptq_linear']+=1
   if hasattr(m,'qpn2_packed'):
    counts['prepacked']+=1;layers.append({'name':name,'K':m.g_idx.numel(),'N':m.qweight.shape[-1],'bytes':m._qpn2_prepack_nbytes,'sha256':m._qpn2_extension_sha256})
  elif name.endswith(('gate_up_proj','down_proj')):counts['other_ffn']+=1
 result={'device':str(torch.cuda.get_device_properties(0)),'uuid':str(got),'extension':str(so),'extension_sha256':sha,'load_seconds':time.perf_counter()-start,'free_before':before[0],'free_after':after[0],'peak_allocated':torch.cuda.max_memory_allocated(),'total_prepack_bytes':sum(x['bytes'] for x in layers),'counts':dict(counts),'layers':layers}
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='layers'}))
if __name__=='__main__':main()
