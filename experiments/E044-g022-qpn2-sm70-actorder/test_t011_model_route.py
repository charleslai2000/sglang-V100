#!/usr/bin/env python3
"""Load Falcon through production SGLang loader with T011 per-layer bindings."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from collections import Counter
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'E035-g022-actorder-q4-wmma'))
from qpn2_multiwarp import load_module
from qpn2_layer_binding import attach
from gpu_uuid_lock import assert_cuda_uuid

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4');p.add_argument('--build',default='/mnt/data/sglang-V100/t011-build');p.add_argument('--out',default=str(HERE/'results/t011-model-route.json'));a=p.parse_args()
 if torch.cuda.device_count()!=1:raise RuntimeError('only one explicit target GPU allowed')
 assert_cuda_uuid(torch,'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');mod,binpath=load_module(a.build);sha=hashlib.sha256(binpath.read_bytes()).hexdigest()
 os.environ['SGLANG_G022_QPN2_BINDING']='1';os.environ['SGLANG_G022_QPN2_EXTENSION']=str(binpath);os.environ['SGLANG_G022_QPN2_EXTENSION_SHA256']=sha
 from sglang.srt.server_args import ServerArgs,get_global_server_args
 from sglang.srt.configs.model_config import ModelConfig
 from sglang.srt.model_loader.loader import get_model_loader
 from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import GPTQLinearKernel
 ServerArgs(model_path=a.model,trust_remote_code=True,device='cuda',quantization='gptq',dtype='float16')
 server=get_global_server_args();assert server is not None
 mc=ModelConfig(a.model,trust_remote_code=True,model_override_args=None,revision=None,context_length=16384,dtype='float16',quantization='gptq')
 t=time.perf_counter();before=torch.cuda.mem_get_info();model=get_model_loader(server).load_model(model_config=mc,device_config=server.device_config);torch.cuda.synchronize();after=torch.cuda.mem_get_info();counts=Counter();eligible=[];skipped=[]
 for name,layer in model.named_modules():
  method=getattr(layer,'quant_method',None)
  if isinstance(method,GPTQLinearKernel):
   counts['gptq_layers']+=1;K=layer.g_idx.numel();N=layer.qweight.shape[1]
   if K in (3072,12288) and N in (3072,12288,24576):
    r=attach(layer,4,128,str(binpath),sha);(eligible if r['eligible'] else skipped).append({'name':name,**r})
   else:skipped.append({'name':name,'eligible':False,'reason':f'unsupported-shape-K{K}-N{N}'})
   if hasattr(layer,'_t011_qpn2_packed'):counts['prepacked']+=1
 result={'uuid':str(torch.cuda.get_device_properties(0).uuid),'binary':str(binpath),'binary_sha256':sha,'load_seconds':time.perf_counter()-t,'free_before_bytes':before[0],'free_after_bytes':after[0],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'counts':dict(counts),'eligible':eligible,'skipped':skipped};Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('eligible','skipped')}))
if __name__=='__main__':main()
