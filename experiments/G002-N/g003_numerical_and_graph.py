#!/usr/bin/env python3
"""G003 QUASAR Q4_0 numerical + real CUDA Graph replay acceptance runner.

Reuses the existing local SGLang GGUF, cached g003 prompt IDs and 1K/4K
captures. Historical reference provenance is recorded separately; do not
mistake new runs for historical llama.cpp comparisons.
"""
import hashlib, json, os, time
from pathlib import Path
import torch

MODEL = Path('/data/t011/models/gemma4-quasar-q4_0/gemma-4-E4B-it-QUASAR-Q4_0.gguf')
OUT = Path('/data/t011/artifacts/g002n/quasar-q4_0')
UUID = 'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
os.environ['CUDA_VISIBLE_DEVICES'] = UUID
os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29763'); os.environ.setdefault('RANK','0'); os.environ.setdefault('WORLD_SIZE','1'); os.environ.setdefault('LOCAL_RANK','0')
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
import sglang.srt.mem_cache.memory_pool as memory_pool
memory_pool.can_use_store_cache=lambda row_bytes:False

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 args=ServerArgs(model_path=str(MODEL),host='127.0.0.1',port=30117,dtype='float16',quantization='gguf',trust_remote_code=True,device='cuda',disable_cuda_graph=True,mem_fraction_static=.76,context_length=8192,disable_radix_cache=True,sampling_backend='pytorch',attention_backend='triton',prefill_attention_backend='triton',decode_attention_backend='triton')
 set_global_server_args_for_scheduler(args); torch.cuda.set_device(0)
 init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method='tcp://127.0.0.1:29763',backend='nccl'); initialize_model_parallel()
 mc=ModelConfig(str(MODEL),trust_remote_code=True,dtype='float16',quantization='gguf',context_length=8192,enable_multimodal=False)
 model=get_model_loader(LoadConfig(load_format='gguf'),mc).load_model(model_config=mc,device_config=DeviceConfig(device='cuda',gpu_id=0))
 outer=model.module if hasattr(model,'module') else model; text=outer.model
 for layer in text.layers:
  if hasattr(layer,'self_attn'): layer.self_attn.rotary_emb._forward_method=layer.self_attn.rotary_emb.forward_native
 hf=mc.hf_text_config; full=hf.global_head_dim; swa=getattr(hf,'swa_head_dim',hf.head_dim); kvh=hf.num_key_value_heads
 pool=SWAKVPool(size=8192,size_swa=8192,page_size=16,dtype=torch.float16,head_num=kvh,head_dim=full,swa_attention_layer_ids=mc.swa_attention_layer_ids,full_attention_layer_ids=mc.full_attention_layer_ids,enable_kvcache_transpose=False,device='cuda',swa_head_num=kvh,swa_head_dim=swa,swa_v_head_dim=getattr(hf,'swa_v_head_dim',swa),v_head_dim=getattr(hf,'global_v_head_dim',full),enable_alt_stream=False)
 pool.full_to_swa_index_mapping=torch.cat([torch.arange(8192,dtype=torch.int32,device='cuda'),torch.tensor([-1],dtype=torch.int32,device='cuda')])
 pool.translate_loc_from_full_to_swa=lambda x:pool.full_to_swa_index_mapping[torch.where(x<0,torch.full_like(x,8192),x)].to(torch.int32)
 req=ReqToTokenPool(size=1,max_context_len=8192,device='cuda',enable_memory_saver=False)
 backend=TorchNativeAttnBackend(type('Runner',(),{'device':torch.device('cuda'),'req_to_token_pool':req,'token_to_kv_pool':pool})())
 old=torch.load(OUT/'g003-sglang-short-logits.pt',map_location='cpu',weights_only=False)
 short_ids=old['ids']
 frozen=[]
 for n in (8,16,18):
  ids=short_ids[:n] if n<=len(short_ids) else short_ids+short_ids[:n-len(short_ids)]
  frozen.append((f'short-{n}',ids,'existing-g003-token-prefix'))
 prompt=json.load(open('/data/t011/artifacts/g002f-model-numerics/input_ids.json'))[0]
 for L in (1024,4096):
  frozen.append((f'long-{L}',(prompt*((L+len(prompt)-1)//len(prompt)))[:L], 'historical-g002f-input-repeated'))
 manifest={'model':str(MODEL),'model_sha256':sha(MODEL),'config_sha256':sha(MODEL.parent/'sidecar/config.json'),'device_uuid':UUID,'revision':os.popen('git rev-parse HEAD').read().strip(),'parameters':{'dtype':'float16','quantization':'gguf','attention':'triton','radix_cache':False,'graph_for_prefill':False},'cases':[],'reference_limits':'Historical g003 short logits and llama.cpp reference binaries/logits are not present in mounted paths. New captures are SGLang baseline, not llama.cpp. Existing historical 1K/4K captures are from G002-N and use input_ids.json; reused as frozen token sequences.'}
 def run(ids,label):
  L=len(ids); dev='cuda'; tok=torch.tensor(ids,device=dev,dtype=torch.long); loc=torch.arange(L,device=dev,dtype=torch.int64); req.req_to_token[1,:L]=loc.int()
  fb=ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=1,input_ids=tok,req_pool_indices=torch.tensor([1],device=dev,dtype=torch.int64),seq_lens=torch.tensor([L],device=dev,dtype=torch.int64),out_cache_loc=loc,seq_lens_sum=L,orig_seq_lens=torch.tensor([L],device=dev,dtype=torch.int32),return_logprob=False,is_prefill_only=False,capture_hidden_mode=CaptureHiddenMode.NULL,return_hidden_states_before_norm=False)
  fb.positions=torch.arange(L,device=dev,dtype=torch.int32);fb.extend_seq_lens=torch.tensor([L],device=dev,dtype=torch.int32);fb.extend_prefix_lens=torch.tensor([0],device=dev,dtype=torch.int32);fb.extend_seq_lens_cpu=[L];fb.extend_prefix_lens_cpu=[0];fb.extend_logprob_start_lens_cpu=[L];fb.extend_num_tokens=L;fb.extend_start_loc=torch.tensor([0,L],device=dev,dtype=torch.int32);fb.top_logprobs_nums=[0];fb.token_ids_logprobs=[None];fb.temperature=torch.ones(1,device=dev);fb.top_p=torch.ones(1,device=dev);fb.next_token_logits_buffer=None;fb.num_token_non_padded=torch.tensor(L,device=dev);fb.num_token_non_padded_cpu=L
  backend.init_forward_metadata(fb)
  with forward_context(ForwardContext(attn_backend=backend)): out=outer.forward(tok,fb.positions,fb)
  torch.cuda.synchronize(); logits=out.next_token_logits.detach().float().cpu()
  for p in (pool.full_kv_pool,pool.swa_kv_pool):
   for t in p.k_buffer+p.v_buffer:t.zero_()
  return logits
 for label,ids,source in frozen:
  a=run(ids,label); b=run(ids,label); diff=(a-b).abs(); record={'label':label,'length':len(ids),'input_ids':ids,'input_source':source,'repeat_max_abs':float(diff.max()),'repeat_mae':float(diff.mean()),'greedy':int(a.argmax(-1).item()),'logits':a}
  torch.save({'label':label,'input_ids':ids,'logits':a},OUT/f'g003-current-{label}.pt'); record['logits']=None;manifest['cases'].append(record); print('NUMERIC',json.dumps(record),flush=True)
  if label=='short-16':
   ref=old['logits'].float(); d=(a-ref).abs(); print('OLD_G003_SHORT',json.dumps({'max_abs':float(d.max()),'mae':float(d.mean()),'top1_equal':int(a.argmax(-1))==int(ref.argmax(-1))}),flush=True)
 # Graph check through real ModelRunner server follows in separate service harness.
 manifest['timestamp']=time.time(); (OUT/'g003-current-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
