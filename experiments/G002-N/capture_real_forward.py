#!/usr/bin/env python3
"""Run actual Gemma4 E4B SGLang forward and save Full-layer attention inputs."""
import os, json, torch
os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29677'); os.environ.setdefault('RANK','0'); os.environ.setdefault('WORLD_SIZE','1'); os.environ.setdefault('LOCAL_RANK','0')
os.environ['CUDA_VISIBLE_DEVICES']='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
import sglang.srt.mem_cache.memory_pool as memory_pool
memory_pool.can_use_store_cache=lambda row_bytes:False
MODEL='/data/t011/models/gemma4-w4a16-ct'; OUT='/data/t011/artifacts/g002n'
args=ServerArgs(model_path=MODEL,host='127.0.0.1',port=30117,dtype='float16',quantization='compressed-tensors',trust_remote_code=True,device='cuda',disable_cuda_graph=True,mem_fraction_static=0.76,context_length=8192,disable_radix_cache=True,sampling_backend='pytorch',attention_backend='triton',prefill_attention_backend='triton',decode_attention_backend='triton')
set_global_server_args_for_scheduler(args); torch.cuda.set_device(0)
init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method='tcp://127.0.0.1:29677',backend='nccl'); initialize_model_parallel()
mc=ModelConfig(MODEL,trust_remote_code=True,dtype='float16',quantization='compressed-tensors',context_length=8192,enable_multimodal=False)
lc=LoadConfig(load_format='auto'); model=get_model_loader(lc,mc).load_model(model_config=mc,device_config=DeviceConfig(device='cuda',gpu_id=0))
outer=model.module if hasattr(model,'module') else model; base=outer.language_model; text=base
# Diagnostic only: use the existing PyTorch RoPE implementation instead of the
# unrelated legacy SM70 JIT helper that references an unavailable BF16 intrinsic.
for layer in text.layers:
    if hasattr(layer, 'self_attn'):
        layer.self_attn.rotary_emb._forward_method = layer.self_attn.rotary_emb.forward_native
# Retain testable actual attention operands after Gemma4 Q/K normalization and RoPE.
records={}
for i, layer in enumerate(text.layers):
    attn=layer.self_attn.attn
    if attn.head_dim==512 and i in (5,11,17,23,29,35,41):
        layer.self_attn._g002n_index=i
        def pre(mod,args,kwargs,attention=layer.self_attn):
            q,k,v=args[:3]; fb=kwargs['forward_batch']; index=attention._g002n_index
            records[index]={'q':q.detach().clone(), 'k':None if k is None else k.detach().clone(), 'v':None if v is None else v.detach().clone(), 'layer_id':int(attention.attn.layer_id), 'shared':bool(attention.is_kv_shared_layer), 'save_kv_cache':bool(kwargs.get('save_kv_cache',True)), 'out_cache_loc':fb.out_cache_loc.detach().clone(), 'seq_lens':fb.seq_lens.detach().clone(), 'extend_prefix_lens':fb.extend_prefix_lens.detach().clone()}
        attn.register_forward_pre_hook(pre,with_kwargs=True)
rpool=ReqToTokenPool(size=1,max_context_len=8192,device='cuda',enable_memory_saver=False)
head_dim=mc.hf_text_config.head_dim; swa_dim=getattr(mc.hf_text_config,'swa_head_dim',head_dim); kvheads=mc.hf_text_config.num_key_value_heads; swavdim=getattr(mc.hf_text_config,'swa_v_head_dim',swa_dim)
kv=SWAKVPool(size=8192,size_swa=8192,page_size=16,dtype=torch.float16,head_num=kvheads,head_dim=head_dim,swa_attention_layer_ids=mc.swa_attention_layer_ids,full_attention_layer_ids=mc.full_attention_layer_ids,enable_kvcache_transpose=False,device='cuda',swa_head_num=kvheads,swa_head_dim=swa_dim,swa_v_head_dim=swavdim,v_head_dim=head_dim,enable_alt_stream=False)
kv.full_to_swa_index_mapping=torch.arange(8192,dtype=torch.int32,device='cuda')
backend=TorchNativeAttnBackend(type('Runner',(),{'device':torch.device('cuda'),'req_to_token_pool':rpool,'token_to_kv_pool':kv})())
ids=json.load(open('/data/t011/artifacts/g002f-model-numerics/input_ids.json'))[0]
for length in (1024,4096):
    # Extend existing 8-token official prompt deterministically to requested length.
    seq=(ids*((length+len(ids)-1)//len(ids)))[:length]; tokens=torch.tensor(seq,device='cuda',dtype=torch.long); loc=torch.arange(length,device='cuda',dtype=torch.int64); rpool.req_to_token[1,:length]=loc.to(torch.int32)
    fb=ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=1,input_ids=tokens,req_pool_indices=torch.tensor([1],device='cuda',dtype=torch.int64),seq_lens=torch.tensor([length],device='cuda',dtype=torch.int64),out_cache_loc=loc,seq_lens_sum=length,orig_seq_lens=torch.tensor([length],device='cuda',dtype=torch.int32),return_logprob=False,is_prefill_only=False,capture_hidden_mode=CaptureHiddenMode.NULL,return_hidden_states_before_norm=False)
    fb.positions=torch.arange(length,device='cuda',dtype=torch.int32)-1;fb.extend_seq_lens=torch.tensor([length],device='cuda',dtype=torch.int32);fb.extend_prefix_lens=torch.tensor([0],device='cuda',dtype=torch.int32);fb.extend_seq_lens_cpu=[length];fb.extend_prefix_lens_cpu=[0];fb.extend_logprob_start_lens_cpu=[length];fb.extend_num_tokens=length;fb.extend_start_loc=torch.tensor([0,length],device='cuda',dtype=torch.int32);fb.top_logprobs_nums=[0];fb.token_ids_logprobs=[None];fb.temperature=torch.ones(1,device='cuda');fb.top_p=torch.ones(1,device='cuda');fb.next_token_logits_buffer=None;fb.num_token_non_padded=torch.tensor(length,device='cuda');fb.num_token_non_padded_cpu=length
    backend.init_forward_metadata(fb)
    with forward_context(ForwardContext(attn_backend=backend)): out=outer.forward(tokens,fb.positions,fb)
    torch.cuda.synchronize(); torch.save({'length':length,'input_ids':tokens.cpu(),'attention_inputs':{i:r for i,r in records.items()},'logits':out.next_token_logits.detach().cpu()},f'{OUT}/capture-{length}.pt'); records.clear()
    for pool in (kv.full_kv_pool,kv.swa_kv_pool):
        for t in pool.k_buffer+pool.v_buffer:t.zero_()
print('captured real Gemma4 Full attention inputs at 1K/4K')
