#!/usr/bin/env python3
"""Run real single-GPU Gemma4 QUASAR Q4_0 SGLang prefill at 1K and 4K."""
import json
import os

import torch

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29677")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189"

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
import sglang.srt.mem_cache.memory_pool as memory_pool

MODEL = "/data/t011/models/gemma4-quasar-q4_0/gemma-4-E4B-it-QUASAR-Q4_0.gguf"
OUTPUT_DIR = "/data/t011/artifacts/g002n/quasar-q4_0"
os.makedirs(OUTPUT_DIR, exist_ok=True)
memory_pool.can_use_store_cache = lambda row_bytes: False
args = ServerArgs(
    model_path=MODEL,
    host="127.0.0.1",
    port=30117,
    dtype="float16",
    quantization="gguf",
    trust_remote_code=True,
    device="cuda",
    disable_cuda_graph=True,
    mem_fraction_static=0.76,
    context_length=8192,
    disable_radix_cache=True,
    sampling_backend="pytorch",
    attention_backend="triton",
    prefill_attention_backend="triton",
    decode_attention_backend="triton",
)
set_global_server_args_for_scheduler(args)
torch.cuda.set_device(0)
init_distributed_environment(
    world_size=1,
    rank=0,
    local_rank=0,
    distributed_init_method="tcp://127.0.0.1:29677",
    backend="nccl",
)
initialize_model_parallel()
mc = ModelConfig(
    MODEL,
    trust_remote_code=True,
    dtype="float16",
    quantization="gguf",
    context_length=8192,
    enable_multimodal=False,
)
model = get_model_loader(LoadConfig(load_format="gguf"), mc).load_model(
    model_config=mc, device_config=DeviceConfig(device="cuda", gpu_id=0)
)
outer = model.module if hasattr(model, "module") else model
text = outer.model
for layer in text.layers:
    if hasattr(layer, "self_attn"):
        layer.self_attn.rotary_emb._forward_method = layer.self_attn.rotary_emb.forward_native
rpool = ReqToTokenPool(size=1, max_context_len=8192, device="cuda", enable_memory_saver=False)
hf = mc.hf_text_config
full_dim = hf.global_head_dim
swa_dim = getattr(hf, "swa_head_dim", hf.head_dim)
kvheads = hf.num_key_value_heads
swa_v_dim = getattr(hf, "swa_v_head_dim", swa_dim)
full_v_dim = getattr(hf, "global_v_head_dim", full_dim)
kv = SWAKVPool(
    size=8192,
    size_swa=8192,
    page_size=16,
    dtype=torch.float16,
    head_num=kvheads,
    head_dim=full_dim,
    swa_attention_layer_ids=mc.swa_attention_layer_ids,
    full_attention_layer_ids=mc.full_attention_layer_ids,
    enable_kvcache_transpose=False,
    device="cuda",
    swa_head_num=kvheads,
    swa_head_dim=swa_dim,
    swa_v_head_dim=swa_v_dim,
    v_head_dim=full_v_dim,
    enable_alt_stream=False,
)
kv.full_to_swa_index_mapping = torch.cat(
    [
        torch.arange(8192, dtype=torch.int32, device="cuda"),
        torch.tensor([-1], dtype=torch.int32, device="cuda"),
    ]
)
kv.translate_loc_from_full_to_swa = lambda indices: kv.full_to_swa_index_mapping[
    torch.where(indices < 0, torch.full_like(indices, 8192), indices)
].to(torch.int32)
backend = TorchNativeAttnBackend(
    type(
        "Runner",
        (),
        {"device": torch.device("cuda"), "req_to_token_pool": rpool, "token_to_kv_pool": kv},
    )()
)
with open("/data/t011/artifacts/g002f-model-numerics/input_ids.json") as f:
    prompt = json.load(f)[0]
for length in (1024, 4096):
    seq = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
    tokens = torch.tensor(seq, device="cuda", dtype=torch.long)
    loc = torch.arange(length, device="cuda", dtype=torch.int64)
    rpool.req_to_token[1, :length] = loc.to(torch.int32)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=tokens,
        req_pool_indices=torch.tensor([1], device="cuda", dtype=torch.int64),
        seq_lens=torch.tensor([length], device="cuda", dtype=torch.int64),
        out_cache_loc=loc,
        seq_lens_sum=length,
        orig_seq_lens=torch.tensor([length], device="cuda", dtype=torch.int32),
        return_logprob=False,
        is_prefill_only=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )
    fb.positions = torch.arange(length, device="cuda", dtype=torch.int32)
    fb.extend_seq_lens = torch.tensor([length], device="cuda", dtype=torch.int32)
    fb.extend_prefix_lens = torch.tensor([0], device="cuda", dtype=torch.int32)
    fb.extend_seq_lens_cpu = [length]
    fb.extend_prefix_lens_cpu = [0]
    fb.extend_logprob_start_lens_cpu = [length]
    fb.extend_num_tokens = length
    fb.extend_start_loc = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    fb.top_logprobs_nums = [0]
    fb.token_ids_logprobs = [None]
    fb.temperature = torch.ones(1, device="cuda")
    fb.top_p = torch.ones(1, device="cuda")
    fb.next_token_logits_buffer = None
    fb.num_token_non_padded = torch.tensor(length, device="cuda")
    fb.num_token_non_padded_cpu = length
    backend.init_forward_metadata(fb)
    with forward_context(ForwardContext(attn_backend=backend)):
        output = outer.forward(tokens, fb.positions, fb)
    torch.cuda.synchronize()
    logits = output.next_token_logits.detach()
    assert torch.isfinite(logits).all(), f"non-finite logits at length {length}"
    path = f"{OUTPUT_DIR}/capture-{length}.pt"
    torch.save(
        {"length": length, "input_ids": tokens.cpu(), "logits": logits.cpu()}, path
    )
    print(
        f"length={length} logits_shape={tuple(logits.shape)} "
        f"finite=true absmax={logits.float().abs().max().item():.6f} saved={path}",
        flush=True,
    )
    for pool in (kv.full_kv_pool, kv.swa_kv_pool):
        for tensor in pool.k_buffer + pool.v_buffer:
            tensor.zero_()
print("real Gemma4 QUASAR Q4_0 1K/4K prefill passed", flush=True)
