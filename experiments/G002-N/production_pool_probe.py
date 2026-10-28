#!/usr/bin/env python3
"""Inspect real Gemma4 ModelRunner KV pool and allocator without running a Forward."""
import os
import torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29741")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner

MODEL = "/data/t011/models/gemma4-w4a16-ct"
args = ServerArgs(
    model_path=MODEL, tokenizer_path=MODEL, trust_remote_code=True,
    dtype="float16", quantization="compressed-tensors", device="cuda",
    attention_backend="triton", prefill_attention_backend="triton",
    decode_attention_backend="triton", disable_radix_cache=True,
    disable_cuda_graph=True, disable_piecewise_cuda_graph=True,
    skip_server_warmup=True, mem_fraction_static=0.76,
    context_length=8192, max_running_requests=32, chunked_prefill_size=4096,
)
set_global_server_args_for_scheduler(args)
mc = ModelConfig.from_server_args(args)
runner = ModelRunner(
    model_config=mc, mem_fraction_static=args.mem_fraction_static,
    gpu_id=0, tp_rank=0, tp_size=1, moe_ep_rank=0, moe_ep_size=1,
    pp_rank=0, pp_size=1, nccl_port=29741, server_args=args,
)
pool = runner.token_to_kv_pool
allocator = runner.token_to_kv_pool_allocator
print("RUNNER", type(runner).__name__, "BACKEND", type(runner.attn_backend).__name__)
print("POOL", type(pool).__name__, "PAGE", pool.page_size, "DTYPE", pool.dtype)
print("ALLOCATOR", type(allocator).__name__, "PAGE", allocator.page_size)
print("FULL_LAYERS", mc.full_attention_layer_ids, "SWA_LAYERS", mc.swa_attention_layer_ids)
for layer_id in mc.full_attention_layer_ids:
    attn = next(m for m in runner.model.modules() if getattr(m, "layer_id", None) == layer_id and hasattr(m, "is_kv_shared_layer"))
    owner = attn.kv_shared_layer_index if attn.is_kv_shared_layer else layer_id
    k, v = pool.get_kv_buffer(owner)
    print("LAYER", layer_id, "OWNER", owner, "SHARED", attn.is_kv_shared_layer,
          "K_SHAPE", tuple(k.shape), "K_STRIDE", k.stride(), "V_SHAPE", tuple(v.shape), "V_STRIDE", v.stride(),
          "POOL_MAP", getattr(pool, "layers_mapping", {}).get(owner))
print("DEVICE_MEM", torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

if os.environ.get("G002N_RUN_FORWARD_AB") == "1":
    import json, statistics, time
    from sglang.bench_one_batch import prepare_synthetic_inputs_for_latency_test
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.bench_one_batch import TreeCacheNamespace
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.layers.dp_attention import DpPaddingMode
    from sglang.bench_one_batch import _maybe_prepare_mlp_sync_batch
    lengths = (1024, 4096)
    token_ids = json.load(open("/data/t011/artifacts/g002f-model-numerics/input_ids.json"))[0]
    attention_modules = [m for m in runner.model.modules() if hasattr(m, "is_kv_shared_layer") and getattr(m, "head_dim", None) == 512]
    captured = {}
    hooks = []
    for module in attention_modules:
        layer_id = module.layer_id
        def capture(mod, args, kwargs, _id=layer_id):
            if os.environ.get("SGLANG_G002N_CAPTURE") == "1":
                kwargs["forward_batch"]._g002n_layer = _id
        # Outputs are captured at the returned RadixAttention module boundary.
        attn_mod = module.attn
        def hook(mod, args, output, _id=layer_id):
            if os.environ.get("SGLANG_G002N_CAPTURE") == "1":
                captured[_id] = output.detach().clone()
        hooks.append(attn_mod.register_forward_hook(hook))
        hooks.append(attn_mod.register_forward_pre_hook(capture, with_kwargs=True))
    for length in lengths:
        ids = (token_ids * ((length + len(token_ids) - 1) // len(token_ids)))[:length]
        rows = []
        for arm in ("triton", "wmma", "wmma", "triton", "triton", "wmma"):
            runner.req_to_token_pool.clear()
            runner.token_to_kv_pool_allocator.clear()
            reqs = prepare_synthetic_inputs_for_latency_test(1, length, custom_inputs=[ids])
            os.environ["SGLANG_G002N_WMMA_PREFILL"] = "1" if arm == "wmma" else "0"
            os.environ["SGLANG_G002N_CAPTURE"] = "1"
            captured.clear()
            torch.cuda.reset_peak_memory_stats()
            tree = TreeCacheNamespace(page_size=runner.server_args.page_size, device=runner.device, token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator)
            batch = ScheduleBatch.init_new(reqs=reqs, req_to_token_pool=runner.req_to_token_pool, token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, tree_cache=tree, model_config=runner.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
            batch.prepare_for_extend()
            if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
                batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
                batch.prefill_input_ids_cpu = None
            _maybe_prepare_mlp_sync_batch(batch, runner)
            fb = ForwardBatch.init_new(batch, runner)
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                result = runner.forward(fb)
            logits = result.logits_output.next_token_logits
            next_ids = runner.sample(result.logits_output, fb)
            end.record(); torch.cuda.synchronize()
            row = {"length": length, "arm": arm, "event_ms": start.elapsed_time(end),
                   "peak_allocated": torch.cuda.max_memory_allocated(),
                   "peak_reserved": torch.cuda.max_memory_reserved(),
                   "logits": logits.detach().cpu(), "next": next_ids.detach().cpu(),
                   "attention": {i:t.cpu() for i,t in captured.items()}}
            rows.append(row)
            print("AB_SAMPLE", json.dumps({"length":length,"arm":arm,"event_ms":row["event_ms"],"peak_allocated":row["peak_allocated"],"peak_reserved":row["peak_reserved"],"next":row["next"].tolist(),"layers":sorted(row["attention"])}),flush=True)
            runner.req_to_token_pool.clear(); runner.token_to_kv_pool_allocator.clear()
        ref = next(r for r in rows if r["arm"] == "triton")
        for row in rows:
            if row["arm"] != "wmma": continue
            diff = (row["logits"].float()-ref["logits"].float()).abs()
            layer_diffs = {}
            for i in row["attention"]:
                a, b = row["attention"][i].float(), ref["attention"][i].float()
                if a.shape != b.shape:
                    layer_diffs[str(i)] = {"shape_candidate": list(a.shape), "shape_baseline": list(b.shape), "status": "SHAPE_MISMATCH"}
                else:
                    delta = (a - b).abs()
                    layer_diffs[str(i)] = {"max": float(delta.max()), "mae": float(delta.mean())}
            print("AB_COMPARE", json.dumps({"length":length,"logits_max_abs":float(diff.max()),"logits_mae":float(diff.mean()),"next_equal":bool(torch.equal(row["next"],ref["next"])),"layer_diffs":layer_diffs}),flush=True)
    for h in hooks: h.remove()
