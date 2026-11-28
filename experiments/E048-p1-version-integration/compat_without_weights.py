"""P1: Falcon-H1 / Gemma compatibility checks that run without weights.

Levels, all real execution:
  1. HF config.json -> our ModelConfig (arch parser, dtype, head counts),
  2. model class construction on the meta device (no CUDA memory),
  3. causal_conv1d dispatch: width 2/3/4 must still take the sgl_kernel path
     that Falcon-H1 and older hybrid models rely on (only width>4 is rerouted
     to Triton by the E047 guard).
"""

import json
import os
import tempfile

import torch

from sglang.srt.configs.model_config import ModelConfig


GEMMA = {
    "architectures": ["GemmaForCausalLM"],
    "model_type": "gemma",
    "vocab_size": 256000,
    "hidden_size": 2048,
    "intermediate_size": 16384,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-06,
    "hidden_act": "gelu_pytorch_tanh",
    "torch_dtype": "float16",
    "rope_theta": 10000.0,
}

GEMMA2 = {
    "architectures": ["Gemma2ForCausalLM"],
    "model_type": "gemma2",
    "vocab_size": 256000,
    "hidden_size": 2304,
    "intermediate_size": 9216,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-06,
    "hidden_act": "gelu_pytorch_tanh",
    "torch_dtype": "float16",
    "rope_theta": 10000.0,
    "query_pre_attn_scalar": 256,
    "sliding_window": 4096,
    "final_logit_softcapping": 30.0,
    "attn_logit_softcapping": 50.0,
}

FALCON_H1 = {
    "architectures": ["FalconH1ForCausalLM"],
    "model_type": "falcon_h1",
    "vocab_size": 65024,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-05,
    "hidden_act": "silu",
    "torch_dtype": "float16",
    "tie_word_embeddings": False,
    "mamba_d_state": 128,
    "mamba_d_ssm": 1536,
    "mamba_n_heads": 24,
    "mamba_n_groups": 1,
    "mamba_d_head": 64,
    "mamba_expand": 2,
    "mamba_conv_bias": True,
    "mamba_proj_bias": False,
    "mamba_rms_norm": False,
    "embedding_multiplier": 1.0,
    "lm_head_multiplier": 1.0,
    "key_multiplier": 1.0,
    "attention_in_multiplier": 1.0,
    "attention_out_multiplier": 1.0,
    "ssm_in_multiplier": 1.0,
    "ssm_out_multiplier": 1.0,
    "mlp_multipliers": [1.0, 1.0],
    "ssm_multipliers": [1.0, 1.0, 1.0, 1.0, 1.0],
}


def parse(name, cfg):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        json.dump(cfg, open(path, "w"))
        mc = ModelConfig(model_path=path, trust_remote_code=False, dtype="float16")
        print(
            f"{name:10s} arch={mc.hf_config.architectures[0]:24s} layers={mc.num_hidden_layers} "
            f"heads={mc.num_attention_heads} kv={getattr(mc, 'num_key_value_heads', None)} "
            f"ctx={getattr(mc, 'context_len', getattr(mc, 'context_length', None))} dtype={mc.dtype} "
            f"hybrid_swa={getattr(mc, 'is_hybrid_swa', None)} "
            f"hybrid_ssm={getattr(mc, 'is_hybrid_ssm', None)}"
        )
        return mc


def init_dist_once():
    """Publish a single-rank topology and build the process groups once."""
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.runtime_context import get_parallel

    if ps._TP is not None:
        return

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29513",
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    sa = ServerArgs(
        model_path="/var/tmp/lfm3", trust_remote_code=True, dtype="float16",
        log_level="error",
    )
    set_global_server_args_for_scheduler(sa)
    # v0.5.23 reads topology widths and ranks from the published parallel
    # context instead of initialize_model_parallel() arguments.
    get_parallel().override_permanently(
        tp_rank=0,
        tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_dp_rank=0,
        attn_dp_size=1,
        dp_rank=0,
        dp_size=1,
    )
    initialize_model_parallel()
    initialize_dp_attention(sa)


def build_on_meta(name, mc, module_path, cls_name):
    init_dist_once()
    module = __import__(module_path, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    with torch.device("meta"):
        model = cls(config=mc.hf_config, quant_config=None)
    total = sum(p.numel() for p in model.parameters())
    names = {n for n, _ in model.named_parameters()}
    print(
        f"{name:10s} built {cls_name}: {len(names)} tensors, {total/1e6:.1f} M params, "
        f"embed={'model.embed_tokens.weight' in names}"
    )
    return model


def conv_dispatch_check():
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn

    torch.manual_seed(0)
    out = {}
    for width in (2, 3, 4, 7):
        dim, cu_len = 64, 8
        x = torch.randn(dim, cu_len, device="cuda", dtype=torch.float16)
        w = torch.randn(dim, width, device="cuda", dtype=torch.float16) * 0.1
        conv_state = torch.randn(3, dim, width - 1, device="cuda", dtype=torch.float16)
        try:
            fn_out = causal_conv1d_fn(
                x,
                w,
                None,
                conv_states=conv_state,
                query_start_loc=torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32),
                cache_indices=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
                has_initial_state=torch.zeros(2, device="cuda", dtype=torch.bool),
                activation="silu",
            )
            # fp32 reference: sliding window of width `width` ending at each step
            xs = x.float()
            ref = torch.zeros_like(xs)
            for t in range(cu_len):
                acc = torch.zeros(dim, device="cuda", dtype=torch.float32)
                for k in range(width):
                    idx = t - (width - 1) + k
                    if idx >= 0:
                        acc += xs[:, idx] * w.float()[:, k]
                ref[:, t] = acc
            ref = torch.nn.functional.silu(ref)
            max_err = float((fn_out.float() - ref).abs().max())
            out[width] = {"ran": True, "max_abs_err_vs_fp32_ref": round(max_err, 6)}
        except Exception as exc:
            out[width] = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"conv width={width}: {out[width]}")
    return out


def main():
    results = {}
    for name, cfg, module_path, cls_name in (
        ("gemma", GEMMA, "sglang.srt.models.gemma", "GemmaForCausalLM"),
        ("gemma2", GEMMA2, "sglang.srt.models.gemma2", "Gemma2ForCausalLM"),
        ("falcon_h1", FALCON_H1, "sglang.srt.models.falcon_h1", "FalconH1ForCausalLM"),
    ):
        try:
            mc = parse(name, cfg)
        except Exception as exc:
            results[name] = f"CONFIG_FAIL: {type(exc).__name__}: {exc}"
            print(f"{name:10s} config failed: {results[name]}")
            continue
        try:
            build_on_meta(name, mc, module_path, cls_name)
            results[name] = "OK"
        except Exception as exc:
            results[name] = f"BUILD_FAIL: {type(exc).__name__}: {exc}"
            print(f"{name:10s} build failed: {results[name]}")
    try:
        results["conv_dispatch"] = conv_dispatch_check()
    except Exception as exc:
        results["conv_dispatch"] = f"FAIL: {type(exc).__name__}: {exc}"
    print("COMPAT", json.dumps(results))


if __name__ == "__main__":
    main()
