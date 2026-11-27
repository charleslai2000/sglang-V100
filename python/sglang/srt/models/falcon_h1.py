import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.falcon_h1 import FalconH1Config
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, make_layers

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_G023_TRACE_LOCK = threading.Lock()
_G023_PROMPT_LENS = {}
_G023_DECODE_STEPS = {}


def _g023_trace(path, row):
    with _G023_TRACE_LOCK, open(path, "a") as trace:
        trace.write(json.dumps(row, sort_keys=True) + "\n")
        trace.flush()


class FalconH1MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: int,
        mlp_multipliers: List[float],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.layer_id = layer_id

        self.intermediate_size = intermediate_size
        self.tp_size = get_tensor_model_parallel_world_size()

        self.gate_multiplier, self.down_multiplier = mlp_multipliers
        self.prefill_fp32_down = os.environ.get("SGLANG_T011_FP32_PREFILL_DOWN") == "1"
        self.prefill_q4_tc = os.environ.get("SGLANG_T012_PREFILL_Q4_TC") == "1"

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
    ):
        if forward_batch is not None and forward_batch.forward_mode.is_decode() and self.layer_id == 0 and forward_batch.rids:
            self.gate_up_proj._g023_last_rids = list(forward_batch.rids)
        g023_trace = os.environ.get("G023_CAPTURE_PATH") if forward_batch is not None and forward_batch.forward_mode.is_decode() and self.layer_id == 0 and forward_batch.rids and not torch.cuda.is_current_stream_capturing() else None
        step = None
        if g023_trace is not None:
            seq_len = int(forward_batch.seq_lens.detach().cpu()[0].item())
            step = seq_len - _G023_PROMPT_LENS.setdefault(forward_batch.rids[0], seq_len) + 1
            if step == 1:
                _g023_trace(g023_trace, {"request_id": list(forward_batch.rids), "forward_mode": forward_batch.forward_mode.name, "decode_step": [step], "execution": "eager", "kind": "layer0_mlp_input", "values": x.detach().float().cpu().flatten().tolist()})
        gate_up, _ = self.gate_up_proj(x)
        if g023_trace is not None and step == 1:
            _g023_trace(g023_trace, {"request_id": list(forward_batch.rids), "forward_mode": forward_batch.forward_mode.name, "decode_step": [step], "execution": "eager", "kind": "layer0_mlp_gate_up", "values": gate_up.detach().float().cpu().flatten().tolist(), "weight_sha256": hashlib.sha256(self.gate_up_proj.qweight.detach().contiguous().cpu().numpy().tobytes()).hexdigest(), "scales_sha256": hashlib.sha256(self.gate_up_proj.scales.detach().contiguous().cpu().numpy().tobytes()).hexdigest(), "g_idx_sha256": hashlib.sha256(self.gate_up_proj.g_idx.detach().contiguous().cpu().numpy().tobytes()).hexdigest(), "scheme": type(getattr(self.gate_up_proj, 'scheme', None)).__name__, "kernel": type(getattr(getattr(self.gate_up_proj, 'scheme', None), 'kernel', None)).__name__})
        gate_up[:, : self.intermediate_size // self.tp_size] *= self.gate_multiplier

        x = self.act_fn(gate_up)
        if g023_trace is not None and step == 1:
            _g023_trace(g023_trace, {"request_id": list(forward_batch.rids), "forward_mode": forward_batch.forward_mode.name, "decode_step": [step], "execution": "eager", "kind": "layer0_mlp_activation", "values": x.detach().float().cpu().flatten().tolist()})
        if self.prefill_q4_tc and forward_batch is not None and forward_batch.forward_mode.is_extend():
            from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import apply_t012_prefill_q4
            output = apply_t012_prefill_q4(self.down_proj, x)
            x = (output * self.down_multiplier).to(self.down_proj.params_dtype)
        elif self.prefill_fp32_down and forward_batch is not None and forward_batch.forward_mode.is_extend():
            # The native SM70 GPTQ kernel returns FP16 before Falcon's small
            # down multiplier. Preserve the FP32 projection through scaling.
            from sglang.srt.layers.quantization.gptq.gptq import GPTQLinearMethod

            quant = self.down_proj.quant_method
            if not isinstance(quant, GPTQLinearMethod):
                raise RuntimeError("FP32 Prefill down path is limited to GPTQ linear layers")
            if use_reduce_scatter:
                raise RuntimeError("FP32 Prefill down path does not support reduce-scatter")
            if self.down_proj.reduce_results and get_tensor_model_parallel_world_size() > 1:
                raise RuntimeError("FP32 Prefill down path is currently TP1-only")
            output = torch.ops.t011_prefill.gptq_fp32_prefill(
                x.contiguous(),
                self.down_proj.qweight.contiguous(),
                self.down_proj.qzeros.contiguous(),
                self.down_proj.scales.contiguous(),
                self.down_proj.g_idx.contiguous(),
                float(self.down_multiplier),
                False,
                os.environ.get("SGLANG_T011_FP16_RESIDUAL_DOWN") == "1",
            )
            x = (output * self.down_multiplier).to(self.down_proj.params_dtype)
        else:
            x, _ = self.down_proj(
                x,
                skip_all_reduce=use_reduce_scatter,
            )
            x = x * self.down_multiplier
        return x


class FalconH1HybridAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: FalconH1Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_parameters["rope_theta"]
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = config.rope_parameters
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.layer_id = layer_id

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            # Falcon-H1 GPTQ checkpoints keep attention projections in BF16;
            # only the feed-forward projections use the global GPTQ config.
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=None,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        self.d_ssm = (
            int(config.mamba_expand * config.hidden_size)
            if config.mamba_d_ssm is None
            else config.mamba_d_ssm
        )

        self.mamba = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            n_groups=config.mamba_n_groups,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.hidden_act,
            use_rms_norm=config.mamba_rms_norm,
            prefix=f"{prefix}.mixer",
        )

        # FalconH1 all layers are dense and have no nextn now
        self.is_layer_sparse = False
        is_previous_layer_sparse = False
        is_next_layer_sparse = False

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.feed_forward = FalconH1MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
            mlp_multipliers=config.mlp_multipliers,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.pre_ff_layernorm,
            allow_reduce_scatter=True,
        )

        self.alt_stream = alt_stream
        self.key_multiplier = config.key_multiplier

        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.ssm_in_multiplier = config.ssm_in_multiplier

        self.attention_in_multiplier = config.attention_in_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.groups_time_state_size = self.mamba.n_groups * config.mamba_d_state
        self.zxbcdt_multipliers = config.ssm_multipliers
        self._init_mup_vector()

    def _init_mup_vector(self):
        """
        Non learnable per-block scaling vector composed of element-wise
        multipliersapplied to each separate contiguous block of the output
        of the linear projection (in_proj) before further processing
        (gating, convolution, SSM):

            - Z block:  [0 : d_ssm]                      → zxbcdt_multipliers[0]
            - X block:  [d_ssm : 2 * d_ssm]              → zxbcdt_multipliers[1]
            - B block:  [2 * d_ssm : 2 * d_ssm + G * S]  → zxbcdt_multipliers[2]
            - C block:  [2 * d_ssm + G * S : 2 * d_ssm + 2 * G * S]
                        → zxbcdt_multipliers[3]
            - dt block: [2 * d_ssm + 2 * G * S : end]    → zxbcdt_multipliers[4]

        where:
            - d_ssm:     Dimension of state-space model latent
            - G:         Number of groups (n_groups)
            - S:         SSM state size per group
            - All indices are divided by tp_size to support tensor parallelism
        """
        vector_shape = (
            2 * self.d_ssm + 2 * self.groups_time_state_size + self.config.mamba_n_heads
        ) // self.tp_size
        mup_vector = torch.ones(1, vector_shape)
        # Z vector 0 -> d_ssm
        mup_vector[:, : self.d_ssm // self.tp_size] *= self.zxbcdt_multipliers[0]
        # X vector d_ssm -> 2 * d_ssm
        mup_vector[
            :, (self.d_ssm // self.tp_size) : (2 * self.d_ssm // self.tp_size)
        ] *= self.zxbcdt_multipliers[1]
        # B vector 2 * d_ssm -> 2 * d_ssm + (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm)
            // self.tp_size : (2 * self.d_ssm + self.groups_time_state_size)
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[2]
        # C vector 2 * d_ssm + (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm + self.groups_time_state_size)
            // self.tp_size : (2 * self.d_ssm + 2 * self.groups_time_state_size)
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[3]
        # dt vector 2 * d_ssm + 2 * (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state) + n_heads
        mup_vector[
            :,
            (2 * self.d_ssm + 2 * self.groups_time_state_size) // self.tp_size :,
        ] *= self.zxbcdt_multipliers[4]

        self.register_buffer("mup_vector", mup_vector, persistent=False)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        capture_path = os.environ.get("G023_CAPTURE_PATH")
        if (
            capture_path
            and self.layer_id == 0
            and forward_batch.rids
            and forward_batch.forward_mode.is_extend()
            and not torch.cuda.is_current_stream_capturing()
        ):
            pos_cpu = positions.detach().cpu().reshape(-1).tolist()
            for request_id in forward_batch.rids:
                for position in (0, 767, 768):
                    matches = [i for i, value in enumerate(pos_cpu) if int(value) == position]
                    if matches:
                        i = matches[0]
                        qkv_single, _ = self.qkv_proj(hidden_states[i : i + 1])
                        _g023_trace(capture_path, {
                            "request_id": [request_id],
                            "forward_mode": forward_batch.forward_mode.name,
                            "decode_step": None,
                            "execution": "eager",
                            "kind": "layer0_prefill_qkv_sample",
                            "position": position,
                            "hidden_hash": hashlib.sha256(hidden_states[i].detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                            "hidden": hidden_states[i].detach().float().cpu().flatten().tolist(),
                            "rope_position": int(positions[i].item()),
                            "position_hidden_norm": float(hidden_states[i].float().norm().item()),
                            "position_hidden_sum": float(hidden_states[i].float().sum().item()),
                            "q": q[i].detach().float().cpu().flatten().tolist(),
                            "k": k[i].detach().float().cpu().flatten().tolist(),
                            "v": v[i].detach().float().cpu().flatten().tolist(),
                            "qkv_single": qkv_single[0].detach().float().cpu().flatten().tolist(),
                        })
        k = k * self.key_multiplier
        q, k = self.rotary_emb(positions, q, k)
        capture_path = os.environ.get("G023_CAPTURE_PATH")
        if (
            capture_path
            and self.layer_id == 0
            and forward_batch.forward_mode.is_decode()
            and forward_batch.rids
            and not torch.cuda.is_current_stream_capturing()
        ):
            seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
            steps = [
                int(length) - _G023_PROMPT_LENS.setdefault(rid, int(length)) + 1
                for rid, length in zip(forward_batch.rids, seq_lens)
            ]
            if all(step == 1 for step in steps):
                _g023_trace(capture_path, {
                    "request_id": list(forward_batch.rids),
                    "forward_mode": forward_batch.forward_mode.name,
                    "decode_step": steps,
                    "execution": "eager",
                    "kind": "layer0_attention_qkv",
                    "positions": positions.detach().cpu().tolist(),
                    "input_hash": hashlib.sha256(hidden_states.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "q": q.detach().float().cpu().flatten().tolist(),
                    "k": k.detach().float().cpu().flatten().tolist(),
                    "v": v.detach().float().cpu().flatten().tolist(),
                })

        attn_output = self.attn(q, k, v, forward_batch)
        if (
            os.environ.get("G023_CAPTURE_PATH")
            and self.layer_id == 0
            and forward_batch.forward_mode.is_decode()
            and forward_batch.rids
            and not torch.cuda.is_current_stream_capturing()
        ):
            req_ids = list(forward_batch.rids)
            seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
            steps = [
                int(length) - _G023_PROMPT_LENS.setdefault(rid, int(length)) + 1
                for rid, length in zip(req_ids, seq_lens)
            ]
            if all(step == 1 for step in steps):
                backend = get_attn_backend()
                req_pool = backend.req_to_token_pool.req_to_token
                kv_slots = req_pool[
                    forward_batch.req_pool_indices[:1], : int(seq_lens[0])
                ].reshape(-1)
                key = backend.token_to_kv_pool.get_key_buffer(self.layer_id)
                value = backend.token_to_kv_pool.get_value_buffer(self.layer_id)
                if key.ndim == 4:
                    key = key.view(-1, key.shape[-2], key.shape[-1])
                    value = value.view(-1, value.shape[-2], value.shape[-1])
                key_rows = key.index_select(0, kv_slots.to(torch.long))
                value_rows = value.index_select(0, kv_slots.to(torch.long))
                _g023_trace(os.environ["G023_CAPTURE_PATH"], {
                    "request_id": req_ids,
                    "forward_mode": forward_batch.forward_mode.name,
                    "decode_step": steps,
                    "execution": "eager",
                    "kind": "layer0_attention_kv_rows",
                    "all_kv": os.environ.get("G023_CAPTURE_ALL_KV") == "1",
                    "seq_len": int(seq_lens[0]),
                    "kv_slots_sha256": hashlib.sha256(kv_slots.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "kv_slots_head": kv_slots[:8].detach().cpu().tolist(),
                    "kv_slots_tail": kv_slots[-8:].detach().cpu().tolist(),
                    "kv_slot_samples": {
                        str(index): int(kv_slots[index].item())
                        for index in sorted({0, 1, min(767, kv_slots.numel() - 1), min(768, kv_slots.numel() - 1), kv_slots.numel() - 1})
                    },
                    "cached_k": hashlib.sha256(key_rows.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "cached_v": hashlib.sha256(value_rows.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "kv_rows": (
                        {
                            "k": key_rows.detach().float().cpu().flatten().tolist(),
                            "v": value_rows.detach().float().cpu().flatten().tolist(),
                        }
                        if os.environ.get("G023_CAPTURE_ALL_KV") == "1"
                        else None
                    ),
                    "kv_sample_hashes": {
                        str(index): {
                            "slot": int(kv_slots[index].item()),
                            "k": hashlib.sha256(key_rows[index].detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                            "v": hashlib.sha256(value_rows[index].detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                        }
                        for index in sorted({0, 1, min(767, key_rows.shape[0] - 1), min(768, key_rows.shape[0] - 1), key_rows.shape[0] - 1})
                    },
                    "q": hashlib.sha256(q.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "k": hashlib.sha256(k.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "v": hashlib.sha256(v.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "attn_output": hashlib.sha256(attn_output.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                })

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        capture_path = os.environ.get("G023_CAPTURE_PATH")
        capture_extend = (
            capture_path is not None
            and self.layer_id in (0, 1, 43)
            and forward_batch.forward_mode.is_extend()
            and forward_batch.rids
            and not torch.cuda.is_current_stream_capturing()
        )
        capture_sample = None
        if capture_extend:
            positions_cpu = positions.detach().cpu().reshape(-1).tolist()
            prefix_len = max(forward_batch.extend_prefix_lens_cpu or [0])
            sample_positions = sorted(
                {
                    0,
                    max(0, prefix_len - 1),
                    prefix_len,
                    int(max(positions_cpu)),
                }
            )
            capture_sample = {
                request_id: {
                    position: [i for i, value in enumerate(positions_cpu) if int(value) == position]
                    for position in sample_positions
                }
                for request_id in forward_batch.rids
            }
            for request_id, samples in capture_sample.items():
                for position, indices in samples.items():
                    if indices:
                        index = indices[0]
                        _g023_trace(capture_path, {
                            "request_id": [request_id],
                            "forward_mode": forward_batch.forward_mode.name,
                            "decode_step": None,
                            "execution": "eager",
                            "kind": "falcon_layer_input_sample",
                            "layer": self.layer_id,
                            "position": position,
                            "row_index": index,
                            "prefix_len": prefix_len,
                            "hidden": hidden_states[index].detach().float().cpu().tolist(),
                            "residual": residual[index].detach().float().cpu().tolist() if residual is not None else None,
                            "forward_iter": forward_batch.forward_iter,
                        })

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            # Attention block
            attention_hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states * self.attention_in_multiplier,
                forward_batch=forward_batch,
            )
            attention_hidden_states = attention_hidden_states * self.attn_out_multiplier
            if capture_sample:
                for request_id, samples in capture_sample.items():
                    for position, indices in samples.items():
                        if indices:
                            index = indices[0]
                            _g023_trace(capture_path, {
                                "request_id": [request_id],
                                "forward_mode": forward_batch.forward_mode.name,
                                "decode_step": None,
                                "execution": "eager",
                                "kind": "falcon_attention_output_sample",
                                "layer": self.layer_id,
                                "position": position,
                                "row_index": index,
                                "values": attention_hidden_states[index].detach().float().cpu().tolist(),
                            })

            attn_backend = get_attn_backend()
            assert isinstance(attn_backend, HybridLinearAttnBackend)
            assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)
            # Mamba block
            mamba_input = hidden_states * self.ssm_in_multiplier
            capture_path = os.environ.get("G023_CAPTURE_PATH")
            decode_steps = None
            if forward_batch.forward_mode.is_decode() and not torch.cuda.is_current_stream_capturing():
                seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
                for rid, seq_len in zip(forward_batch.rids or [], seq_lens):
                    _G023_PROMPT_LENS.setdefault(rid, int(seq_len))
                decode_steps = [
                    int(length) - _G023_PROMPT_LENS[rid] + 1
                    for rid, length in zip(forward_batch.rids or [], seq_lens)
                ]
            if (
                capture_path
                and forward_batch.forward_mode.is_decode()
                and self.layer_id == 0
                and decode_steps
                and all(step == 1 for step in decode_steps)
                and not torch.cuda.is_current_stream_capturing()
            ):
                for rid, step in zip(forward_batch.rids or [], decode_steps):
                    _G023_DECODE_STEPS[rid] = step
                row = {
                    "request_id": list(forward_batch.rids or []) or ["<synthetic-forward>"],
                    "forward_mode": forward_batch.forward_mode.name,
                    "decode_step": decode_steps,
                    "execution": "eager",
                    "kind": "falcon_layer_input",
                    "layer": self.layer_id,
                    "mamba_input": hashlib.sha256(mamba_input.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "mamba_input_values": mamba_input.detach().float().cpu().flatten().tolist(),
                    "attention_output": hashlib.sha256(attention_hidden_states.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "attention_output_values": attention_hidden_states.detach().float().cpu().flatten().tolist(),
                }
                _g023_trace(capture_path, row)
            mamba_hidden_states = torch.empty_like(hidden_states)
            attn_backend.linear_attn_backend.forward(
                self.mamba,
                mamba_input,
                mamba_hidden_states,
                layer_id=self.layer_id,
                forward_batch=forward_batch,
                mup_vector=self.mup_vector,
            )
            mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

            hidden_states = attention_hidden_states + mamba_hidden_states
            if capture_sample:
                for request_id, samples in capture_sample.items():
                    for position, indices in samples.items():
                        if indices:
                            index = indices[0]
                            _g023_trace(capture_path, {
                                "request_id": [request_id],
                                "forward_mode": forward_batch.forward_mode.name,
                                "decode_step": None,
                                "execution": "eager",
                                "kind": "falcon_combined_output_sample",
                                "layer": self.layer_id,
                                "position": position,
                                "row_index": index,
                                "values": hidden_states[index].detach().float().cpu().tolist(),
                            })
            if capture_path and self.layer_id == 0 and decode_steps and all(step == 1 for step in decode_steps) and not torch.cuda.is_current_stream_capturing():
                _g023_trace(capture_path, {
                    "request_id": list(forward_batch.rids or []),
                    "forward_mode": forward_batch.forward_mode.name,
                    "decode_step": decode_steps,
                    "execution": "eager",
                    "kind": "layer0_mamba_output",
                    "mamba_hidden": hashlib.sha256(mamba_hidden_states.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "mamba_hidden_values": mamba_hidden_states.detach().float().cpu().flatten().tolist(),
                    "combined_hidden": hashlib.sha256(hidden_states.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "combined_hidden_values": hidden_states.detach().float().cpu().flatten().tolist(),
                })

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.feed_forward(
            hidden_states, forward_batch, use_reduce_scatter
        )
        if capture_sample:
            for request_id, samples in capture_sample.items():
                for position, indices in samples.items():
                    if indices:
                        index = indices[0]
                        _g023_trace(capture_path, {
                            "request_id": [request_id],
                            "forward_mode": forward_batch.forward_mode.name,
                            "decode_step": None,
                            "execution": "eager",
                            "kind": "falcon_mlp_output_sample",
                            "layer": self.layer_id,
                            "position": position,
                            "row_index": index,
                            "values": hidden_states[index].detach().float().cpu().tolist(),
                        })

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        if capture_sample:
            for request_id, samples in capture_sample.items():
                for position, indices in samples.items():
                    if indices:
                        index = indices[0]
                        _g023_trace(capture_path, {
                            "request_id": [request_id],
                            "forward_mode": forward_batch.forward_mode.name,
                            "decode_step": None,
                            "execution": "eager",
                            "kind": "falcon_layer_output_sample",
                            "layer": self.layer_id,
                            "position": position,
                            "row_index": index,
                            "hidden": hidden_states[index].detach().float().cpu().tolist(),
                            "residual": residual[index].detach().float().cpu().tolist() if residual is not None else None,
                        })

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "falcon_h1": FalconH1HybridAttentionDecoderLayer,
}


class FalconH1Model(nn.Module):
    def __init__(
        self,
        config: FalconH1Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.embedding_multiplier = config.embedding_multiplier

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            use_attn_tp_group=is_dp_attention_enabled(),
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.infer_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        # mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        if inputs_embeds is not None:
            hidden_states = inputs_embeds * self.embedding_multiplier
        else:
            hidden_states = self.embed_tokens(input_ids) * self.embedding_multiplier

        capture_path = os.environ.get("G023_CAPTURE_PATH")
        capture_enabled = capture_path is not None and not torch.cuda.is_current_stream_capturing()
        request_meta = None
        if capture_enabled and not forward_batch.rids:
            _g023_trace(capture_path, {"request_id": ["<synthetic-forward>"], "forward_mode": forward_batch.forward_mode.name, "decode_step": None, "execution": "eager", "kind": "synthetic_forward", "forward_iter": forward_batch.forward_iter})
        if capture_enabled and forward_batch.rids:
            request_ids = list(forward_batch.rids)
            decode_mode = forward_batch.forward_mode.is_decode()
            seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
            if decode_mode:
                decode_steps = [int(seq_len) - _G023_PROMPT_LENS.get(rid, int(seq_len)) + 1 for rid, seq_len in zip(request_ids, seq_lens)]
            else:
                decode_steps = None
            request_meta = {
                "request_id": request_ids,
                "forward_mode": forward_batch.forward_mode.name,
                "decode_step": decode_steps,
                "execution": "eager",
                "forward_iter": forward_batch.forward_iter,
                "req_pool_indices": forward_batch.req_pool_indices.detach().cpu().tolist(),
            }
            _g023_trace(capture_path, {**request_meta, "kind": "forward_begin", "input_ids": input_ids.detach().cpu().tolist() if input_ids is not None else None})
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
            if capture_enabled and forward_batch.rids and (
                (forward_batch.forward_mode.is_decode() and decode_steps and all(step == 1 for step in decode_steps))
                or (forward_batch.forward_mode.is_extend() and i in (0, 43))
            ):
                row = {
                    **request_meta,
                    "kind": "falcon_layer",
                    "time_ns": time.time_ns(),
                    "layer": i,
                    "input_token_ids": input_ids.detach().cpu().tolist() if i == 0 and input_ids is not None else None,
                    "embedding_hidden": hashlib.sha256(hidden_states.detach().contiguous().cpu().numpy().tobytes()).hexdigest() if i == 0 else None,
                    "hidden_values": hidden_states.detach().float().cpu().flatten().tolist() if forward_batch.forward_mode.is_decode() and all(step == 1 for step in decode_steps or []) else None,
                    "positions": positions.detach().cpu().tolist() if i == 0 else None,
                    "seq_lens": forward_batch.seq_lens.detach().cpu().tolist() if i == 0 else None,
                    "hidden_shape": list(hidden_states.shape),
                    "hidden_dtype": str(hidden_states.dtype),
                    "hidden": hashlib.sha256(hidden_states.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "residual": None if residual is None else hashlib.sha256(residual.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                "residual_values": residual.detach().float().cpu().flatten().tolist() if residual is not None and forward_batch.forward_mode.is_decode() and all(step == 1 for step in decode_steps or []) else None,
                }
                _g023_trace(capture_path, row)

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.final_layernorm(hidden_states)
            else:
                hidden_states, _ = self.final_layernorm(hidden_states, residual)

        if capture_enabled and request_meta is not None:
            _g023_trace(capture_path, {**request_meta, "kind": "forward_end"})
        return hidden_states


class FalconH1ForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: FalconH1Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank
        self.quant_config = quant_config
        self.model = FalconH1Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.lm_head = self.lm_head.float()
        self.lm_head_multiplier = config.lm_head_multiplier
        self.logits_processor = LogitsProcessor(
            config, logit_scale=self.lm_head_multiplier
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq" in name:
                continue

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            if "A_log" in name:
                name = name.replace("A_log", "A")

            original_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue

                # Keep the checkpoint name immutable while probing Q/K/V.
                # Quantized attention candidates may be absent for this mixed
                # BF16-attention/GPTQ-FFN checkpoint; later probes must not
                # rewrite an earlier candidate into a bogus parameter name.
                candidate_name = original_name.replace(weight_name, param_name)
                if candidate_name.endswith(".bias") and candidate_name not in params_dict:
                    continue
                if candidate_name not in params_dict:
                    continue
                param = params_dict[candidate_name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                name = candidate_name
                break
            else:
                name = original_name
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # if is_pp_missing_parameter(name, self):
                #     continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


EntryClass = FalconH1ForCausalLM
