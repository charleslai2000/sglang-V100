"""Native text-only SGLang implementation of LiquidAI LFM3.

LFM3 is a tri-hybrid architecture with three operator flavours per layer:

  * ``conv``              - gated short convolution (kernel width 7)
  * ``sliding_attention`` - GQA + sliding window (256) + full RoPE (theta 1e4)
  * ``full_attention``    - GQA + global attention (theta 1e6)

RoPE follows HF's ``Lfm3RotaryEmbedding``: full-head-dimension frequencies
(``partial_rotary_factor`` does not truncate them) and float32 cos/sin.

Every operator output passes through a per-head sigmoid gate (attention) or a
second projection gate (conv), and optionally a sandwich RMSNorm before joining
the residual stream.

Attention layers use ``RadixAttention`` (KV cache); conv layers use the
``HybridReqToTokenPool`` mamba conv-state cache exactly like LFM2.  LFM3 width 7
exceeds the width-4 in-tree fused kernels, so the conv uses the reference
``F.conv1d`` formulation (the same fallback upstream SGLang uses off Hopper).
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.lfm3 import Lfm3Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    track_mamba_states_if_needed,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.utils.common import add_prefix, make_layers, set_weight_attrs

logger = logging.getLogger(__name__)

_ATTENTION_TYPES = frozenset({"full_attention", "sliding_attention"})


class Lfm3RotaryEmbedding(nn.Module):
    """Per-layer-type RoPE matching HF's ``Lfm3RotaryEmbedding`` exactly.

    The inverse frequencies follow the full head dimension (``base **
    (arange(0, head_dim, 2) / head_dim)``) even though
    ``partial_rotary_factor`` is 0.5 for the full-attention layers, and the
    cos/sin tables stay in float32; only the rotated Q/K are cast back to the
    activation dtype.  Reproducing both details is required for long-context
    parity: truncating the frequency table grows a position-dependent error
    that reaches O(1) relative RMS at 4K tokens.
    """


    def __init__(self, config: Lfm3Config, layer_type: str):
        super().__init__()
        rope_params = config.rope_parameters[layer_type]
        base = rope_params["rope_theta"]
        head_dim = config.head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq.to(
            device=positions.device, dtype=torch.float32
        )
        cos = freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            # [T, H, D] -> [1, H, T, D]; HF applies RoPE in float32.
            x = x.transpose(0, 1).unsqueeze(0).float()
            x1 = x[..., : self.head_dim // 2]
            x2 = x[..., self.head_dim // 2 :]
            rotated = torch.cat((-x2, x1), dim=-1)
            out = x * cos + rotated * sin
            return out.squeeze(0).transpose(0, 1).to(dtype=query.dtype)

        return _rotate(query), _rotate(key)


class Lfm3MLP(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Lfm3Attention(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        layer_id: int,
        layer_type: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.rotary_emb = Lfm3RotaryEmbedding(config, layer_type)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        # One logit per query head, multiplied into the attention output.
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.norm_eps, cast_x_before_out_mul=True
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.norm_eps, cast_x_before_out_mul=True
        )

        self.num_local_q_heads = self.qkv_proj.num_heads
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads
        self.attn = RadixAttention(
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            # HF's sliding_window is a token COUNT; SGLang wants the max q-kv
            # distance, hence the -1.
            sliding_window_size=(
                self.sliding_window - 1 if self.sliding_window is not None else -1
            ),
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        t = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_local_q_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(t, -1, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(t, -1, self.head_dim)
        v = v.view(t, -1, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)

        gate, _ = self.gate_proj(hidden_states)
        gate = torch.sigmoid(gate).unsqueeze(-1)

        attn_output = self.attn(
            q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1), forward_batch
        )
        attn_output = attn_output.view(t, -1, self.head_dim) * gate
        output, _ = self.o_proj(attn_output.reshape(t, -1))
        return output


class Lfm3ShortConv(nn.Module):
    """Gated short conv: in_proj -> split(B, C, x) -> Bx -> conv1d -> C * out_proj."""

    def __init__(
        self,
        config: Lfm3Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.conv_kernel = int(config.conv_L_cache)
        self.use_bias = bool(config.conv_bias)
        self.hidden_size = config.hidden_size
        self.hidden_size_per_partition = (
            self.hidden_size // get_tensor_model_parallel_world_size()
        )

        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.hidden_size] * 3,
            bias=self.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("in_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        self.conv_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(self.conv_weight, {"weight_loader": sharded_weight_loader(0)})
        if self.use_bias:
            self.conv_bias = nn.Parameter(
                torch.empty(self.hidden_size_per_partition)
            )
            set_weight_attrs(
                self.conv_bias, {"weight_loader": sharded_weight_loader(0)}
            )
        else:
            self.register_parameter("conv_bias", None)

    def _depthwise_conv(self, window: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv over (D, K-1+T); returns the T new positions."""
        weight = self.conv_weight.to(window.dtype).unsqueeze(1)
        bias = self.conv_bias.to(window.dtype) if self.conv_bias is not None else None
        out = F.conv1d(
            window.unsqueeze(0),
            weight,
            bias,
            padding=0,
            groups=self.hidden_size_per_partition,
        )
        return out[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        req_to_token_pool = get_req_to_token_pool()
        layer_cache = req_to_token_pool.mamba2_layer_cache(self.layer_id)
        conv_state = layer_cache.conv[0]
        # Metadata-owned state slots: identical to a per-batch req_pool_indices
        # lookup, but precomputed by the linear-attention backend so CUDA graph
        # replay sees the captured state-index buffer.
        metadata = get_attn_backend().linear_attn_backend.forward_metadata
        mamba_indices = metadata.mamba_cache_indices

        proj, _ = self.in_proj(hidden_states)
        b_gate, c_gate, x = proj.chunk(3, dim=-1)
        bx = b_gate * x

        num_prefills = metadata.num_prefills
        if num_prefills == 0:
            conv_out = self._decode_step(bx, conv_state, mamba_indices)
            self._track_decode_state(
                conv_state, layer_cache, forward_batch, mamba_indices, 0
            )
        else:
            # extra_buffer prefix caching: snapshot B*x at the boundaries the
            # backend planned, so a later prefix hit can restore this state.
            self._store_tracked_prefill_state(
                bx, conv_state, forward_batch, metadata
            )
            conv_out = self._prefill(
                bx, conv_state, mamba_indices, metadata, forward_batch
            )

        output, _ = self.out_proj(c_gate * conv_out)
        return output

    def _store_tracked_prefill_state(
        self,
        bx: torch.Tensor,
        conv_state: torch.Tensor,
        forward_batch: ForwardBatch,
        metadata,
    ) -> None:
        """Write the conv window ending at each planned cache boundary."""
        if metadata.track_conv_indices is None:
            return
        mask = forward_batch.mamba_track_mask
        if mask is None:
            return
        dst = forward_batch.mamba_track_indices[mask]
        if dst.numel() == 0:
            return
        # bx[indices]: [n_track, K-1, D] -> [n_track, D, K-1]
        conv_state[dst] = bx[metadata.track_conv_indices].transpose(1, 2).to(
            conv_state.dtype
        )

    def _track_decode_state(
        self,
        conv_state: torch.Tensor,
        layer_cache,
        forward_batch: ForwardBatch,
        mamba_indices: torch.Tensor,
        prefill_count: int,
    ) -> None:
        """Copy just-advanced decode states to their checkpoint slots.

        Runs inside the captured decode graph: the Triton helper is a pure
        indexed copy with a data-dependent mask, so it needs no host sync.
        """
        mask = forward_batch.mamba_track_mask
        if mask is None:
            return
        num_decodes = mamba_indices.shape[0] - prefill_count
        if num_decodes <= 0:
            return
        track_mamba_states_if_needed(
            conv_state,
            layer_cache.temporal,
            mamba_indices[prefill_count:],
            mask[-num_decodes:],
            forward_batch.mamba_track_indices[-num_decodes:],
            num_decodes,
        )

    def _decode_step(
        self,
        bx: torch.Tensor,
        conv_state: torch.Tensor,
        mamba_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token rolling update for every request in the batch."""
        state_len = conv_state.shape[-1]
        state = conv_state.index_select(0, mamba_indices).to(bx.dtype)
        window = torch.cat((state, bx.unsqueeze(-1)), dim=-1)
        out = (window * self.conv_weight.to(bx.dtype).unsqueeze(0)).sum(-1)
        if self.conv_bias is not None:
            out = out + self.conv_bias.to(bx.dtype)
        conv_state[mamba_indices] = window[:, :, -state_len:].to(conv_state.dtype)
        return out

    def _prefill(
        self,
        bx: torch.Tensor,
        conv_state: torch.Tensor,
        mamba_indices: torch.Tensor,
        metadata,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Varlen causal conv per request, seeded with the cached conv window.

        The backend clears freshly allocated state slots and copies cache-hit
        slots before the forward, so every request can read its stored window:
        an empty prefix reads zeros, a continued chunk or a prefix hit reads the
        carried (or restored) window.
        """
        num_prefills = metadata.num_prefills
        starts = metadata.query_start_loc[: num_prefills + 1].tolist()
        slots = mamba_indices[:num_prefills].tolist()
        layer_cache = get_req_to_token_pool().mamba2_layer_cache(self.layer_id)

        state_len = conv_state.shape[-1]
        chunks = []
        for i, slot in enumerate(slots):
            begin, end = starts[i], starts[i + 1]
            if begin == end:
                continue
            prefix = conv_state[slot].to(bx.dtype)
            window = torch.cat((prefix, bx[begin:end].transpose(0, 1)), dim=-1)
            chunks.append(self._depthwise_conv(window).transpose(0, 1))
            conv_state[slot] = window[:, -state_len:].to(conv_state.dtype)

        if num_prefills < len(mamba_indices):
            # Mixed batch: the decode rows follow the prefill tokens.
            decode_out = self._decode_step(
                bx[starts[-1] :], conv_state, mamba_indices[num_prefills:]
            )
            chunks.append(decode_out)
            self._track_decode_state(
                conv_state, layer_cache, forward_batch, mamba_indices, num_prefills
            )

        if not chunks:
            return bx.new_empty((0, bx.shape[-1]))
        return torch.cat(chunks, dim=0)


class Lfm3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type in _ATTENTION_TYPES
        self.use_sandwich_norm = bool(config.use_sandwich_norm)

        rmsnorm_kwargs = {"eps": config.norm_eps, "cast_x_before_out_mul": True}
        self.operator_norm = RMSNorm(config.hidden_size, **rmsnorm_kwargs)
        self.ffn_norm = RMSNorm(config.hidden_size, **rmsnorm_kwargs)
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size, **rmsnorm_kwargs)
            self.post_ffn_norm = RMSNorm(config.hidden_size, **rmsnorm_kwargs)

        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(
                config=config,
                layer_id=layer_id,
                layer_type=self.layer_type,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        elif self.layer_type == "conv":
            self.conv = Lfm3ShortConv(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )
        else:
            raise ValueError(f"Unsupported LFM3 layer_type {self.layer_type!r}")

        self.feed_forward = Lfm3MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)
        if not forward_batch.forward_mode.is_idle():
            if self.is_attention_layer:
                hidden_states = self.self_attn(
                    positions, hidden_states, forward_batch
                )
            else:
                hidden_states = self.conv(hidden_states, forward_batch)
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            hidden_states = self.post_ffn_norm(hidden_states)
        return residual + hidden_states


class Lfm3Model(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Lfm3DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.embedding_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, cast_x_before_out_mul=True
        )
        self.num_attention_layers = sum(
            1 for lt in config.layer_types if lt in _ATTENTION_TYPES
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        return self.embedding_norm(hidden_states)


class Lfm3ForCausalLM(nn.Module):
    """LFM3 causal LM with hybrid attention/SWA/short-conv layers."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Lfm3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank

        self.quant_config = quant_config
        self.model = Lfm3Model(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
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
            )
        self.logits_processor = LogitsProcessor(config)
        self.num_attention_layers = self.model.num_attention_layers

    def get_num_kv_cache_layers(self) -> int:
        return self.num_attention_layers

    def get_attention_sliding_window_size(self) -> Optional[int]:
        sliding_window = self.config.sliding_window
        return sliding_window - 1 if sliding_window is not None else None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_mtp: bool = False,
    ) -> set:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("down_proj", "w2", None),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        unexpected_weights = []

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and name.startswith("lm_head."):
                continue

            # HF stores the depthwise conv weight as `.conv.conv.weight` of shape
            # (D, 1, K); SGLang keeps a (D, K) parameter `.conv.conv_weight`.
            if ".conv.conv.weight" in name:
                name = name.replace(".conv.conv.weight", ".conv.conv_weight")
                loaded_weight = loaded_weight.squeeze(1)
            elif ".conv.conv.bias" in name:
                name = name.replace(".conv.conv.bias", ".conv.conv_bias")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    unexpected_weights.append(name)
                    break
                param = params_dict[mapped]
                if shard_id is None:
                    param.weight_loader(param, loaded_weight)
                else:
                    param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name not in params_dict:
                    unexpected_weights.append(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if unexpected_weights:
            raise ValueError(f"Unexpected LFM3 weights: {unexpected_weights}")
        missing = set(params_dict) - loaded_params
        if missing:
            raise ValueError(f"Unloaded LFM3 parameters: {sorted(missing)}")
        return loaded_params


EntryClass = [Lfm3ForCausalLM]
