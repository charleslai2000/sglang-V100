"""Opt-in dense WMMA adapter for Gemma4 D512 Full Attention.

The attention kernel is unmodified from upstream fb38933. Production KV writes,
allocation, RoPE and metadata stay in SGLang; only owner-cache gather and dense
layout conversion happen here. BSD-3-Clause attribution is in UPSTREAM.md.
"""

import sys
from pathlib import Path

import torch

DENSE_TIMINGS = []


def _record_event_pair(events):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = events()
    end.record()
    return result, (start, end)


def dense_full_prefill(q, layer, backend, forward_batch, kv_indptr, kv_indices, scale):
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import g002n_dense_wmma
    except ImportError as exc:
        raise RuntimeError("G002-N requires the pinned dense WMMA extension") from exc

    pool = backend.token_to_kv_pool
    prefix_len = int((kv_indptr[1] - kv_indptr[0]).item())
    prefix_slots = kv_indices[:prefix_len].to(torch.long)
    extend_slots = forward_batch.out_cache_loc.to(torch.long)
    slots = torch.cat((prefix_slots, extend_slots))
    length = q.shape[0]
    if slots.numel() != length:
        raise RuntimeError(f"G002-N expected {length} cache slots, got {slots.numel()}")

    k_cache, v_cache = pool.get_kv_buffer(layer.layer_id)
    (dense_k, dense_v), gather_events = _record_event_pair(
        lambda: (k_cache.index_select(0, slots).contiguous(),
                 v_cache.index_select(0, slots).contiguous())
    )
    (q_dense, k_dense, v_dense), layout_events = _record_event_pair(
        lambda: (q.permute(1, 0, 2).unsqueeze(0).contiguous(),
                 dense_k.permute(1, 0, 2).unsqueeze(0).contiguous(),
                 dense_v.permute(1, 0, 2).unsqueeze(0).contiguous())
    )
    out_tuple, kernel_events = _record_event_pair(lambda: g002n_dense_wmma.fwd(
        q_dense, k_dense, v_dense, None, None,
        0.0, float(scale), True, -1, -1, 0.0, False, None,
    ))
    out, _, _, _ = out_tuple
    result, output_events = _record_event_pair(
        lambda: out.squeeze(0).permute(1, 0, 2).contiguous()
    )
    DENSE_TIMINGS.append({
        "layer": int(layer.layer_id), "length": length,
        "events": {"gather": gather_events, "qkv_layout": layout_events,
                   "kernel": kernel_events, "output_layout": output_events},
    })
    return result


def shadow_compare_full_prefill(
    q, k_extend, v_extend, layer, backend, forward_batch, kv_indptr, kv_indices,
    scale, causal, logit_cap, sinks, sliding_window_size, k_descale, v_descale,
    max_extend_len,
):
    if not causal or logit_cap != 0 or sinks is not None or sliding_window_size != -1:
        return
    if forward_batch.batch_size != 1 or forward_batch.extend_prefix_lens_cpu != [0]:
        return
    length = q.shape[0]
    if length != int(forward_batch.extend_seq_lens_cpu[0]):
        return
    pool = backend.token_to_kv_pool
    prefix_len = int((kv_indptr[1] - kv_indptr[0]).item())
    prefix_slots = kv_indices[:prefix_len].to(torch.long)
    extend_slots = forward_batch.out_cache_loc.to(torch.long)
    slots = torch.cat((prefix_slots, extend_slots))
    if not torch.equal(slots, extend_slots) or slots.numel() != length:
        raise RuntimeError("G002-N production attention slot ordering mismatch")
    k_cache, v_cache = pool.get_kv_buffer(layer.layer_id)
    dense_k = k_cache.index_select(0, slots).contiguous()
    dense_v = v_cache.index_select(0, slots).contiguous()
    if not torch.equal(dense_k, k_extend) or not torch.equal(dense_v, v_extend):
        raise RuntimeError("G002-N gathered cache differs from production K/V inputs")

    out_wmma = dense_full_prefill(q, layer, backend, forward_batch, kv_indptr, kv_indices, scale)
    out_triton = torch.empty_like(out_wmma)
    backend.extend_attention_fwd(
        q, k_extend.contiguous(), v_extend.contiguous(),
        out_triton.view(length, layer.tp_q_head_num, layer.v_head_dim),
        pool.get_key_buffer(layer.layer_id), pool.get_value_buffer(layer.layer_id),
        backend.forward_metadata.qo_indptr, kv_indptr, kv_indices,
        backend.forward_metadata.custom_mask, causal,
        backend.forward_metadata.mask_indptr, max_extend_len,
        k_descale, v_descale, scale, logit_cap=logit_cap,
        sliding_window_size=sliding_window_size, sinks=sinks,
        window_kv_offsets=None, xai_temperature_len=layer.xai_temperature_len,
    )
    delta = (out_wmma.float() - out_triton.float()).abs()
    current_layer = getattr(forward_batch, "_g002n_layer", int(layer.layer_id))
    owner = next((owner for owner, mapping in pool.layers_mapping.items()
                  if mapping[0] == pool.layers_mapping[layer.layer_id][0] and not mapping[1]),
                 layer.layer_id)
    print("G002N_DENSE_SHADOW", {
        "layer": int(current_layer), "owner": int(owner), "length": length,
        "slots_match": True, "scale": float(scale), "causal": causal,
        "max_abs": float(delta.max()), "mae": float(delta.mean()),
    }, flush=True)
