"""Opt-in same-input WMMA shadow for Gemma4 D512 Full Attention.

The Triton result is returned by the caller. This diagnostic computes an
independent WMMA result from identical post-RoPE Q and read-only owner KV, and
never writes cache or alters model execution. Upstream is BSD-3-Clause; see
experiments/G002-N/UPSTREAM.md and vendor/flash-attention-v100/LICENSE.
"""

import sys
from pathlib import Path

import torch


def shadow_compare_full_prefill(
    q,
    k_extend,
    v_extend,
    layer,
    backend,
    forward_batch,
    kv_indptr,
    kv_indices,
    scale,
    causal,
    logit_cap,
    sinks,
    sliding_window_size,
    k_descale,
    v_descale,
    max_extend_len,
):
    if not causal or logit_cap != 0 or sinks is not None or sliding_window_size != -1:
        return
    if forward_batch.batch_size != 1 or forward_batch.extend_prefix_lens_cpu != [0]:
        return
    length = q.shape[0]
    if length != int(forward_batch.extend_seq_lens_cpu[0]):
        return

    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from g002n_wmma_prefill import prefill_paged_fwd
    except ImportError as exc:
        raise RuntimeError("G002-N shadow requires the pinned WMMA extension") from exc

    pool = backend.token_to_kv_pool
    # Triton metadata defines the key sequence ordering. This is the actual
    # token-slot mapping consumed by the production attention kernel.
    # The extend Triton kernel receives newly computed K/V separately when
    # prefix length is zero; its kv_indices metadata is intentionally empty.
    # Preserve exactly that semantic order: prefix metadata (empty), followed
    # by out_cache_loc for the current extend tokens.
    prefix_slots = kv_indices[: int(kv_indptr[1].item())].to(dtype=torch.long)
    extend_slots = forward_batch.out_cache_loc.to(dtype=torch.long)
    slots = torch.cat((prefix_slots, extend_slots))
    if slots.numel() != length:
        raise RuntimeError(f"G002-N shadow expected {length} keys, got {slots.numel()}")
    k_cache, v_cache = pool.get_kv_buffer(layer.layer_id)  # resolves shared owner
    layers_mapping = getattr(pool, "layers_mapping", {})
    owner_layer = next(
        (owner for owner, mapping in layers_mapping.items()
         if mapping[0] == layers_mapping[layer.layer_id][0] and not mapping[1]),
        layer.layer_id,
    )
    dense_k = k_cache.index_select(0, slots).contiguous()
    dense_v = v_cache.index_select(0, slots).contiguous()
    if not torch.equal(slots, forward_batch.out_cache_loc.to(torch.long)):
        raise RuntimeError("G002-N token order differs from out_cache_loc order")
    if not torch.equal(dense_k, k_extend) or not torch.equal(dense_v, v_extend):
        raise RuntimeError("G002-N gathered owner cache differs from attention K/V inputs")

    # Pinned upstream binding has no dense ABI. Its only entry point requires
    # [pages,16,2,512]; pack the same dense sequence into scratch pages without
    # modifying production pool or cache. The preparation is diagnostic only.
    page_size = 16
    n_pages = (length + page_size - 1) // page_size
    padded = n_pages * page_size
    if padded != length:
        pad = padded - length
        dense_k = torch.cat((dense_k, dense_k.new_zeros((pad, 2, 512))), dim=0)
        dense_v = torch.cat((dense_v, dense_v.new_zeros((pad, 2, 512))), dim=0)
    paged_k = dense_k.view(n_pages, page_size, 2, 512)
    paged_v = dense_v.view(n_pages, page_size, 2, 512)
    q_wmma = q.permute(1, 0, 2).unsqueeze(0).contiguous()
    block_table = torch.arange(n_pages, dtype=torch.int32, device=q.device).view(1, -1)
    seq_lens = torch.tensor([length], dtype=torch.int32, device=q.device)
    out_wmma = prefill_paged_fwd(
        q_wmma, paged_k, paged_v, None, block_table, seq_lens,
        float(scale), "auto", 1.0, 1.0, True, -1,
    )
    out_wmma = out_wmma.squeeze(0).permute(1, 0, 2).contiguous().view(length, -1)
    out_triton = torch.empty_like(out_wmma)
    backend.extend_attention_fwd(
        q,
        k_extend.contiguous(),
        v_extend.contiguous(),
        out_triton.view(length, layer.tp_q_head_num, layer.v_head_dim),
        pool.get_key_buffer(layer.layer_id),
        pool.get_value_buffer(layer.layer_id),
        backend.forward_metadata.qo_indptr,
        kv_indptr,
        kv_indices,
        backend.forward_metadata.custom_mask,
        causal,
        backend.forward_metadata.mask_indptr,
        max_extend_len,
        k_descale,
        v_descale,
        scale,
        logit_cap=logit_cap,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        window_kv_offsets=None,
        xai_temperature_len=layer.xai_temperature_len,
    )
    delta = (out_wmma.float() - out_triton.float()).abs()
    current_layer = getattr(forward_batch, "_g002n_layer", int(layer.layer_id))
    # RadixAttention receives the resolved KV owner id on shared layers.
    record = {
        "layer": int(current_layer), "length": length,
        "owner": int(owner_layer), "slots_match": True,
        "scale": float(scale), "causal": causal,
        "max_abs": float(delta.max()), "mae": float(delta.mean()),
    }
    print("G002N_SHADOW", record, flush=True)
