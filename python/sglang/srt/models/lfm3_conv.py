"""LFM3 width-7 short-conv decode kernel.

The Short Conv layer of LFM3 runs a depthwise causal convolution of width 7 over
``B * x`` (elementwise gate of the packed in-projection), keeps a 6-token rolling
window per sequence, and gates the conv output with ``C``::

    bx        = B * x
    conv_out  = depthwise_causal_conv1d(bx, weight, width=7)   # sees 6 past tokens
    output    = C * conv_out

On the decode path every step used to cost six framework kernels per conv layer
(index_select, cat, mul, sum, index_put, gated mul).  This module fuses the whole
step into one Triton kernel: it reads the packed projection, computes ``B * x``,
convolves with the six carried history values, updates the history in place and
writes ``C * conv_out`` ready for ``out_proj``.

Numerics match the previous implementation exactly: products of the fp16
operands are rounded to fp16 before being accumulated in fp32, the fp32 sum is
rounded to fp16 once, bias (if any) is added in fp16, and the ``C`` gate is an
fp16 multiply.  Padded batch rows (cache index < 0) are skipped instead of
aliasing the last cache slot the way negative indexing did.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit()
def _lfm3_short_conv_decode_kernel(
    proj_ptr,  # (n, 3*dim) packed [B | C | X]
    w_ptr,  # (dim, width)
    bias_ptr,
    state_ptr,  # (num_slots, dim, width - 1)
    idx_ptr,  # (n,) int32/int64 cache slot per row, negative for padded rows
    out_ptr,  # (n, dim)
    dim: tl.constexpr,
    stride_p_row: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_s_slot: tl.constexpr,
    stride_s_dim: tl.constexpr,
    stride_s_tok: tl.constexpr,
    stride_o_row: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SKIP_RESERVED_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx_feats < dim

    cache_idx = tl.load(idx_ptr + idx_seq)
    if SKIP_RESERVED_SLOT:
        # Slot 0 is reserved for padded CUDA-graph rows and never allocated to a
        # request; a negative id can only mean the same thing. Fold that into the
        # lane mask (an early return inside non-uniform control flow is unsafe in
        # Triton) and clamp the pointer base so pointer arithmetic stays valid.
        mask = mask & (cache_idx > 0)
        cache_idx = tl.maximum(cache_idx, 1)
    cache_idx = cache_idx.to(tl.int64)

    proj_row = proj_ptr + idx_seq * stride_p_row + idx_feats
    b_gate = tl.load(proj_row, mask=mask, other=0.0)
    c_gate = tl.load(proj_row + dim, mask=mask, other=0.0)
    x = tl.load(proj_row + 2 * dim, mask=mask, other=0.0)
    bx = b_gate * x  # fp16 multiply, same rounding as `b_gate * x` in torch

    state_row = state_ptr + cache_idx * stride_s_slot + idx_feats * stride_s_dim
    w_row = w_ptr + idx_feats * stride_w_dim

    # width-7 depthwise weights and the six carried history values; both are
    # unrolled explicitly because Triton does not allow building lists in jit.
    w0 = tl.load(w_row + 0, mask=mask, other=0.0)
    w1 = tl.load(w_row + 1, mask=mask, other=0.0)
    w2 = tl.load(w_row + 2, mask=mask, other=0.0)
    w3 = tl.load(w_row + 3, mask=mask, other=0.0)
    w4 = tl.load(w_row + 4, mask=mask, other=0.0)
    w5 = tl.load(w_row + 5, mask=mask, other=0.0)
    w6 = tl.load(w_row + 6, mask=mask, other=0.0)

    s0 = tl.load(state_row + 0 * stride_s_tok, mask=mask, other=0.0)
    s1 = tl.load(state_row + 1 * stride_s_tok, mask=mask, other=0.0)
    s2 = tl.load(state_row + 2 * stride_s_tok, mask=mask, other=0.0)
    s3 = tl.load(state_row + 3 * stride_s_tok, mask=mask, other=0.0)
    s4 = tl.load(state_row + 4 * stride_s_tok, mask=mask, other=0.0)
    s5 = tl.load(state_row + 5 * stride_s_tok, mask=mask, other=0.0)

    acc = (s0 * w0).to(tl.float32)
    acc += (s1 * w1).to(tl.float32)
    acc += (s2 * w2).to(tl.float32)
    acc += (s3 * w3).to(tl.float32)
    acc += (s4 * w4).to(tl.float32)
    acc += (s5 * w5).to(tl.float32)
    acc += (bx * w6).to(tl.float32)

    val = acc.to(tl.float16)
    if HAS_BIAS:
        val = val + tl.load(bias_ptr + idx_feats, mask=mask, other=0.0)
    tl.store(out_ptr + idx_seq * stride_o_row + idx_feats, val * c_gate, mask=mask)

    # shift the rolling window left and append the new activation
    tl.store(state_row + 0 * stride_s_tok, s1, mask=mask)
    tl.store(state_row + 1 * stride_s_tok, s2, mask=mask)
    tl.store(state_row + 2 * stride_s_tok, s3, mask=mask)
    tl.store(state_row + 3 * stride_s_tok, s4, mask=mask)
    tl.store(state_row + 4 * stride_s_tok, s5, mask=mask)
    tl.store(state_row + 5 * stride_s_tok, bx, mask=mask)


def lfm3_short_conv_decode(
    proj: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    block_n: int = 256,
) -> torch.Tensor:
    """Fused decode step for the width-7 Short Conv.

    proj: (n, 3*dim) packed in-projection output ``[B | C | X]``
    conv_weight: (dim, width) depthwise weight
    conv_state: (num_slots, dim, width - 1) rolling window, updated in place
    cache_indices: (n,) state slot per row; negative marks padded rows

    Returns the gated conv output ``C * conv1d(B * X)`` of shape (n, dim).
    """
    n_rows = proj.shape[0]
    dim = conv_weight.shape[0]
    out = proj.new_empty((n_rows, dim))
    assert conv_weight.shape[1] == 7, "fused decode kernel is width-7 only"
    assert conv_state.shape[2] == 6
    if n_rows == 0:
        return out
    grid = (n_rows, triton.cdiv(dim, block_n))
    _lfm3_short_conv_decode_kernel[grid](
        proj,
        conv_weight,
        conv_bias if conv_bias is not None else conv_weight,
        conv_state,
        cache_indices,
        out,
        dim=dim,
        stride_p_row=proj.stride(0),
        stride_w_dim=conv_weight.stride(0),
        stride_s_slot=conv_state.stride(0),
        stride_s_dim=conv_state.stride(1),
        stride_s_tok=conv_state.stride(2),
        stride_o_row=out.stride(0),
        HAS_BIAS=conv_bias is not None,
        SKIP_RESERVED_SLOT=True,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out
