"""Opt-in T012 Falcon GPTQ Prefill binding; T011 Decode remains untouched."""
import hashlib
from pathlib import Path
import torch

_BINDING = {}

def bind_layer(layer, module, packed, perm, gather, scratch, binary_sha):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != binary_sha:
        raise RuntimeError("T012 Prefill binary SHA mismatch")
    K=layer.g_idx.numel();N=layer.qweight.shape[1];G=layer.scales.shape[0]
    if K != 12288 or N != 3072 or G != 96 or layer.scales.dtype != torch.float16:
        raise RuntimeError(f"unsupported T012 GPTQ geometry K={K} N={N} G={G}")
    if layer.qweight.dtype != torch.int32 or layer.qzeros.dtype != torch.int32 or layer.g_idx.dtype != torch.int32:
        raise RuntimeError("T012 expects original GPTQ int32 packed tensors")
    layer._t012_q4_packed=packed
    layer._t012_perm=perm
    layer._t012_module=module
    layer._t012_workspace=(gather,scratch)
    layer._t012_binary_sha256=binary_sha
    return layer

def apply(layer, x, multiplier, graph_static=False):
    if not hasattr(layer,"_t012_module"):
        raise RuntimeError("T012 layer was not bound")
    if x.ndim != 2 or x.shape[-1] != layer.g_idx.numel() or x.dtype != torch.float16:
        raise RuntimeError("T012 expects contiguous 2D FP16 activation with GPTQ K")
    if x.shape[0] not in (512,1024,4096):
        raise RuntimeError(f"T012 qualification binding shape unsupported M={x.shape[0]}")
    gather,scratch=layer._t012_workspace
    y=layer._t012_module.run(x.contiguous(),layer._t012_q4_packed,layer.qzeros,layer.scales,layer._t012_perm,gather,scratch,4,64)
    # Contract: group products/scales/reduction are FP32; preserve projection
    # through original down multiplier. The sole narrowing is final output.
    return (y*float(multiplier)).to(layer.params_dtype)
