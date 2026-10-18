"""T011 per-layer loader binding helpers; no global mutable weight state."""
from __future__ import annotations
import hashlib, importlib.util, os
from pathlib import Path
import torch
from qpn2_multiwarp import prepack_cpu


def attach(layer: torch.nn.Module, weight_bits: int, group_size: int, extension_path: str, expected_sha: str) -> dict:
    """Attach immutable packed layout when verified, else return explicit reason for fallback."""
    reason=None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:reason='non-sm70'
    elif weight_bits!=4:reason='non-q4'
    elif group_size!=128:reason='non-group128'
    elif layer.qweight.device.type!='cuda':reason='weights-not-cuda'
    elif layer.scales.dtype!=torch.float16:reason='non-fp16-scales'
    else:
        K=layer.g_idx.numel();N=layer.qweight.shape[1];G=layer.scales.shape[0]
        if K not in (3072,12288) or N not in (3072,12288,24576) or K%128 or N%32 or G!=K//128:reason=f'unsupported-shape-K{K}-N{N}-G{G}'
        elif layer.qweight.shape[0]*8!=K:reason='qweight-K-shape-mismatch'
        elif layer.qzeros.device!=layer.qweight.device or layer.g_idx.device!=layer.qweight.device:reason='weight-device-mismatch'
    if reason:return {'eligible':False,'reason':reason}
    actual=hashlib.sha256(Path(extension_path).read_bytes()).hexdigest()
    if not expected_sha or actual!=expected_sha:return {'eligible':False,'reason':'extension-sha-mismatch','actual_sha':actual}
    z=layer.qzeros.detach().to(torch.int64)
    if not bool(torch.all(torch.stack([((z>>(4*b))&15)==7 for b in range(8)]))):return {'eligible':False,'reason':'qzero-not-effective-8'}
    packed,perm=prepack_cpu(layer.qweight,layer.g_idx)
    packed=packed.view(torch.int32).to(layer.qweight.device);perm=perm.to(layer.qweight.device,dtype=torch.int32)
    layer.register_buffer('_t011_qpn2_packed',packed,persistent=False)
    layer.register_buffer('_t011_qpn2_perm',perm,persistent=False)
    spec=importlib.util.spec_from_file_location('qpn2_multiwarp',extension_path)
    if spec is None or spec.loader is None:raise RuntimeError('cannot import SHA-verified T011 extension')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    layer.__dict__['_t011_qpn2_module']=module
    layer.__dict__['_t011_qpn2_sha']=actual
    layer.__dict__['_t011_qpn2_pack_bytes']=packed.numel()*packed.element_size()
    return {'eligible':True,'reason':'packed','binary_sha256':actual,'packed_bytes':layer.__dict__['_t011_qpn2_pack_bytes'],'perm_sha256':hashlib.sha256(perm.cpu().numpy().tobytes()).hexdigest()}


def apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor|None, split: int|None=None, scratch: tuple[torch.Tensor,torch.Tensor]|None=None):
    """Run only eligible M1-4. Returns None for explicit native fallback."""
    if not hasattr(layer,'_t011_qpn2_packed') or x.device.type!='cuda' or x.dtype!=torch.float16:return None
    xr=x.reshape(-1,x.shape[-1]);M,K=xr.shape
    if not 1<=M<=4 or K!=layer.g_idx.numel():return None
    if split is None:split=8 if K==3072 else 16
    # Caller may supply request-owned scratch for graph pools; otherwise allocate
    # fresh per invocation, with no global/cross-layer alias.
    if scratch is None:
        gx=torch.empty_like(xr);out=torch.empty((M,layer.qweight.shape[1]),device=x.device,dtype=torch.float32)
    else:
        gx,out=scratch
        if gx.shape!=xr.shape or gx.dtype!=torch.float16 or out.shape!=(M,layer.qweight.shape[1]) or out.dtype!=torch.float32 or gx.device!=x.device or out.device!=x.device:raise ValueError('request-owned QPN scratch shape/device/dtype mismatch')
    y=layer.__dict__['_t011_qpn2_module'].run(xr,layer._t011_qpn2_packed,layer.scales,layer._t011_qpn2_perm,gx,out,split).to(torch.float16)
    if bias is not None:y=y+bias
    return y.reshape(x.shape[:-1]+(layer.qweight.shape[-1],))
