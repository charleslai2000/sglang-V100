from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from pathlib import Path
import importlib.util

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.parameter import BasevLLMParameter, permute_param_layout_
from sglang.srt.layers.quantization.marlin_utils import (
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_moe_permute_scales,
    marlin_permute_scales,
    marlin_sort_g_idx,
    marlin_zero_points,
)
from sglang.srt.layers.quantization.utils import (
    get_scalar_types,
    replace_parameter,
    unpack_cols,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

ScalarType, _ = get_scalar_types()


def _unsupported_kernel(*args, **kwargs):
    raise RuntimeError("GPTQ CUDA kernels are unavailable on the current platform.")


gptq_gemm = _unsupported_kernel
gptq_marlin_repack = _unsupported_kernel
_g017_dispatch_logged = False
_t011_logged_layers = set()
_t011_module_cache = {}
gptq_shuffle = _unsupported_kernel
_rebuilt_c57_module = None
_rebuilt_c57_library_path = None
_t012_prefill_q4_module = None
_t012_prefill_q4_sha = None
_t012_prefill_workspaces = {}


def apply_t012_prefill_q4(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Experimental compressed-Q4 SM70 Prefill; leaves the Decode route alone."""
    global _t012_prefill_q4_module, _t012_prefill_q4_sha
    if x.ndim != 2 or x.dtype != torch.float16 or x.shape[0] <= 0:
        raise RuntimeError(f"T012 Prefill expects a positive-row FP16 input, got {tuple(x.shape)} {x.dtype}")
    K=x.shape[1];N=layer.qweight.shape[1]
    if K != 12288 or N != 3072 or layer.scales.shape != (96,3072):
        raise RuntimeError(f"unsupported T012 GPTQ geometry K={K} N={N}")
    build=Path(os.environ.get("SGLANG_T012_PREFILL_Q4_BUILD","/data/t011/build/t012-prefill-q4-mtile-run"))
    source=Path(os.environ.get("SGLANG_T012_PREFILL_Q4_SOURCE","/data/workspace/3rdparty/sglang-V100/experiments/E044-g022-sm70-rebuilt-c57/prefill_q4_tc.py"))
    if _t012_prefill_q4_module is None:
        spec=importlib.util.spec_from_file_location("t012_prefill_q4_tc",source)
        if spec is None or spec.loader is None: raise RuntimeError(f"cannot load T012 source {source}")
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        _t012_prefill_q4_module=mod.load(str(build))
        binary=Path(_t012_prefill_q4_module.__file__).resolve();_t012_prefill_q4_sha=hashlib.sha256(binary.read_bytes()).hexdigest()
        expected=os.environ.get("SGLANG_T012_PREFILL_Q4_SHA256")
        if expected and expected != _t012_prefill_q4_sha: raise RuntimeError("T012 Prefill binary SHA mismatch")
    packed=getattr(layer,"_t012_q4_packed",None)
    if packed is None:
        spec=importlib.util.spec_from_file_location("t012_prefill_q4_tc",source)
        if spec is None or spec.loader is None: raise RuntimeError(f"cannot load T012 source {source}")
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        packed,perm=mod.prepack(layer.qweight,layer.g_idx,N);packed=packed.to(device=x.device);perm=perm.to(device=x.device)
        if not hasattr(layer,"_t012_prepack_nbytes"):
            layer._t012_prepack_nbytes=packed.numel()*packed.element_size()+perm.numel()*perm.element_size()
        layer._t012_q4_packed=packed;layer._t012_q4_perm=perm
    m=x.shape[0]
    key=(x.device.index,K,N,m)
    workspace=_t012_prefill_workspaces.get(key)
    if workspace is None:
        workspace=(torch.empty((m,K),device=x.device,dtype=torch.float16),torch.empty((m,N),device=x.device,dtype=torch.float32))
        _t012_prefill_workspaces[key]=workspace
    gather,scratch=workspace
    split,mtile=(16,8) if m<64 else (4,64)
    return _t012_prefill_q4_module.run(x.contiguous(),layer._t012_q4_packed,layer.qzeros,layer.scales,layer._t012_q4_perm,gather,scratch,split,mtile)


def gptq_gemm_fp32_groupwise(x: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    """FP32 GPTQ Q4 projection for isolated SM70 Prefill overflow diagnosis.

    Keeps the checkpoint Q4 values, group zero points/scales and g_idx mapping,
    but accumulates the dequantized group contributions in FP32. This is opt-in
    and intended to defer output casting until the caller applies its multiplier.
    """
    if x.ndim != 2 or x.dtype != torch.float16:
        raise RuntimeError("groupwise FP32 GPTQ path requires 2D FP16 activations")
    if x.shape[0] > 2048:
        raise RuntimeError(f"groupwise FP32 GPTQ probe capped at M<=2048, got {x.shape[0]}")
    qw = layer.qweight
    qz = layer.qzeros
    scales = layer.scales
    gidx = layer.g_idx
    k_size = gidx.numel()
    n_size = qw.shape[1]
    if qw.dtype != torch.int32 or qz.dtype != torch.int32 or scales.dtype != torch.float16:
        raise RuntimeError("groupwise FP32 GPTQ path requires int32 Q4 and FP16 scales")
    n = torch.arange(n_size, device=x.device, dtype=torch.int64)
    n_byte = n // 8
    n_shift = (n % 8) * 4
    packed_zero = qz.to(torch.int64)
    zero_values = ((packed_zero[:, n_byte] >> n_shift) & 0xF) + 1
    output = torch.zeros((x.shape[0], n_size), device=x.device, dtype=torch.float32)
    packed_weight = qw.to(torch.int64)
    for group in range(scales.shape[0]):
        k = torch.nonzero(gidx == group, as_tuple=False).flatten()
        if not k.numel():
            continue
        packed_rows = packed_weight.index_select(0, k // 8)
        shifts = ((k % 8) * 4).to(torch.int64).unsqueeze(1)
        q_values = ((packed_rows >> shifts) & 0xF).to(torch.float32)
        zeros = zero_values[group].to(torch.float32)
        w = (q_values - zeros.unsqueeze(0)) * scales[group].to(torch.float32).unsqueeze(0)
        x_group = x.index_select(1, k).to(torch.float32)
        output.add_(torch.matmul(x_group, w))
    return output


def _get_rebuilt_c57_op():
    global _rebuilt_c57_module, _rebuilt_c57_library_path
    library_path = os.environ.get("SGLANG_REBUILT_C57_LIBRARY")
    expected_sha = os.environ.get("SGLANG_REBUILT_C57_SHA256")
    if not library_path or not expected_sha:
        raise RuntimeError("REBUILT-C57 opt-in requires library path and SHA256")
    real_path = os.path.realpath(library_path)
    with open(real_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(f"REBUILT-C57 SHA mismatch: expected {expected_sha}, got {actual_sha}")
    if _rebuilt_c57_library_path != real_path:
        torch.ops.load_library(real_path)
        _rebuilt_c57_library_path = real_path
        _rebuilt_c57_module = torch.ops.rebuilt_c57
    schema = torch._C._dispatch_find_schema_or_throw(
        "rebuilt_c57::gptq_q4_fp32_m4_candidate", ""
    ).schema()
    expected_schema = (
        "rebuilt_c57::gptq_q4_fp32_m4_candidate(Tensor a, Tensor b_q_weight, "
        "Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, "
        "bool return_fp32=False) -> Tensor"
    )
    if str(schema) != expected_schema or not torch._C._dispatch_has_kernel_for_dispatch_key(
        "rebuilt_c57::gptq_q4_fp32_m4_candidate", "CUDA"
    ):
        raise RuntimeError(
            f"REBUILT-C57 op registration mismatch: schema={schema}, "
            "CUDA implementation missing"
        )
    return _rebuilt_c57_module

try:
    from sgl_kernel import gptq_gemm, gptq_shuffle

    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
except Exception:
    pass

# On SM70 (V100) the stock JIT gptq_marlin_repack is an __CUDA_ARCH__<800 stub
# that writes nothing -> expert weights would be repacked to all zeros. Prefer
# marlin_v100's real SM70 repack when it is available.
try:
    from sglang.srt.layers.quantization.marlin_utils import (
        _sm70_marlin_v100_repack_ops,
    )

    _sm70_gptq_repack, _ = _sm70_marlin_v100_repack_ops()
    if _sm70_gptq_repack is not None:
        gptq_marlin_repack = _sm70_gptq_repack
except Exception:
    pass


@dataclass
class MarlinLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool


def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = gptq_marlin_repack(b_q_weight[e], perm[e], size_k, size_n, num_bits)
    return output


class GPTQLinearKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config
        self.use_shuffle = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Isolated T011 opt-in: attach immutable per-layer packed Q4 + matching
        # stable activation K permutation; retain original GPTQ parameters for prefill.
        t011_status=None
        if os.environ.get("SGLANG_G022_QPN2_BINDING") == "1":
            if not (self.use_shuffle is False and self.quant_config and self.quant_config.weight_bits == 4 and self.quant_config.group_size == 128 and self.quant_config.desc_act):
                t011_status="fallback:config-ineligible"
                logging.getLogger(__name__).warning("T011 pack fallback %s: config-ineligible", getattr(layer, 'prefix', layer.__class__.__name__))
            else:
                ext_path=os.environ.get("SGLANG_G022_QPN2_EXTENSION");expected=os.environ.get("SGLANG_G022_QPN2_EXTENSION_SHA256")
                if not ext_path or not expected:raise RuntimeError("T011 binding requires extension path and SHA256")
                ext_path=os.path.realpath(ext_path);actual=hashlib.sha256(Path(ext_path).read_bytes()).hexdigest()
                if actual!=expected:raise RuntimeError(f"T011 extension SHA mismatch {actual} != {expected}")
                key=(ext_path,actual);qpn_mod=_t011_module_cache.get(key)
                if qpn_mod is None:
                    spec=importlib.util.spec_from_file_location("qpn2_multiwarp",ext_path)
                    if spec is None or spec.loader is None:raise RuntimeError("T011 extension import failed")
                    qpn_mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(qpn_mod);_t011_module_cache[key]=qpn_mod
                qw=layer.qweight.data;gidx=layer.g_idx.data;qz=layer.qzeros.data;sc=layer.scales.data;K=gidx.numel();N=qw.shape[1];G=sc.shape[0]
                reason=None
                if qw.device.type!="cuda" or torch.cuda.get_device_capability(qw.device)[0]!=7:reason="non-sm70-cuda"
                elif qw.dtype!=torch.int32 or sc.dtype!=torch.float16:reason="dtype-mismatch"
                elif K not in (3072,12288) or N not in (3072,12288,24576) or K%128 or N%32 or G!=K//128 or qw.shape[0]*8!=K:reason=f"shape-K{K}-N{N}-G{G}"
                elif not bool(torch.all(torch.stack([((qz.to(torch.int64)>>(4*b))&15)==7 for b in range(8)]))):reason="effective-zero-not-8"
                if reason:
                    t011_status="fallback:"+reason;layer.__dict__["_t011_qpn2_status"]=t011_status;logging.getLogger(__name__).warning("T011 pack fallback layer=%s reason=%s K=%d N=%d",getattr(layer,'prefix',layer.__class__.__name__),reason,K,N)
                else:
                    # CPU pack decodes the source Q4 and applies stable argsort(g_idx).
                    import numpy as np
                    qraw=qw.detach().contiguous().cpu().numpy().view("uint32");perm_np=np.argsort(gidx.detach().cpu().numpy(),kind="stable").astype("int32");q=np.empty((K,N),dtype="uint8")
                    for bit in range(8):q[bit::8]=((qraw>>(4*bit))&15).astype("uint8")
                    col=__import__('numpy').array([((lane>>2)&3)*8+(lane&3)+(4 if lane&16 else 0) for lane in range(32)],dtype="int32");ko=__import__('numpy').array([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15],dtype="int32");packed=__import__('numpy').empty((N//32,K//16,32,8),dtype="uint8")
                    for tile in range(N//32):
                        cols=tile*32+col
                        for kg in range(K//16):
                            ks=perm_np[(kg//8)*128+(kg%8)*16+ko];v=q[ks[:,None],cols[None,:]].T;packed[tile,kg]=v[:,0::2]|(v[:,1::2]<<4)
                    # Copy one module at a time from host staging to avoid a second
                    # full-model GPU allocation peak while original GPTQ weights remain resident.
                    layer.register_buffer("_t011_qpn2_packed",torch.from_numpy(packed).view(torch.int32).to(qw.device),persistent=False)
                    layer.register_buffer("_t011_qpn2_perm",torch.from_numpy(perm_np.copy()).to(qw.device,dtype=torch.int32),persistent=False)
                    layer.__dict__["_t011_qpn2_module"]=qpn_mod;layer.__dict__["_t011_qpn2_sha"]=actual;layer.__dict__["_t011_qpn2_status"]="packed";layer.__dict__["_t011_qpn2_pack_bytes"]=packed.nbytes;t011_status="packed"
                    del packed,q,perm_np,col,ko

        # for torch.compile
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = torch.nn.Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if self.use_shuffle:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty(
                    (0,), dtype=torch.int, device=layer.g_idx.device
                )
            gptq_shuffle(layer.qweight, layer.g_idx, self.quant_config.weight_bits)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        global _g017_dispatch_logged
        out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        candidate_expected_sha = os.environ.get("SGLANG_G017_CANDIDATE_SHA256")
        candidate_extension_path = os.environ.get("SGLANG_KERNEL_G017_CANDIDATE_PATH")
        t011_optin=os.environ.get("SGLANG_G022_QPN2_BINDING")=="1"
        rebuilt_c57_requested = os.environ.get("SGLANG_REBUILT_C57_ENABLE") == "1"
        rebuilt_c57_enabled = (
            not t011_optin
            and rebuilt_c57_requested
            and not self.use_shuffle
            and layer.quant_config.weight_bits == 4
            and reshaped_x.dtype == torch.float16
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(reshaped_x.device)[0] == 7
        )
        candidate_enabled = (
            not t011_optin
            and not rebuilt_c57_enabled
            and not self.use_shuffle
            and 1 <= reshaped_x.shape[0] <= 4
            and layer.quant_config.weight_bits == 4
            and reshaped_x.dtype == torch.float16
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(reshaped_x.device)[0] == 7
            and os.environ.get("SGLANG_G017_Q4_FP32_CANDIDATE") == "1"
        )
        if t011_optin and (os.environ.get("SGLANG_G017_Q4_FP32_CANDIDATE") == "1" or rebuilt_c57_requested):
            raise RuntimeError("T011 and c57 candidate opt-ins are mutually exclusive")
        if rebuilt_c57_requested and (
            t011_optin
            or self.use_shuffle
            or layer.quant_config.weight_bits != 4
            or reshaped_x.dtype != torch.float16
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability(reshaped_x.device)[0] != 7
        ):
            raise RuntimeError(
                "REBUILT-C57 was explicitly enabled but this layer is ineligible "
                f"(shuffle={self.use_shuffle}, dtype={reshaped_x.dtype})"
            )
        if rebuilt_c57_enabled and reshaped_x.shape[0] > 4:
            raise RuntimeError(
                "REBUILT-C57 explicitly requested but M exceeds kernel limit: "
                f"M={reshaped_x.shape[0]}"
            )
        t011_enabled=(t011_optin and getattr(layer,"_t011_qpn2_status",None)=="packed" and not self.use_shuffle and 1<=reshaped_x.shape[0]<=4 and reshaped_x.dtype==torch.float16 and reshaped_x.shape[1]==layer.g_idx.numel() and torch.cuda.is_available() and torch.cuda.get_device_capability(reshaped_x.device)[0]==7)
        if t011_enabled:
            expected=os.environ.get("SGLANG_G022_QPN2_EXTENSION_SHA256")
            if expected!=layer.__dict__.get("_t011_qpn2_sha"):raise RuntimeError("T011 dispatch SHA changed after load")
            key=(os.path.realpath(os.environ["SGLANG_G022_QPN2_EXTENSION"]),expected);qpn_mod=_t011_module_cache[key]
            layer_name=getattr(layer,"prefix",layer.__class__.__name__)
            if layer_name not in _t011_logged_layers:
                logging.getLogger(__name__).warning("T011 dispatch active layer=%s binary_sha256=%s M=%d K=%d N=%d packed_bytes=%d",layer_name,expected,reshaped_x.shape[0],reshaped_x.shape[1],layer.qweight.shape[1],layer._t011_qpn2_pack_bytes);_t011_logged_layers.add(layer_name)
            # Invocation-owned scratch is retained by captured graphs and never shared across layers/calls.
            gx=torch.empty_like(reshaped_x);fp32=torch.empty((reshaped_x.shape[0],layer.qweight.shape[1]),device=reshaped_x.device,dtype=torch.float32)
            split=8 if reshaped_x.shape[1]==3072 else 16
            output=qpn_mod.run(reshaped_x,layer._t011_qpn2_packed,layer.scales,layer._t011_qpn2_perm,gx,fp32,split).to(reshaped_x.dtype)
        elif rebuilt_c57_enabled:
            module = _get_rebuilt_c57_op()
            global _g017_dispatch_logged
            if not _g017_dispatch_logged:
                logging.getLogger(__name__).warning(
                    "REBUILT-C57 dispatch active layer=%s library=%s sha256=%s M=%d K=%d N=%d",
                    getattr(layer, "prefix", layer.__class__.__name__),
                    _rebuilt_c57_library_path, os.environ.get("SGLANG_REBUILT_C57_SHA256"),
                    reshaped_x.shape[0], reshaped_x.shape[1], layer.qweight.shape[-1],
                )
                _g017_dispatch_logged = True
            output = module.gptq_q4_fp32_m4_candidate(
                reshaped_x, layer.qweight.to(torch.int32), layer.qzeros.to(torch.int32),
                layer.scales, layer.g_idx, False,
            )
        elif candidate_enabled:
            # Both exact extension identity and bytes are required before dispatch.
            if not candidate_extension_path:
                raise RuntimeError(
                    "G017 candidate is enabled but SGLANG_KERNEL_G017_CANDIDATE_PATH is unset"
                )
            loaded_common_ops = __import__("sgl_kernel").common_ops
            loaded_path = os.path.realpath(loaded_common_ops.__file__)
            expected_path = os.path.realpath(candidate_extension_path)
            if loaded_path != expected_path:
                raise RuntimeError(
                    f"G017 candidate extension path mismatch: expected {expected_path}, got {loaded_path}"
                )
            with open(loaded_path, "rb") as module_file:
                loaded_sha = hashlib.sha256(module_file.read()).hexdigest()
            if not candidate_expected_sha:
                raise RuntimeError(
                    "G017 candidate is enabled but SGLANG_G017_CANDIDATE_SHA256 is unset"
                )
            if loaded_sha != candidate_expected_sha:
                raise RuntimeError(
                    f"G017 candidate extension SHA mismatch: expected {candidate_expected_sha}, got {loaded_sha} at {loaded_path}"
                )
            if not _g017_dispatch_logged:
                logging.getLogger(__name__).warning(
                    "G017 candidate GPTQ dispatch active: path=%s sha256=%s M=%d K=%d N=%d",
                    loaded_path,
                    loaded_sha,
                    reshaped_x.shape[0],
                    reshaped_x.shape[1],
                    layer.qweight.shape[-1],
                )
                _g017_dispatch_logged = True
            torch.cuda.nvtx.range_push("G017_CANDIDATE_GPTQ_FP32")
        if candidate_enabled:
            # Call the registered candidate op directly: the generic sgl_kernel
            # editable loader can otherwise bind gptq_gemm from another module.
            output = torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(
                reshaped_x,
                layer.qweight.to(torch.int32),
                layer.qzeros.to(torch.int32),
                layer.scales,
                layer.g_idx,
                False,
            )
            probe_expected = os.environ.get("G017_EXPECTED_CANDIDATE_PROBE") == "1"
            if probe_expected:
                logging.getLogger(__name__).warning(
                    "G017 candidate probe matched dispatched op M=%d K=%d N=%d",
                    reshaped_x.shape[0], reshaped_x.shape[1], layer.qweight.shape[-1]
                )
        elif not t011_enabled:
            output = gptq_gemm(
                reshaped_x,
                layer.qweight,
                layer.qzeros,
                layer.scales,
                layer.g_idx,
                self.use_shuffle,
                self.quant_config.weight_bits,
            )
        if candidate_enabled:
            torch.cuda.nvtx.range_pop()
        if output.dtype != reshaped_x.dtype:
            output = output.to(reshaped_x.dtype)
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


class GPTQMarlinLinearKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, "qweight").device
        c = self.kernel_config

        check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace(device)

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        self.w_q_name = "qweight"
        self.w_s_name = "scales"
        self.w_zp_name = "qzeros"
        self.w_gidx_name = "g_idx"

        def _transform_param(
            layer: torch.nn.Module, name: Optional[str], fn: Callable
        ) -> None:
            if name is not None and getattr(layer, name, None) is not None:

                old_param = getattr(layer, name)
                new_param = fn(old_param)
                # replace the parameter with torch.nn.Parameter for TorchDynamo
                # compatibility
                replace_parameter(
                    layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
                )

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                num_bits=c.weight_type.size_bits,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                group_size=c.group_size,
            )
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            _transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            _transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        _transform_param(layer, self.w_q_name, transform_w_q)
        _transform_param(layer, self.w_s_name, transform_w_s)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.kernel_config

        def _get_weight_params(
            layer: torch.nn.Module,
        ) -> tuple[
            torch.Tensor,  # w_q
            torch.Tensor,  # w_s
            Optional[torch.Tensor],  # w_zp,
            Optional[torch.Tensor],  # w_gidx
        ]:
            return (
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                getattr(layer, self.w_zp_name or "", None),
                getattr(layer, self.w_gidx_name or "", None),
            )

        w_q, w_s, w_zp, w_gidx = _get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            bias=bias,
        )


class GPTQMarlinMoEKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # Process act_order
        if self.quant_config.desc_act:
            # Get sorting based on g_idx
            num_experts = layer.w13_g_idx.shape[0]
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_g_idx[e][w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_g_idx[e][w2_g_idx_sort_indices[e]]
            replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        else:
            # Reset g_idx related tensors
            num_experts = layer.w13_g_idx.shape[0]
            device = layer.w13_g_idx.device
            layer.w13_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
        # Repack weights
        marlin_w13_qweight = gptq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w13_qweight.shape[2],
            self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
        marlin_w2_qweight = gptq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2],
            self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w13_scales", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.w2_scales.shape[1]
            * (
                self.quant_config.group_size
                if self.quant_config.group_size != -1
                else self.quant_config.pack_factor
            ),
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        assert get_moe_runner_backend().is_auto()
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_qweight,
            w2_qweight=layer.w2_qweight,
            w13_scales=layer.w13_scales,
            w2_scales=layer.w2_scales,
            w13_g_idx=layer.w13_g_idx,
            w2_g_idx=layer.w2_g_idx,
            w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
            weight_bits=self.quant_config.weight_bits,
            is_k_full=self.is_k_full,
        )

        return self.runner.run(dispatch_output, quant_info)
