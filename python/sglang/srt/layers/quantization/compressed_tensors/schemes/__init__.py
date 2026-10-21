# SPDX-License-Identifier: Apache-2.0

from .compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
)
from .compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4

# Keep SM100 MXINT4 kernels optional: their FlashInfer APIs are not present in
# all supported builds. The unavailable sentinel fails explicitly if selected.
try:
    from .compressed_tensors_w4a4_mxint4_moe import CompressedTensorsMxInt4MoE
except ImportError as exc:
    _MXINT4_IMPORT_ERROR = exc

    class CompressedTensorsMxInt4MoE:
        """Explicit unavailable sentinel for unsupported SM100 MXINT4 builds."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "MXINT4 MoE requires FlashInfer SM100 MXINT4 APIs unavailable "
                "in this installation"
            ) from _MXINT4_IMPORT_ERROR


from .compressed_tensors_w4a4_nvfp4_moe import CompressedTensorsW4A4Nvfp4MoE
from .compressed_tensors_w4a8_int8_moe import NPUCompressedTensorsW4A8Int8DynamicMoE
from .compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
from .compressed_tensors_w8a8_fp8_moe import CompressedTensorsW8A8Fp8MoE
from .compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
    NPUCompressedTensorsW8A8Int8,
)
from .compressed_tensors_w8a8_int8_moe import NPUCompressedTensorsW8A8Int8DynamicMoE
from .compressed_tensors_w8a16_fp8 import CompressedTensorsW8A16Fp8
from .compressed_tensors_wNa16 import WNA16_SUPPORTED_BITS, CompressedTensorsWNA16
from .compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
    NPUCompressedTensorsW4A16Int4DynamicMoE,
)

__all__ = [
    "CompressedTensorsLinearScheme",
    "CompressedTensorsMoEScheme",
    "CompressedTensorsW8A8Fp8",
    "CompressedTensorsW8A8Fp8MoE",
    "CompressedTensorsW8A16Fp8",
    "CompressedTensorsW8A8Int8",
    "NPUCompressedTensorsW8A8Int8",
    "NPUCompressedTensorsW8A8Int8DynamicMoE",
    "CompressedTensorsWNA16",
    "CompressedTensorsWNA16MoE",
    "CompressedTensorsWNA16TritonMoE",
    "NPUCompressedTensorsW4A16Int4DynamicMoE",
    "WNA16_SUPPORTED_BITS",
    "CompressedTensorsW4A4Fp4",
    "CompressedTensorsMxInt4MoE",
    "CompressedTensorsW4A4Nvfp4MoE",
    "NPUCompressedTensorsW4A8Int8DynamicMoE",
]
