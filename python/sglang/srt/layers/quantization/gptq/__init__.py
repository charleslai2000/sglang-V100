# SPDX-License-Identifier: Apache-2.0

def gptq_marlin_moe_repack(*args, **kwargs):
    # Keep this public package export without importing gptq_kernels during
    # package initialization; the kernel module imports quantization utilities.
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack as _repack,
    )

    return _repack(*args, **kwargs)

from .gptq import (
    GPTQAscendConfig,
    GPTQConfig,
    GPTQLinearAscendMethod,
    GPTQLinearMethod,
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
    GPTQMarlinMoEMethod,
    GPTQMoEAscendMethod,
    check_marlin_format,
)
from .schemes import (
    GPTQAscendLinearScheme,
    GPTQLinearScheme,
    GPTQMarlinLinearScheme,
    GPTQMarlinMoEScheme,
    GPTQMoEAscendScheme,
)

__all__ = [
    "GPTQConfig",
    "GPTQAscendConfig",
    "GPTQMarlinConfig",
    "GPTQLinearMethod",
    "GPTQMoEAscendMethod",
    "GPTQMarlinLinearMethod",
    "GPTQLinearAscendMethod",
    "GPTQMarlinMoEMethod",
    "GPTQLinearScheme",
    "GPTQAscendLinearScheme",
    "GPTQMarlinLinearScheme",
    "GPTQMoEAscendScheme",
    "GPTQMarlinMoEScheme",
    "check_marlin_format",
    "gptq_marlin_moe_repack",
]
