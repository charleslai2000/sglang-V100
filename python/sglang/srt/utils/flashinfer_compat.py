"""Compatibility aliases for the pinned flashinfer build.

This harness runs SGLang on Tesla V100 (SM70) with the CUDA 12.x toolchain and a
locally built flashinfer, so the flashinfer wheel is older than the one upstream
SGLang pins. Some upstream modules import helper names that exist under a
different spelling in that build; the helpers have identical signatures here, so
aliasing them keeps those modules importable.

Nothing is monkeypatched unless the newer name is actually missing, and no
behaviour is substituted: only a rename is bridged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# newer name -> older name it was renamed from
_RENAMES = {
    "get_w2_permute_indices_with_cache": "_maybe_get_cached_w2_permute_indices",
}


def install_flashinfer_compat() -> None:
    """Add the newer flashinfer helper names when the build only has the old ones."""
    try:
        from flashinfer.fused_moe import core as fused_moe_core
    except Exception:  # flashinfer missing or import-time failure
        return
    for new_name, old_name in _RENAMES.items():
        if hasattr(fused_moe_core, new_name) or not hasattr(fused_moe_core, old_name):
            continue
        setattr(fused_moe_core, new_name, getattr(fused_moe_core, old_name))
        logger.debug("flashinfer compat: %s -> %s", new_name, old_name)
