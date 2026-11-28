# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Trace seams for the extent backend, inert in this tree.

The upstream source of this backend carries a marker-gated durability probe
(``mem_cache/trace_context.py``, operator-created marker files under /tmp) that
was purpose-built for that lineage's P0 durability vehicle. That probe is not
ported: ``marker_present`` is always False here, so every gated record is
skipped and ``attach_trace_context`` degenerates to the identity function. The
call sites are kept so the backend's control flow stays identical to the source
and a future probe can be dropped in without touching them.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional


def marker_present(path: str) -> bool:
    """Tracing is disabled in this tree; see the module docstring."""
    return False


def attach_trace_context(record: dict[str, Any]) -> dict[str, Any]:
    """Identity: no trace context is attached when tracing is off."""
    return record


def ordered_key_set_digest(keys) -> tuple[str, int]:
    """Privacy-safe digest and count for an ordered key sequence."""
    digest = hashlib.sha256()
    count = 0
    for key in keys:
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    return digest.hexdigest(), count
