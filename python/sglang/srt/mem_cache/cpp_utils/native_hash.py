import hashlib
import logging
import os
import platform
import struct
import sys
from array import array
from functools import lru_cache
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_HASH_BACKEND_ENV = "SGLANG_HICACHE_HASH_BACKEND"


def _cpu_supports_avx2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "avx2" in f.read().lower()
    except OSError:
        return False


@lru_cache(maxsize=1)
def _load_native_hash_module() -> Any:
    if sys.byteorder != "little" or not sys.platform.startswith("linux"):
        raise RuntimeError(
            "HiCache native hash is only supported on little-endian Linux"
        )

    try:
        from sglang.srt.utils.cpp_extension_loader import (
            load_extension_with_recovery,
        )

        abs_path = os.path.dirname(os.path.abspath(__file__))
        extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
        if _cpu_supports_avx2():
            extra_cflags.append("-mavx2")
        return load_extension_with_recovery(
            name="hicache_hash_cpp",
            sources=[f"{abs_path}/hash_binding.cpp"],
            extra_cflags=extra_cflags,
            extra_ldflags=["-lcrypto"],
            with_cuda=False,
        )
    except Exception as exc:
        raise RuntimeError("Failed to load HiCache native hash extension") from exc


def _native_hash_input(token_ids: Any) -> tuple[array, int, int, bool]:
    raw_token_ids = getattr(token_ids, "raw_token_ids", None)
    raw = (
        raw_token_ids()
        if raw_token_ids is not None
        else getattr(token_ids, "token_ids", token_ids)
    )

    logical_len = len(token_ids)
    is_bigram = getattr(token_ids, "is_bigram", False)

    if isinstance(raw, array) and raw.typecode in ("I", "q", "Q", "L"):
        if is_bigram and logical_len > 0 and len(raw) < logical_len + 1:
            raise ValueError("bigram token buffer is shorter than logical length")
        return raw, logical_len, 2 if is_bigram else 1, is_bigram

    if is_bigram:
        return array("q", raw[: logical_len + 1]), logical_len, 2, is_bigram

    if logical_len == 0:
        return array("I"), logical_len, 1, is_bigram

    first_token = raw[0]
    if isinstance(first_token, tuple):
        unit_width = len(first_token)
        return (
            array("q", (elem for token in raw[:logical_len] for elem in token)),
            logical_len,
            unit_width,
            is_bigram,
        )

    return array("q", raw[:logical_len]), logical_len, 1, is_bigram


def _u32(value: int) -> int:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError("token id does not fit in uint32")
    return int(value)


def _pack(words: List[int], wide: bool) -> bytes:
    fmt = "<%d%s" % (len(words), "Q" if wide else "I")
    return struct.pack(fmt, *words)


def _hash_page(payload: bytes, prior: Optional[bytes], wide: bool = False) -> bytes:
    """Mirror hash_binding.cpp: prior digest, then the wide tag, then the page."""
    digest = hashlib.sha256()
    if prior is not None:
        digest.update(prior)
    if wide:
        digest.update(b"\xff")
    digest.update(payload)
    return digest.digest()


def _page_units(raw, start, count, unit_width: int, is_bigram: bool) -> List[int]:
    """Token words of one page, exactly as fill_regular_page/fill_bigram_page."""
    if is_bigram:
        prev = int(raw[start])
        out = []
        for i in range(count):
            nxt = int(raw[start + i + 1])
            out.extend((prev, nxt))
            prev = nxt
        return out
    return [int(raw[start * unit_width + i]) for i in range(count * unit_width)]


def _needs_wide(raw, count: int) -> bool:
    for i in range(count):
        value = int(raw[i])
        if value < 0:
            raise ValueError("token id must be a non-negative int64")
        if value > 0xFFFFFFFF:
            return True
    return False


def _wide_words(raw, count: int, is_bigram: bool) -> List[int]:
    if is_bigram:
        out = []
        for i in range(count):
            out.extend((int(raw[i]), int(raw[i + 1])))
        return out
    return [int(raw[i]) for i in range(count)]


def python_get_hash(
    raw,
    logical_len: int,
    unit_width: int,
    is_bigram: bool,
    prior_digest: Optional[bytes],
    page_size: Optional[int] = None,
):
    """Pure-Python equivalent of the ``hicache_hash_cpp`` extension.

    Used when the C++ extension cannot be built (it needs OpenSSL headers) so
    HiCache keeps working on hosts without libssl-dev. The digest chain and the
    key strings are byte-identical to the extension's.
    """
    if is_bigram:
        unit_width = 2
    raw_count = (
        logical_len + 1 if (is_bigram and logical_len > 0) else logical_len * unit_width
    )

    if page_size is None:
        digest = prior_digest
        if _needs_wide(raw, raw_count):
            words = _wide_words(raw, logical_len if is_bigram else raw_count, is_bigram)
            digest = _hash_page(_pack(words, True), digest, wide=True)
        elif logical_len > 0:
            # An empty token list hashes the (optional) prior digest only, the
            # same guard hash_all() uses before filling a bigram page.
            words = _page_units(raw, 0, logical_len, unit_width, is_bigram)
            digest = _hash_page(_pack(words, False), digest)
        else:
            digest = _hash_page(b"", digest)
        return digest.hex()

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    digests = []
    digest = prior_digest
    for start in range(0, logical_len, page_size):
        page_units = min(page_size, logical_len - start)
        page_start = start if is_bigram else start * unit_width
        count = page_units + 1 if is_bigram else page_units * unit_width
        if _needs_wide(raw[page_start:], count):
            words = _wide_words(
                raw[page_start:], page_units if is_bigram else count, is_bigram
            )
            digest = _hash_page(_pack(words, True), digest, wide=True)
        else:
            words = _page_units(raw, start, page_units, unit_width, is_bigram)
            digest = _hash_page(_pack(words, False), digest)
        digests.append(digest.hex())
    return digests


@lru_cache(maxsize=1)
def _native_backend_state() -> str:
    """Resolve which hash backend to use: "native", "python", or "disabled"."""
    requested = (os.environ.get(_HASH_BACKEND_ENV) or "auto").strip().lower()
    if requested in ("python", "pure"):
        return "disabled"
    if requested in ("native", "cpp"):
        return "native"
    try:
        _load_native_hash_module()
        return "native"
    except Exception as exc:  # missing OpenSSL headers, unsupported host, ...
        logger.warning(
            "HiCache native hash extension unavailable (%s); using the "
            "pure-Python SHA-256 implementation with identical digests",
            exc,
        )
        return "disabled"


def get_native_hash(
    token_ids: Any, prior_digest: Optional[bytes], page_size: Optional[int] = None
) -> str | list[str]:
    raw, logical_len, unit_width, is_bigram = _native_hash_input(token_ids)
    if _native_backend_state() == "native":
        try:
            return _load_native_hash_module().get_hash(
                raw, logical_len, unit_width, is_bigram, prior_digest, page_size
            )
        except Exception as exc:
            logger.warning("HiCache native hash call failed (%s); using Python", exc)
    return python_get_hash(
        raw, logical_len, unit_width, is_bigram, prior_digest, page_size
    )
