"""Unit tests for the pure-Python HiCache page-hash fallback.

``hicache_hash_cpp`` needs OpenSSL headers at build time; hosts without
``libssl-dev`` cannot build it. ``native_hash.python_get_hash`` reproduces the
extension's digests exactly (SHA-256 over little-endian uint32 page words,
chained through the prior digest, with the ``0xff`` tag on the wide path), so
HiCache storage keeps working there. These tests pin the byte-level semantics
against digests computed by hand, independent of the extension.

Pure CPU, no CUDA, no server. Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_python_hash.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import hashlib
import struct
import unittest
from array import array

from sglang.srt.mem_cache.cpp_utils.native_hash import (
    get_native_hash,
    python_get_hash,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestPythonPageHash(unittest.TestCase):
    def test_single_page_matches_plain_sha256(self):
        tokens = [1, 2, 3, 4]
        expected = sha256(struct.pack("<4I", *tokens))
        self.assertEqual(python_get_hash(tokens, 4, 1, False, None, None), expected)

    def test_page_chain_resumes_from_prior_digest(self):
        tokens = [1, 2, 3, 4]
        page_hashes = python_get_hash(tokens, 4, 1, False, None, 2)
        self.assertEqual(len(page_hashes), 2)

        first = sha256(struct.pack("<2I", 1, 2))
        second = sha256(bytes.fromhex(first) + struct.pack("<2I", 3, 4))
        self.assertEqual(page_hashes, [first, second])

    def test_prior_digest_is_prepended(self):
        prior = bytes(range(32))
        tokens = [7, 8]
        expected = sha256(prior + struct.pack("<2I", *tokens))
        self.assertEqual(python_get_hash(tokens, 2, 1, False, prior, None), expected)

    def test_bigram_hashes_token_pairs(self):
        tokens = [5, 6, 7]
        # three bigrams over four raw entries: (5,6), (6,7) -> 2 bigrams for len 3
        expected_words = (5, 6, 6, 7)
        expected = sha256(struct.pack("<4I", *expected_words))
        self.assertEqual(python_get_hash(tokens, 2, 2, True, None, None), expected)

    def test_wide_path_sets_0xff_tag_and_uint64_words(self):
        wide = 1 << 33
        tokens = [wide, 3]
        expected = sha256(b"\xff" + struct.pack("<2Q", wide, 3))
        self.assertEqual(python_get_hash(tokens, 2, 1, False, None, None), expected)

    def test_int64_array_input_matches_list_input(self):
        tokens = [11, 12, 13]
        from_array = python_get_hash(array("q", tokens), 3, 1, False, None, None)
        from_list = python_get_hash(tokens, 3, 1, False, None, None)
        self.assertEqual(from_array, from_list)

    def test_empty_token_list_hashes_only_the_prior(self):
        self.assertEqual(python_get_hash([], 0, 1, False, None, None), sha256(b""))
        prior = bytes(range(32))
        self.assertEqual(
            python_get_hash([], 0, 1, False, prior, None), sha256(prior)
        )
        # bigram over an empty list must not index the (absent) first token
        self.assertEqual(python_get_hash([], 0, 2, True, None, None), sha256(b""))
        self.assertEqual(python_get_hash([], 0, 2, True, None, 4), [])

    def test_entry_point_falls_back_when_extension_is_forced_off(self):
        import os
        from unittest import mock

        expected = python_get_hash([9, 10], 2, 1, False, None, None)
        with mock.patch.dict(os.environ, {"SGLANG_HICACHE_HASH_BACKEND": "python"}):
            from sglang.srt.mem_cache.cpp_utils import native_hash as nh

            nh._native_backend_state.cache_clear()
            try:
                self.assertEqual(get_native_hash([9, 10], None), expected)
            finally:
                nh._native_backend_state.cache_clear()


if __name__ == "__main__":
    unittest.main()
