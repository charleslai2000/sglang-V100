# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Extent-file storage backend for HiCache L3.

Replaces the per-key-file ``HiCacheFile`` with a single preallocated sparse
file + RAM index: a batch of consecutive page writes packs into fixed-size
extents (default 64 MiB), so one pread/pwrite covers a whole batch instead of
one open+read+close per page. See ``extent_allocator.py`` (pure index logic)
and ``extent_file_backend.py`` (HiCacheStorage implementation + aggregated
IO).
"""
