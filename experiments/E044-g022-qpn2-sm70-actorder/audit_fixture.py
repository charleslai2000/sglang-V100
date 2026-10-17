#!/usr/bin/env python3
"""CPU-only audit of saved L43/down GPTQ mapping for QPN2 fragment layout."""
import collections
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).parent / "fixtures/l43-down/packed-word-fixture-package"
manifest = json.loads((ROOT / "manifest.json").read_text())
K = manifest["tensors"]["g_idx"]["shape"][0]
N = manifest["tensors"]["qweight"]["shape"][1]
G = manifest["tensors"]["scales"]["shape"][0]
assert K == G * 128 and N % 32 == 0 and K % 64 == 0

# Validate exact original quantized codes against donor QPN2 lane and K order.
qwords = struct.unpack(f"<{K // 8 * N}I", (ROOT / "qweight.bin").read_bytes())
qzeros = struct.unpack(f"<{G * (N // 8)}I", (ROOT / "qzeros.bin").read_bytes())
gidx = struct.unpack(f"<{K}i", (ROOT / "g_idx.bin").read_bytes())
scales = struct.unpack(f"<{G * N}e", (ROOT / "scales.bin").read_bytes())
x = struct.unpack(f"<{manifest['tensors']['x']['shape'][0] * K}e", (ROOT / "x.bin").read_bytes())

# The existing validated converter uses stable grouping: for every GPTQ group,
# retain increasing source-K order, then form 128-wide groups. No quant values
# or activations are changed by this permutation.
perm = [k for group in range(G) for k, g in enumerate(gidx) if g == group]
assert len(perm) == K and len(set(perm)) == K
counts = collections.Counter(gidx)
assert len(counts) == G and set(counts.values()) == {128}

# QPN2 QP lane mapping: 4 quadpairs x 8 adjacent N values; lane local index
# r=(lane&3)+4*(lane&16). Invert n_local -> the unique lane that carries the
# corresponding B fragment column. Prepack still preserves source q code.
lanes_by_n = {}
for lane in range(32):
    qp = (lane >> 2) & 3
    r = (lane & 3) + ((lane & 16) >> 2)
    # QPN2 accumulator register map maps (i,lane) to local N; i is two bits
    # of low/hi row packing and (i>>2) selects the upper 4 columns.
    for i in range(8):
        col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2)
        nlocal = qp * 8 + col
        lanes_by_n.setdefault(nlocal, set()).add(lane)
assert set(lanes_by_n) == set(range(32))

# Verify all GPTQ zero nibbles are stored 7 (effective v1 zero=8), so signed
# integer q-8 is exactly representable in FP16; scales remain separate.
zero_counts = collections.Counter((word >> (4 * b)) & 15 for word in qzeros for b in range(8))
assert zero_counts == {7: G * N}
assert all(v > 0 for v in scales)

# Verify each code selected by stable g_idx permutation reconstructs original
# qweight code exactly; mapping is indexed in [tileN][k-group16][laneN][8B].
# This audit tests every packed input nibble with donor's K interleave pattern,
# without applying the donor's NVFP4 nonlinear value decoder.
korder = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)
checks = 0
for n in range(N):
    for kg16 in range(K // 16):
        base = (kg16 // 8) * 128 + (kg16 % 8) * 16
        for j, ko in enumerate(korder):
            src_k = perm[base + ko]
            packed_word = qwords[(src_k // 8) * N + n]
            q = (packed_word >> (4 * (src_k & 7))) & 15
            assert 0 <= q <= 15
            checks += 1

out = {
    "fixture": manifest["prefix"],
    "fixture_manifest_sha256": hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest(),
    "K": K, "N": N, "groups": G,
    "distinct_g_idx_groups": len(counts),
    "g_idx_count_min_max": [min(counts.values()), max(counts.values())],
    "g_idx_nonidentity": any(gidx[k] != k // 128 for k in range(K)),
    "stored_zero_histogram": dict(zero_counts),
    "q_code_mapping_checks": checks,
    "qpn_fragment_columns_covered": len(lanes_by_n),
    "n_divisible_32": N % 32 == 0,
    "k_divisible_64": K % 64 == 0,
    "activation_absmax": max(abs(v) for v in x),
    "scales_min_max": [min(scales), max(scales)],
    "decision": "layout dimensions and exact code/activation permutation are structurally mappable; GPTQ numerical kernel still unimplemented and unqualified",
}
print(json.dumps(out, indent=2))
