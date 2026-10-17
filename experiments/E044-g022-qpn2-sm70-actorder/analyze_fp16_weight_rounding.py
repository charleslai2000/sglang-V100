#!/usr/bin/env python3
"""CPU numerical feasibility experiment: QPN-style FP16 weights, FP32 accumulation.

For each GPTQ weight q, the proposed arithmetic rounds the separately decoded
(q - 8) to FP16, then applies its independent FP16 GPTQ scale to an FP16
activation and accumulates in FP32. This is only a mathematical model of the
proposed MMA operands, not a GPU kernel simulation (MMA reduction ordering is
not represented).
"""
import hashlib, json, struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent / "fixtures/l43-down/packed-word-fixture-package"
m = json.loads((ROOT / "manifest.json").read_text())
K = m["tensors"]["g_idx"]["shape"][0]
N = m["tensors"]["qweight"]["shape"][1]
G = m["tensors"]["scales"]["shape"][0]
M = m["tensors"]["x"]["shape"][0]
qword = np.fromfile(ROOT / "qweight.bin", dtype="<u4").reshape(K // 8, N)
qzero = np.fromfile(ROOT / "qzeros.bin", dtype="<u4").reshape(G, N // 8)
scale = np.fromfile(ROOT / "scales.bin", dtype="<f2").reshape(G, N).astype(np.float64)
gidx = np.fromfile(ROOT / "g_idx.bin", dtype="<i4")
xh = np.fromfile(ROOT / "x.bin", dtype="<f2").reshape(M, K)
stress = np.fromfile(ROOT / "c57_stress_x_fp16.bin", dtype="<f2").reshape(M, K)

q = np.empty((K, N), np.float64)
for b in range(8):
    q[b::8] = ((qword >> (4*b)) & 15).astype(np.float64)
z = np.empty((G, N), np.float64)
for b in range(8):
    z[:, b::8] = ((qzero >> (4*b)) & 15).astype(np.float64) + 1
assert np.all(z == 8)
w64 = (q - z[gidx]) * scale[gidx]

# Candidate-order model. Half MMA operand must contain the signed integer
# exactly, and independent half scales multiply the FP32 MMA subtotal after
# each original GPTQ group. Stable grouping preserves original K order within
# each group; inter-group sum follows ascending group id as the existing oracle
# route's deterministic group order (the exact kernel may need compensated
# summation to meet the frozen tolerance).
perm = np.concatenate([np.flatnonzero(gidx == g) for g in range(G)])
qgroup = q[perm].reshape(G, 128, N)
scale_group = scale
# For effective zero 8, signed values [-8,7] are exactly representable in FP16.
q_signed_half = (qgroup - 8).astype(np.float16).astype(np.float64)
# Simulate FP16 MMA A/B inputs then FP32 accumulation; multiplication is exact
# in Python float64 after operand rounding. Scale remains FP32 post-MMA.
def evaluate(xh, label):
    x = xh.astype(np.float64)
    ref = (x @ w64).astype(np.float32).astype(np.float64)
    xgroup = x[:, perm].reshape(M, G, 128)
    partial = np.empty((M, G, N), np.float64)
    for m_idx in range(M):
        for g in range(G):
            a = xgroup[m_idx, g].astype(np.float16).astype(np.float64)
            partial[m_idx, g] = a @ q_signed_half[g]
    scaled = partial * scale_group[None, :, :]
    y = np.zeros((M, N), np.float64)
    for g in range(G):
        y += scaled[:, g, :]
    y32 = y.astype(np.float32).astype(np.float64)
    lim = 1e-4 + 5e-5 * np.abs(ref)
    err = np.abs(y32 - ref)
    ratio = err / lim
    bad = np.argwhere(err > lim)
    order = np.argsort(ratio.reshape(-1))[-10:][::-1]
    coords = []
    for ix in order:
        r, c = np.unravel_index(ix, ratio.shape)
        coords.append({"m": int(r), "n": int(c), "actual": float(y32[r,c]), "reference": float(ref[r,c]), "error": float(err[r,c]), "allowed": float(lim[r,c]), "ratio": float(ratio[r,c])})
    return {
        "input": label, "strict_bad": int(len(bad)), "total_outputs": int(ref.size),
        "max_abs_error": float(err.max()), "max_tolerance_ratio": float(ratio.max()),
        "ref_absmax": float(np.abs(ref).max()), "x_absmax": float(np.abs(x).max()),
        "signed_integer_fp16_exact": bool(np.array_equal(q_signed_half, qgroup-8)),
        "top10": coords,
    }

out = {
    "fixture": m["prefix"], "manifest_sha256": hashlib.sha256((ROOT/"manifest.json").read_bytes()).hexdigest(),
    "method": "FP64 oracle vs stable-group QPN-style FP16 (q-8), FP16 activation, FP32 exact dot partial, FP32 independent scale and ascending FP32 group sum; no GPU MMA rounding/order",
    "cases": [evaluate(xh, "ordinary"), evaluate(stress, "4x_saved_stress")],
}
print(json.dumps(out, indent=2))
