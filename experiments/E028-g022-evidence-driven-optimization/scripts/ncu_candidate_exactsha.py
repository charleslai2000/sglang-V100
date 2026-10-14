#!/usr/bin/env python3
"""One matched-geometry L43 GPTQ M4 launch for root NCU profiling.

Run only under the V100-only MemAvailable supervisor. Exact extension path/SHA
is explicit: baseline uses frozen S1, candidate uses accepted G017 final SHA.
This is KERNEL_SCOPE NCU evidence, not a whole-Forward/service measurement.
"""
import hashlib
import importlib.machinery
import json
import os
import sys
from pathlib import Path

import torch

if len(sys.argv) != 6:
    raise SystemExit("usage: ncu_candidate_exactsha.py FIXTURE {s1|candidate} EXTENSION {gate_up|down} {1|2|4}")
fixture = Path(sys.argv[1])
route = sys.argv[2]
extension = Path(sys.argv[3]).resolve()
projection = sys.argv[4]
rows = int(sys.argv[5])
if rows not in (1, 2, 4):
    raise SystemExit("row count must be 1, 2, or 4")
expected = {
    "s1": "fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842",
    "candidate": "02f923ea7329912a4413f179f292acb444a5e001caebf110b24f531cabf70d8b",
}[route]
sha = hashlib.sha256(extension.read_bytes()).hexdigest()
if sha != expected:
    raise SystemExit(f"extension SHA mismatch: route={route} expected={expected} actual={sha}")
os.environ["SGLANG_G017_Q4_FP32_CANDIDATE"] = "1" if route == "candidate" else "0"
os.environ["SGLANG_G017_Q4_FP32_CANDIDATE_PROBE"] = "1" if route == "candidate" else "0"
mod = importlib.machinery.ExtensionFileLoader("common_ops", str(extension)).load_module()
manifest = json.loads((fixture / "manifest.json").read_text())
tensors = manifest["tensors"]

def load(name, dtype):
    meta = tensors[name]
    path = fixture / f"{name}.bin"
    n = path.stat().st_size // torch.empty((), dtype=dtype).element_size()
    return torch.from_file(str(path), shared=False, size=n, dtype=dtype).view(*meta["shape"]).cuda()

x = load("x", torch.float16)[:rows].contiguous()
qweight = load("qweight", torch.int32)
qzeros = load("qzeros", torch.int32)
scales = load("scales", torch.float16)
g_idx = load("g_idx", torch.int32)
expected = ((rows, 3072), (384, 24576)) if projection == "gate_up" else ((rows, 12288), (1536, 3072))
if (tuple(x.shape), tuple(qweight.shape)) != expected:
    raise SystemExit(f"not exact M4 {projection} geometry: x={tuple(x.shape)} qw={tuple(qweight.shape)}")
if route == "candidate":
    output = torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(
        x, qweight, qzeros, scales, g_idx, False
    )
else:
    output = torch.ops.sgl_kernel.gptq_gemm(
        x, qweight.view(torch.uint32), qzeros.view(torch.uint32), scales, g_idx, False, 4
    )
torch.cuda.synchronize()
if not torch.isfinite(output).all():
    raise SystemExit("non-finite GPTQ output")
print(json.dumps({
    "event": "G022_NCU_TARGET_READY",
    "route": route,
    "fixture": manifest["prefix"],
    "projection": projection,
    "selected_rows_from_captured_M4_operand": rows,
    "actual_mnk": [int(x.shape[0]), int(qweight.shape[1]), int(x.shape[1])],
    "extension_path": str(extension),
    "extension_sha256": sha,
    "output_dtype": str(output.dtype),
    "output_finite": True,
}, sort_keys=True), flush=True)
