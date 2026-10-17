#!/usr/bin/env python3
"""Deterministic real-fixture target for exact-c57 NCU application replay.

Loads only the exact existing extension, one real Falcon GPTQ fixture and
executes one selected M1 or M4 projection. Prints a stable launch/identity
manifest on every application replay. It is not a timing benchmark.
"""
import hashlib
import importlib.machinery
import json
import os
import sys
from pathlib import Path

import torch

UUID = "GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189"
C57 = Path("/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-c57-runtime/sgl_kernel/sm70/common_ops.abi3.so")
C57_SHA = "c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9"
FIXROOT = Path("/mnt/data/pc01/sglang-v100-migration/remote-evidence/sglang-logs/s4-m3a-fixtures")

if len(sys.argv) != 4 or sys.argv[1] not in ("gate_up", "down") or int(sys.argv[2]) not in (1, 4):
    raise SystemExit("usage: ncu_fixture_target.py {gate_up|down} {1|4} {warmup|measure}")
projection, rows, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if mode not in ("warmup", "measure", "bench"):
    raise SystemExit("mode must be warmup, measure or bench")
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
# sudo's approved NCU wrapper may sanitize this variable. The authoritative
# check below is the CUDA runtime device UUID, after NCU's physical-UUID bind.
if visible not in (None, UUID, "0"):
    raise SystemExit(f"unexpected CUDA_VISIBLE_DEVICES value: {visible!r}")
if hashlib.sha256(C57.read_bytes()).hexdigest() != C57_SHA:
    raise SystemExit("exact c57 extension SHA mismatch")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible GPU, got {torch.cuda.device_count()}")
actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower()
if actual_uuid != UUID.lower().removeprefix("gpu-"):
    raise SystemExit(f"CUDA UUID mismatch: {actual_uuid}")

fixture = FIXROOT / ("43-mlp-gate_up_proj-call0000" if projection == "gate_up" else "43-mlp-down_proj-call0000")
manifest_path = fixture / "manifest.json"
manifest = json.loads(manifest_path.read_text())
shapes = {
    "gate_up": ((4, 3072), (384, 24576), (24, 3072), (24, 24576), 3072),
    "down": ((4, 12288), (1536, 3072), (96, 384), (96, 3072), 12288),
}
xshape, qwshape, qzshape, scaleshape, K = shapes[projection]
assert tuple(manifest["tensors"]["x"]["shape"]) == xshape
assert tuple(manifest["tensors"]["qweight"]["shape"]) == qwshape
assert tuple(manifest["tensors"]["qzeros"]["shape"]) == qzshape
assert tuple(manifest["tensors"]["scales"]["shape"]) == scaleshape
assert tuple(manifest["tensors"]["g_idx"]["shape"]) == (K,)

def load(name, dtype):
    meta = manifest["tensors"][name]
    path = fixture / f"{name}.bin"
    expected_sha = meta.get("sha256")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise SystemExit(f"fixture hash mismatch {name}: {actual_sha} != {expected_sha}")
    count = path.stat().st_size // torch.empty((), dtype=dtype).element_size()
    return torch.from_file(str(path), shared=False, size=count, dtype=dtype).view(*meta["shape"]).cuda().contiguous()

x = load("x", torch.float16)[:rows].contiguous()
qw = load("qweight", torch.int32)
qz = load("qzeros", torch.int32)
scales = load("scales", torch.float16)
gidx = load("g_idx", torch.int32).view(-1)
if tuple(x.shape) != (rows, K):
    raise SystemExit(f"activation geometry mismatch: {tuple(x.shape)}")

mod = importlib.machinery.ExtensionFileLoader("common_ops", str(C57)).load_module()
os.environ["SGLANG_G017_Q4_FP32_CANDIDATE"] = "0"
os.environ["SGLANG_G017_Q4_FP32_CANDIDATE_PROBE"] = "0"
# The opt-in extension exposes c57's exact packed_fp32_partial/reducer route.
fn = torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate
out = fn(x, qw, qz, scales, gidx, False)
torch.cuda.synchronize()
if out.dtype != torch.float16 or tuple(out.shape) != (rows, qwshape[1]) or not torch.isfinite(out).all():
    raise SystemExit("bad exact-c57 output dtype/shape/finite check")
if mode == "bench":
    fn32 = torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate
    # Warmup and collect uninstrumented CUDA-event complete-operator samples.
    for _ in range(12):
        fn32(x, qw, qz, scales, gidx, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gy = fn32(x, qw, qz, scales, gidx, False)
    graph.replay(); torch.cuda.synchronize()
    def samples(call, count=60):
        out = []
        for _ in range(count):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); call(); end.record(); end.synchronize()
            out.append(start.elapsed_time(end) * 1000.0)
        return sorted(out)
    eager = samples(lambda: fn32(x, qw, qz, scales, gidx, False))
    graph_samples = samples(graph.replay)
    print(json.dumps({"event":"E040_UNINSTRUMENTED_C57_COMPLETE_LINEAR", "projection":projection,"M":rows,"K":K,"N":qwshape[1],"extension_sha256":C57_SHA,"eager_samples_us":eager,"graph_samples_us":graph_samples,"eager_median_us":eager[len(eager)//2],"graph_median_us":graph_samples[len(graph_samples)//2],"graph_output_sha256":hashlib.sha256(gy.cpu().contiguous().numpy().tobytes()).hexdigest()},sort_keys=True),flush=True)
    raise SystemExit(0)
if mode == "measure":
    print(json.dumps({
        "event": "E040_EXACT_C57_LAUNCH_READY",
        "projection": projection, "M": rows, "K": K, "N": qwshape[1],
        "grid_partial": [qwshape[1] // 128, 1, K // 128],
        "block_partial": [32, 1, 1], "dtype": str(out.dtype),
        "torch_sha": hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        "extension": str(C57), "extension_sha256": C57_SHA,
        "fixture": str(fixture), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "x_sha256": hashlib.sha256((fixture / "x.bin").read_bytes()).hexdigest(),
        "qweight_sha256": hashlib.sha256((fixture / "qweight.bin").read_bytes()).hexdigest(),
        "qzeros_sha256": hashlib.sha256((fixture / "qzeros.bin").read_bytes()).hexdigest(),
        "scales_sha256": hashlib.sha256((fixture / "scales.bin").read_bytes()).hexdigest(),
        "g_idx_sha256": hashlib.sha256((fixture / "g_idx.bin").read_bytes()).hexdigest(),
        "out_sha256": hashlib.sha256(out.cpu().contiguous().numpy().tobytes()).hexdigest(),
        "max_abs": float(out.float().abs().max().item()),
    }, sort_keys=True), flush=True)
