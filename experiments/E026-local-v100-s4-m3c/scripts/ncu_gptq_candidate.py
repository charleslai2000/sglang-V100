#!/usr/bin/env python3
"""One-launch NCU target for the production S1 fallback or G017 candidate."""
import hashlib, json, os, sys
from pathlib import Path
import torch
from importlib.machinery import ExtensionFileLoader

mode, fixture = sys.argv[1], Path(sys.argv[2])
module = os.environ["SGLANG_G017_CANDIDATE_PATH"]
loaded = ExtensionFileLoader("common_ops", module).load_module()
manifest = json.loads((fixture / "manifest.json").read_text())
def tensor(name, dtype):
    info = manifest["tensors"][name]
    path = fixture / f"{name}.bin"
    return torch.from_file(str(path), shared=False,
                           size=path.stat().st_size // torch.empty((), dtype=dtype).element_size(),
                           dtype=dtype).view(*info["shape"]).cuda()
x = tensor("x", torch.float16)
w = tensor("qweight", torch.int32)
z = tensor("qzeros", torch.int32)
s = tensor("scales", torch.float16)
g = tensor("g_idx", torch.int32)
os.environ["SGLANG_G017_Q4_FP32_CANDIDATE"] = "1" if mode == "candidate" else "0"
os.environ["SGLANG_G017_Q4_FP32_CANDIDATE_PROBE"] = "0"
def op():
    if mode == "candidate":
        return torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x, w, z, s, g, False)
    return torch.ops.sgl_kernel.gptq_gemm(x, w.view(torch.uint32), z.view(torch.uint32), s, g, False, 4)
y = op(); torch.cuda.synchronize()
assert torch.isfinite(y).all()
for _ in range(5): op()
torch.cuda.synchronize()
print(json.dumps({"route": mode, "fixture": manifest["prefix"], "shape": list(y.shape),
                  "finite": True, "module_path": loaded.__file__,
                  "module_sha256": hashlib.sha256(Path(loaded.__file__).read_bytes()).hexdigest()}), flush=True)
for _ in range(100): op()
torch.cuda.synchronize()
