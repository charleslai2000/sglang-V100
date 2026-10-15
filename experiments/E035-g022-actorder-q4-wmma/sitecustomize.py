"""Process-local runtime selector for isolated E035 servers and tests."""
import os
import sys
repo = "/home/charles/Workspaces/3rdparty/sglang-V100"
artifact = "/mnt/data/pc01/sglang-v100-migration"
mode = os.environ.get("E035_RUNTIME_MODE", "fp64")
if mode == "fp64":
    package_path = f"{repo}/sgl-kernel/python"
    os.environ["SGLANG_KERNEL_G017_S1_SHA256"] = "fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842"
    os.environ["SGLANG_G022_GPTQ_NUMERICAL_FIX"] = "fp64"
    os.environ["SGLANG_KERNEL_G022_NUMERICAL_FIX_PATH"] = f"{artifact}/experiments/E035/numerical-fix-fp64-optin-build/g022_fp64_optin.so"
    os.environ["SGLANG_G022_NUMERICAL_FIX_SHA256"] = "35998ff295f0a4fe27417886ff2fd86bc9a1c1030975435e8b34aa42fb26b5f6"
    for key in ("SGLANG_G017_Q4_FP32_CANDIDATE", "SGLANG_KERNEL_G017_CANDIDATE_PATH", "SGLANG_G017_CANDIDATE_SHA256"):
        os.environ.pop(key, None)
elif mode == "c57":
    package_path = f"{artifact}/experiments/E026/candidate-c57-runtime"
    os.environ["SGLANG_G017_Q4_FP32_CANDIDATE"] = "1"
    os.environ["SGLANG_KERNEL_G017_CANDIDATE_PATH"] = f"{package_path}/sgl_kernel/sm70/common_ops.abi3.so"
    os.environ["SGLANG_G017_CANDIDATE_SHA256"] = "c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9"
    for key in ("SGLANG_G022_GPTQ_NUMERICAL_FIX", "SGLANG_KERNEL_G022_NUMERICAL_FIX_PATH", "SGLANG_G022_NUMERICAL_FIX_SHA256"):
        os.environ.pop(key, None)
else:
    raise RuntimeError(f"unknown E035_RUNTIME_MODE {mode!r}")
for path in (f"{repo}/python", package_path):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
