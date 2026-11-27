"""Capture a Kineto trace of real SGLang forward passes with per-operator spans.

This is the reusable capture half of the GPU telemetry toolkit: it loads the
model in-process (same code path as ``sglang.bench_one_batch``), wraps selected
submodules with ``record_function`` spans via forward hooks, runs real prefill or
decode forwards and exports a chrome trace that ``op_budget.py`` can attribute.

No model code is modified: hooks are registered only for the capture run.

Example (LFM3, decode batch 8 at 1024 context)::

    PYTHONPATH=python python benchmark/perf_telemetry/capture_forward.py \
        --model-path /var/tmp/lfm3 --profile lfm3 --mode decode \
        --batch-size 8 --input-len 1024 --steps 10 --out-dir /tmp/lfm3_trace

Dependencies: torch with CUDA (Kineto), triton, SGLang importable via PYTHONPATH.
See README.md for the metric contract and what this does not measure.
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import torch


def load_profile(path, name):
    doc = json.loads(Path(path).read_text())
    return doc[name]


def register_hooks(model, patterns, prefix="op:"):
    """Wrap every submodule whose qualified name matches one of ``patterns``."""
    handles = []
    matched = []

    import re

    regexes = [re.compile(p) for p in patterns]
    for mod_name, module in model.named_modules():
        if not mod_name:
            continue
        if not any(r.search(mod_name) for r in regexes):
            continue
        label = f"{prefix}{mod_name}"

        def hook(_mod, _inp, _out, label=label):
            # Emitted around the child calls: the enter/exit pair brackets this
            # module's own kernels plus its already-hooked children.
            return None

        def pre(_mod, _inp, label=label):
            ctx = torch.autograd.profiler.record_function(label)
            ctx.__enter__()
            _mod._pf_ctx = ctx

        def post(_mod, _inp, _out):
            ctx = getattr(_mod, "_pf_ctx", None)
            if ctx is not None:
                ctx.__exit__(None, None, None)
                _mod._pf_ctx = None

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
        matched.append(mod_name)
    return handles, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--profile", default="lfm3", help="key inside --profiles")
    ap.add_argument("--profiles", default=str(Path(__file__).with_name("profiles.json")))
    ap.add_argument("--mode", choices=["decode", "prefill"], required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--attention-backend", default="triton")
    ap.add_argument("--mem-fraction-static", type=float, default=0.60)
    ap.add_argument("--cuda-graph-max-bs", type=int, default=32)
    ap.add_argument("--context-length", type=int, default=8448)
    ap.add_argument("--max-running-requests", type=int, default=32)
    ap.add_argument("--mamba-scheduler-strategy", default="extra_buffer")
    ap.add_argument("--disable-cuda-graph", action="store_true")
    args = ap.parse_args()

    # Imported late so --help works without a CUDA device.
    from sglang.bench_one_batch import (
        _set_envs_and_config,
        load_model,
        prepare_synthetic_inputs_for_latency_test,
    )
    from sglang.srt.entrypoints.engine import PortArgs
    from sglang.srt.server_args import ServerArgs

    profile = load_profile(args.profiles, args.profile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    server_args = ServerArgs(
        model_path=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        sampling_backend="pytorch",
        mem_fraction_static=args.mem_fraction_static,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_running_requests=args.max_running_requests,
        context_length=args.context_length,
        mamba_scheduler_strategy=args.mamba_scheduler_strategy,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    _set_envs_and_config(server_args)
    runner, _ = load_model(server_args, PortArgs.init_new(server_args), 0, 0)
    model = runner.torch_runner.model
    handles, matched = register_hooks(model, profile["hook_patterns"])

    reqs = prepare_synthetic_inputs_for_latency_test(args.batch_size, args.input_len)
    for i, req in enumerate(reqs):
        req.rid = f"trace-{i}"

    meta = {
        "profile": args.profile,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "steps": args.steps,
        "hooked_module_count": len(matched),
        "hooked_modules": matched[:12],
        "cuda_graph": not args.disable_cuda_graph,
        "attention_backend": args.attention_backend,
        "server_args": {
            "mem_fraction_static": args.mem_fraction_static,
            "context_length": args.context_length,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "mamba_scheduler_strategy": args.mamba_scheduler_strategy,
        },
        "torch": torch.__version__,
    }

    warm_until = time.time()
    ids, _, batch = runner.extend(reqs)
    warm_steps = 5 if args.mode == "decode" else 0
    for _ in range(warm_steps):
        ids, _ = runner.decode(ids, batch)
    torch.cuda.synchronize()
    meta["warmup_s"] = round(time.time() - warm_until, 3)

    # CPU + CUDA: the CPU side carries the record_function spans that
    # op_budget.py joins to kernels through launch correlation.
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        if args.mode == "decode":
            for _ in range(args.steps):
                ids, _ = runner.decode(ids, batch)
        else:
            for _ in range(args.steps):
                reqs = prepare_synthetic_inputs_for_latency_test(args.batch_size, args.input_len)
                for i, req in enumerate(reqs):
                    req.rid = f"trace-p{i}"
                runner.extend(reqs)
        torch.cuda.synchronize()

    tag = args.label or f"{args.mode}-bs{args.batch_size}-ctx{args.input_len}"
    trace_path = out_dir / f"trace-{tag}.json.gz"
    tmp = out_dir / f"trace-{tag}.json"
    prof.export_chrome_trace(str(tmp))
    with open(tmp, "rb") as src, gzip.open(trace_path, "wb", compresslevel=9) as dst:
        dst.write(src.read())
    tmp.unlink()
    meta["trace"] = str(trace_path)
    meta["trace_gz_bytes"] = trace_path.stat().st_size
    (out_dir / f"meta-{tag}.json").write_text(json.dumps(meta, indent=2))

    for handle in handles:
        handle.remove()
    summary = {"trace": str(trace_path), "meta": str(out_dir / f'meta-{tag}.json'),
               "hooked": len(matched), "hooked_modules": matched[:6]}
    (out_dir / f"capture-{tag}.log").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    os._exit(0)  # avoid the profiler/SGLang teardown segfault seen on SM70


if __name__ == "__main__":
    main()
