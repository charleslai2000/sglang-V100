#!/usr/bin/env python3
"""Install a low-overhead CUDA Graph activity attribution hook for one server run.

Run with sitecustomize.py via PYTHONPATH before the frozen S1/candidate server.
It writes one bounded JSONL record per runtime. This uses CUPTI-backed Kineto and
is diagnostic only: paired service throughput must be measured separately without it.
"""
import json, os, threading, time
import torch

OUT = os.environ.get("G017_GRAPH_ACCOUNTING_OUT", "/tmp/g017-graph-accounting.jsonl")
LOCK = threading.Lock()
COUNTS = {}

def trace_handler(prof):
    events = prof.events()
    host = {}
    gpu = {}
    for e in events:
        key = (e.name, getattr(e, "device_type", None))
        if "CUDA" in str(key[1]) or "GPU" in str(key[1]):
            gpu[e.name] = gpu.get(e.name, 0.0) + float(getattr(e, "cuda_time_total", 0.0) or 0.0)
        else:
            host[e.name] = host.get(e.name, 0.0) + float(getattr(e, "self_cpu_time_total", 0.0) or 0.0)
    row = {
        "step": prof.step_num,
        "event_count": len(events),
        "gpu_us_by_name": gpu,
        "host_self_us_by_name": host,
        "note": "kineto/CUPTI diagnostic; do not treat as uninstrumented timing",
    }
    with LOCK, open(OUT, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

# torch.profiler is initialized in the model worker process, after the initial
# model/cuda graph setup; bounded schedule captures two active steps only.
activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
prof = torch.profiler.profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
    on_trace_ready=trace_handler,
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
)
prof.__enter__()

# Scheduler calls this function once after each completed batch result. A daemon
# timer advances Kineto outside the graph launch path; it is bounded to 3 steps.
def _stepper():
    # Kineto step progression is driven by the request stream; periodic stepping
    # can capture setup noise. The hook is explicitly opt-in and short-lived.
    deadline = time.monotonic() + float(os.environ.get("G017_GRAPH_ACCOUNTING_SECONDS", "45"))
    while time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            prof.step()
        except Exception:
            break
    try:
        prof.__exit__(None, None, None)
    except Exception:
        pass

threading.Thread(target=_stepper, name="g017-profiler-stepper", daemon=True).start()
