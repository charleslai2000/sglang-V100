"""Export chrome traces (CUDA) for decode bs=8 and prefill 64 tokens, plus busy/idle stats."""
import json
import torch
from sglang.bench_one_batch import _set_envs_and_config, load_model, prepare_synthetic_inputs_for_latency_test
from sglang.srt.entrypoints.engine import PortArgs
from sglang.srt.server_args import ServerArgs


def summarize(path):
    data = json.load(open(path))
    events = data["traceEvents"]
    # kernel intervals on any CUDA stream
    k = []
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_user_annotation", "Kernel"):
            ts = e["ts"]
            dur = e.get("dur", 0)
            if dur > 0:
                k.append((ts, ts + dur))
    if not k:
        return None
    k.sort()
    merged = []
    for s, t in k:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t)
        else:
            merged.append([s, t])
    busy = sum(t - s for s, t in merged)
    span = merged[-1][1] - merged[0][0]
    return {
        "kernels": len(k),
        "span_ms": span / 1000,
        "gpu_busy_ms": busy / 1000,
        "gpu_busy_pct": 100 * busy / span,
        "gpu_idle_ms": (span - busy) / 1000,
    }


def main():
    sa = ServerArgs(model_path="/var/tmp/lfm3", trust_remote_code=True, dtype="float16",
        attention_backend="triton", sampling_backend="pytorch", mem_fraction_static=0.60,
        cuda_graph_max_bs=32, max_running_requests=32, context_length=8448,
        mamba_scheduler_strategy="extra_buffer")
    _set_envs_and_config(sa)
    runner, _ = load_model(sa, PortArgs.init_new(sa), 0, 0)

    out = {}
    # --- prefill 64 tokens x3 ---
    for rep in range(3):
        reqs = prepare_synthetic_inputs_for_latency_test(1, 64)
        for i, r in enumerate(reqs):
            r.rid = f"pw{rep}-{i}"
        runner.extend(reqs)
    torch.cuda.synchronize()

    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    prof.start()
    for rep in range(3):
        reqs = prepare_synthetic_inputs_for_latency_test(1, 64)
        for i, r in enumerate(reqs):
            r.rid = f"pf{rep}-{i}"
        runner.extend(reqs)
    torch.cuda.synchronize()
    prof.stop()
    prof.export_chrome_trace("/var/tmp/m2_trace_prefill64.json")
    out["prefill64"] = summarize("/var/tmp/m2_trace_prefill64.json")

    # --- decode bs=8 @1024 ---
    reqs = prepare_synthetic_inputs_for_latency_test(8, 1024)
    for i, r in enumerate(reqs):
        r.rid = f"d-{i}"
    ids, _, batch = runner.extend(reqs)
    for _ in range(5):
        ids, _ = runner.decode(ids, batch)
    torch.cuda.synchronize()

    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    prof.start()
    for _ in range(10):
        ids, _ = runner.decode(ids, batch)
    torch.cuda.synchronize()
    prof.stop()
    prof.export_chrome_trace("/var/tmp/m2_trace_decode_bs8.json")
    out["decode_bs8"] = summarize("/var/tmp/m2_trace_decode_bs8.json")
    print("TRACESTATS", json.dumps(out, indent=None), flush=True)


if __name__ == "__main__":
    main()
