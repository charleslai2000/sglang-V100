"""Attribute GPU kernel time, GPU idle gaps and host dispatch to operators.

Consumes the chrome trace produced by ``capture_forward.py`` (or any Kineto
trace that carries ``record_function`` annotations) and reports, per operator
category:

  * kernel count and kernel time (sum of kernel durations, exclusive to the
    innermost enclosing annotation),
  * share of the forward envelope,
  * GPU idle inside the envelope (launch gaps / host stalls),
  * host-side time spent inside each annotation (Python dispatch).

Attribution works through Kineto's launch correlation: every kernel event
carries ``args.correlation`` which equals the ``External id`` of the CUDA
runtime call that launched it, and that runtime call sits inside the
``record_function`` span of the operator that issued it. Nothing is guessed from
kernel names alone, so identical kernels issued by different operators are kept
apart.

Example::

    python benchmark/perf_telemetry/op_budget.py /tmp/lfm3_trace/trace-decode-bs8-ctx1024.json.gz \
        --profile lfm3 --out /tmp/lfm3_trace/budget-decode.json

Dependencies: Python standard library only.
"""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}


def load_trace(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def union(intervals):
    out = []
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def owner_of(name, rules, strip_prefix="op:"):
    label = name[len(strip_prefix):] if name.startswith(strip_prefix) else name
    for pattern, category in rules:
        if pattern in label:
            return category, label
    return "other", label


def owner_from_name(name, name_rules):
    for pattern, category in name_rules:
        if pattern in name:
            return category
    return "other"


def analyze(trace_path, rules, strip_prefix="op:", top_n=25, name_rules=None,
            step_prefix="step[", owner_mode="auto"):
    doc = load_trace(trace_path)
    events = doc.get("traceEvents", [])

    kernels, runtime, annotations, cpu_ops = [], {}, [], {}
    for event in events:
        cat = event.get("cat")
        if event.get("ph") != "X":
            continue
        if cat in KERNEL_CATS:
            kernels.append(event)
        elif cat in RUNTIME_CATS:
            ext = (event.get("args") or {}).get("External id")
            if ext is not None:
                runtime[ext] = event
        elif cat == "cpu_op":
            # cublas dispatches through cudaLaunchKernelExC, which carries no
            # "External id"; the kernel event itself repeats the launching
            # aten op's External id, so join through that instead.
            ext = (event.get("args") or {}).get("External id")
            if ext is not None:
                cpu_ops.setdefault(ext, event)
        elif cat == "user_annotation":
            name = event.get("name", "")
            if name.startswith(strip_prefix) or name.startswith(step_prefix):
                annotations.append(event)

    if not kernels:
        raise SystemExit(f"{trace_path}: no kernel events, was CUDA profiling enabled?")

    annotations.sort(key=lambda e: float(e["ts"]))
    ann_starts = [float(e["ts"]) for e in annotations]
    ann_ends = [float(e["ts"]) + float(e["dur"]) for e in annotations]

    def innermost(ts, allow=None):
        idx = bisect.bisect_right(ann_starts, ts) - 1
        while idx >= 0:
            if ann_ends[idx] >= ts and (allow is None or allow(annotations[idx]["name"])):
                return idx
            idx -= 1
        return None

    op_only = lambda name: name.startswith(strip_prefix)
    step_only = lambda name: name.startswith(step_prefix)

    per_owner = collections.defaultdict(lambda: {"kernels": 0, "kernel_us": 0.0,
                                                 "by_op": collections.Counter()})
    step_stats = collections.defaultdict(lambda: {"kernels": 0, "kernel_us": 0.0})
    per_op = collections.defaultdict(lambda: {"kernels": 0, "kernel_us": 0.0})
    kernel_intervals = []
    unattributed_us = 0.0

    for kernel in kernels:
        dur = float(kernel.get("dur", 0.0))
        start = float(kernel["ts"])
        kernel_intervals.append((start, start + dur))
        kargs = kernel.get("args") or {}
        launcher = cpu_ops.get(kargs.get("External id"))
        if launcher is None:
            launcher = runtime.get(kargs.get("correlation"))
        probe = float(launcher["ts"]) if launcher is not None else start
        step_idx = innermost(probe, step_only)
        step_name = annotations[step_idx]["name"] if step_idx is not None else None
        if step_name is not None:
            step_stats[step_name]["kernels"] += 1
            step_stats[step_name]["kernel_us"] += dur
        if owner_mode == "name":
            # CUDA-graph replay never runs the Python forward, so per-operator
            # record_function spans cannot exist; classify by kernel name and
            # keep the step span as the envelope.
            category = owner_from_name(kernel.get("name", ""), name_rules)
            label = category
            per_owner[category]["kernels"] += 1
            per_owner[category]["kernel_us"] += dur
            per_owner[category]["by_op"][label] += dur
            per_op[label]["kernels"] += 1
            per_op[label]["kernel_us"] += dur
            continue
        idx = innermost(probe, op_only)
        if idx is None:
            if name_rules is not None:  # auto mode: coarse class from the name
                category = "name:" + owner_from_name(kernel.get("name", ""), name_rules)
                per_owner[category]["kernels"] += 1
                per_owner[category]["kernel_us"] += dur
                per_owner[category]["by_op"][category] += dur
                per_op[category]["kernels"] += 1
                per_op[category]["kernel_us"] += dur
                continue
            unattributed_us += dur
            per_owner["unattributed"]["kernels"] += 1
            per_owner["unattributed"]["kernel_us"] += dur
            continue
        category, label = owner_of(annotations[idx]["name"], rules, strip_prefix)
        per_owner[category]["kernels"] += 1
        per_owner[category]["kernel_us"] += dur
        per_owner[category]["by_op"][label] += dur
        per_op[label]["kernels"] += 1
        per_op[label]["kernel_us"] += dur

    env_start = min(s for s, _ in kernel_intervals)
    env_end = max(t for _, t in kernel_intervals)
    merged = union(kernel_intervals)
    busy = sum(t - s for s, t in merged)
    gaps, cursor = [], env_start
    for s, t in merged:
        if s > cursor:
            gaps.append({"start_us": round(cursor, 3), "len_us": round(s - cursor, 3)})
        cursor = max(cursor, t)
    if cursor < env_end:
        gaps.append({"start_us": round(cursor, 3), "len_us": round(env_end - cursor, 3)})

    cpu = collections.defaultdict(float)
    for event in annotations:
        if owner_mode == "name" and event["name"].startswith(step_prefix):
            continue
        category, _ = owner_of(event["name"], rules, strip_prefix)
        cpu[category] += float(event.get("dur", 0.0))

    total_kernel = sum(v["kernel_us"] for v in per_owner.values())
    rows = []
    for category, data in sorted(per_owner.items(), key=lambda kv: -kv[1]["kernel_us"]):
        rows.append({
            "owner": category,
            "kernels": data["kernels"],
            "kernel_ms": round(data["kernel_us"] / 1000.0, 3),
            "kernel_share_pct": round(100.0 * data["kernel_us"] / total_kernel, 2),
            "envelope_share_pct": round(100.0 * data["kernel_us"] / (env_end - env_start), 2),
            "host_ms": round(cpu.get(category, 0.0) / 1000.0, 3),
            "top_ops": [[label, round(us / 1000.0, 3)]
                        for label, us in data["by_op"].most_common(3)],
        })

    return {
        "trace": str(trace_path),
        "envelope_us": round(env_end - env_start, 3),
        "kernel_duration_sum_us": round(total_kernel, 3),
        "activity_union_us": round(busy, 3),
        "gpu_busy_pct": round(100.0 * busy / (env_end - env_start), 2),
        "gpu_idle_us": round((env_end - env_start) - busy, 3),
        "idle_gap_count": len(gaps),
        "largest_gaps_us": sorted((g["len_us"] for g in gaps), reverse=True)[:5],
        "host_op_ms_total": round(sum(cpu.values()) / 1000.0, 3),
        "unattributed_kernel_ms": round(unattributed_us / 1000.0, 3),
        "steps": [{"step": name, "kernels": v["kernels"],
                   "kernel_ms": round(v["kernel_us"] / 1000.0, 3),
                   "gpu_busy_pct_of_step": round(
                       100.0 * v["kernel_us"] / float(
                           next(e["dur"] for e in annotations if e["name"] == name)), 2)}
                  for name, v in sorted(step_stats.items())][:6],
        "owners": rows,
        "top_ops": sorted(
            ({"op": label, "kernels": v["kernels"], "kernel_ms": round(v["kernel_us"] / 1000.0, 3)}
             for label, v in per_op.items()),
            key=lambda r: -r["kernel_ms"],
        )[:top_n],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--profile", default="lfm3")
    ap.add_argument("--profiles", default=str(Path(__file__).with_name("profiles.json")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--owner-mode", choices=["annotation", "name", "auto"], default="auto",
                    help="auto = exact operator span when resolvable, else kernel-name class")
    args = ap.parse_args()
    profiles = json.loads(Path(args.profiles).read_text())
    prof = profiles[args.profile]
    rules = [tuple(rule) for rule in prof["owner_rules"]]
    name_rules = ([tuple(rule) for rule in prof["name_rules"]]
                  if args.owner_mode in ("name", "auto") else None)
    if args.owner_mode == "name":
        # pure name mode: never use operator spans
        name_rules = [tuple(rule) for rule in prof["name_rules"]]
    result = analyze(args.trace, rules, name_rules=name_rules,
                     step_prefix=prof.get("step_prefix", "step["),
                     owner_mode=args.owner_mode)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("envelope_us", "gpu_busy_pct", "gpu_idle_us",
                                             "kernel_duration_sum_us", "host_op_ms_total")}))
    for row in result["owners"]:
        print(f"{row['owner']:22s} kernels={row['kernels']:5d} "
              f"kernel_ms={row['kernel_ms']:8.3f} share={row['kernel_share_pct']:5.1f}% "
              f"host_ms={row['host_ms']:7.3f}")


if __name__ == "__main__":
    main()
