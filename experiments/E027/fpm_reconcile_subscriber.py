#!/usr/bin/env python3
"""Bounded exact-endpoint SUB; writes READY only after the socket is connected."""
import json
import os
import sys
import time

import zmq
from sglang.srt.observability.forward_pass_metrics import decode

endpoint, out_path, ready_path = sys.argv[1:4]
duration_s = float(os.environ.get("FPM_CAPTURE_SECONDS", "120"))
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.setsockopt(zmq.RCVTIMEO, 250)
sub.connect(endpoint)
# A short settle allows the local IPC connection and SUB subscription to propagate.
time.sleep(0.5)
with open(ready_path, "w", encoding="utf-8") as f:
    json.dump({"endpoint": endpoint, "ready_monotonic_ns": time.monotonic_ns()}, f)
start_ns = time.monotonic_ns()
deadline = time.monotonic() + duration_s
rows = []
while time.monotonic() < deadline:
    try:
        parts = sub.recv_multipart()
    except zmq.error.Again:
        continue
    if len(parts) != 3:
        rows.append({"transport_malformed": True, "part_count": len(parts)})
        continue
    topic, seq_bytes, payload = parts
    try:
        message = decode(payload)
        rows.append({
            "transport_seq": int.from_bytes(seq_bytes, "big"),
            "counter_id": message.counter_id,
            "version": message.version,
            "worker_id": message.worker_id,
            "dp_rank": message.dp_rank,
            "wall_time_s": message.wall_time,
            "prefill_reqs": message.scheduled_requests.num_prefill_requests,
            "prefill_tokens": message.scheduled_requests.sum_prefill_tokens,
            "decode_reqs": message.scheduled_requests.num_decode_requests,
            "decode_kv_tokens": message.scheduled_requests.sum_decode_kv_tokens,
            "queued_prefill": message.queued_requests.num_prefill_requests,
            "queued_decode": message.queued_requests.num_decode_requests,
        })
    except Exception as e:
        rows.append({"decode_error": repr(e)})
end_ns = time.monotonic_ns()
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "endpoint": endpoint,
        "window_start_monotonic_ns": start_ns,
        "window_end_monotonic_ns": end_ns,
        "received_rows": len(rows),
        "rows": rows,
    }, f, indent=2)
print(json.dumps({"received": len(rows), "out": out_path, "ready": ready_path}))
sub.close(0)
ctx.term()
