#!/usr/bin/env python3
"""Capture a bounded window of SGLang ForwardPassMetrics from its local ZMQ IPC."""
import json, os, sys, time
import zmq
endpoint = sys.argv[1]
out = sys.argv[2]
ctx = zmq.Context(); sub = ctx.socket(zmq.SUB); sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(endpoint)
# PUB/SUB slow joiner window
sub.setsockopt(zmq.RCVTIMEO, 30000)
from sglang.srt.observability.forward_pass_metrics import decode
rows=[]; deadline=time.monotonic()+float(os.environ.get('G017_FPM_SECONDS','90'))
while time.monotonic()<deadline:
 try: parts=sub.recv_multipart()
 except zmq.error.Again: continue
 if len(parts)!=3: continue
 topic,seq_bytes,data=parts
 try:
  v=decode(data)
  rows.append({'version':v.version,'counter_id':v.counter_id,'worker_id':v.worker_id,'dp_rank':v.dp_rank,'gpu_forward_s':v.wall_time,'scheduled':{'decode_reqs':v.scheduled_requests.num_decode_requests,'decode_kv_tokens':v.scheduled_requests.sum_decode_kv_tokens,'prefill_reqs':v.scheduled_requests.num_prefill_requests,'prefill_tokens':v.scheduled_requests.sum_prefill_tokens}})
 except Exception as e: rows.append({'decode_error':str(e),'bytes':len(data)})
with open(out,'w') as f: json.dump({'endpoint':endpoint,'records':rows},f,indent=2)
sub.close(0);ctx.term();print(json.dumps({'records':len(rows),'out':out}))
