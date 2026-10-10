#!/usr/bin/env python3
"""Bounded subscription to the scheduler's local ForwardPassMetrics ZMQ IPC."""
import json, os, sys, time, zmq
from sglang.srt.observability.forward_pass_metrics import decode
endpoint,out=sys.argv[1:3]; ctx=zmq.Context(); sub=ctx.socket(zmq.SUB); sub.setsockopt(zmq.SUBSCRIBE,b""); sub.setsockopt(zmq.RCVTIMEO,30000); sub.connect(endpoint)
rows=[]; deadline=time.monotonic()+float(os.environ.get('G017_FPM_SECONDS','120'))
while time.monotonic()<deadline:
 try: parts=sub.recv_multipart()
 except zmq.error.Again: continue
 if len(parts)!=3: continue
 topic,seq,payload=parts
 try:
  v=decode(payload); rows.append({'counter_id':v.counter_id,'gpu_forward_s':v.wall_time,'decode_reqs':v.scheduled_requests.num_decode_requests,'decode_kv_tokens':v.scheduled_requests.sum_decode_kv_tokens,'prefill_reqs':v.scheduled_requests.num_prefill_requests,'prefill_tokens':v.scheduled_requests.sum_prefill_tokens})
 except Exception as e: rows.append({'decode_error':str(e)})
with open(out,'w') as f:json.dump({'endpoint':endpoint,'records':rows},f,indent=2)
print(json.dumps({'records':len(rows),'out':out})); sub.close(0);ctx.term()
