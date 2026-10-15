#!/usr/bin/env python3
"""Consume the existing service FPM ZMQ endpoint during the matched service matrix."""
import json, os, signal, sys, time
import zmq
from sglang.srt.observability.forward_pass_metrics import decode
endpoint,out,duration=sys.argv[1],sys.argv[2],float(sys.argv[3]);ctx=zmq.Context();s=ctx.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b"");s.setsockopt(zmq.RCVTIMEO,1000);s.connect(endpoint)
rows=[];end=time.monotonic()+duration
while time.monotonic()<end:
 try:p=s.recv_multipart()
 except zmq.error.Again:continue
 if len(p)!=3:continue
 v=decode(p[2]);z=v.scheduled_requests;q=v.queued_requests
 rows.append({'counter_id':v.counter_id,'scheduler_wall_s':v.wall_time,'prefill_requests':z.num_prefill_requests,'prefill_tokens':z.sum_prefill_tokens,'prefill_kv_tokens':z.sum_prefill_kv_tokens,'decode_requests_actual_batch':z.num_decode_requests,'decode_kv_tokens':z.sum_decode_kv_tokens,'queued_prefill':q.num_prefill_requests,'queued_decode':q.num_decode_requests})
with open(out,'w') as f:json.dump({'endpoint':endpoint,'rows':rows},f,indent=2)
print(json.dumps({'out':out,'rows':len(rows)}))
