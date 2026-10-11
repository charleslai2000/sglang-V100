#!/usr/bin/env python3
"""Bounded decoder for the local fork's FPM v1 schema.

Use the exact logged endpoint, including '.<dp_rank>'. Bind is PUB-side.
"""
import json, signal, sys, time
import zmq
from sglang.srt.observability.forward_pass_metrics import decode
endpoint, output, duration = sys.argv[1], sys.argv[2], float(sys.argv[3])
ctx=zmq.Context(); sub=ctx.socket(zmq.SUB); sub.setsockopt(zmq.SUBSCRIBE,b""); sub.setsockopt(zmq.RCVTIMEO,1000); sub.connect(endpoint)
rows=[]; stopping=False
def stop(*_):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
end=time.monotonic()+duration
while not stopping and time.monotonic()<end:
 try: parts=sub.recv_multipart()
 except zmq.error.Again: continue
 row={'received_monotonic_s':time.monotonic(),'part_count':len(parts)}
 if len(parts)==3:
  topic,seq,payload=parts; row.update(topic_hex=topic.hex(),seq=int.from_bytes(seq,'big'))
  try:
   v=decode(payload); row.update(version=v.version,worker_id=v.worker_id,dp_rank=v.dp_rank,counter_id=v.counter_id,gpu_wall_s=v.wall_time,
     scheduled={k:getattr(v.scheduled_requests,k) for k in v.scheduled_requests.__struct_fields__},
     queued={k:getattr(v.queued_requests,k) for k in v.queued_requests.__struct_fields__})
  except Exception as exc: row['decode_error']=repr(exc)
 rows.append(row)
with open(output,'w') as f: json.dump({'endpoint':endpoint,'count':len(rows),'records':rows},f,indent=2)
sub.close(0);ctx.term();print(json.dumps({'endpoint':endpoint,'count':len(rows),'output':output}))
