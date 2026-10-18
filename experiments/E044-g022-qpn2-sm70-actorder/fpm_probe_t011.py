#!/usr/bin/env python3
"""Collect existing scheduler FPM via its versioned msgspec decoder."""
import dataclasses,json,os,time
import zmq
from sglang.srt.observability.forward_pass_metrics import decode
endpoint=os.environ.get('T011_FPM_ENDPOINT','ipc:///mnt/data/pc01/sglang-v100-migration/experiments/E044/t011-fpm-endpoint.0');seconds=float(os.environ.get('T011_FPM_SECONDS','90'));ctx=zmq.Context();s=ctx.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.RCVHWM,10000);s.connect(endpoint);end=time.time()+seconds;n=0
while time.time()<end:
 try:msg=s.recv_multipart(flags=zmq.NOBLOCK)
 except zmq.Again:time.sleep(.01);continue
 n+=1
 try:
  m=decode(msg[-1]);print(json.dumps(dataclasses.asdict(m)),flush=True)
 except Exception as e:print(json.dumps({'error':str(e),'key':msg[1].hex(),'hex':msg[-1].hex()}),flush=True)
print(json.dumps({'count':n,'endpoint':endpoint}))
