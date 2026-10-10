#!/usr/bin/env python3
"""Target-UUID only nvidia-smi telemetry sampler. No system/driver state changes."""
import csv,subprocess,sys,time
UUID=sys.argv[1];OUT=sys.argv[2];DUR=float(sys.argv[3]);PERIOD=float(sys.argv[4]) if len(sys.argv)>4 else 1.0
QUERY_FIELDS=['timestamp','uuid','utilization.gpu','utilization.memory','clocks.current.sm','clocks.current.memory','pstate','power.draw','temperature.gpu','memory.total','memory.used','clocks_throttle_reasons.supported','clocks_throttle_reasons.active']
FIELDS=QUERY_FIELDS
cmd=['nvidia-smi',f'--id={UUID}',f'--query-gpu={",".join(QUERY_FIELDS)}','--format=csv,noheader,nounits']
with open(OUT,'w',newline='') as f:
 w=csv.writer(f);w.writerow(FIELDS);end=time.monotonic()+DUR
 while time.monotonic()<end:
  ts=time.time();r=subprocess.run(cmd,capture_output=True,text=True,check=True);row=next(csv.reader([r.stdout.strip()]));w.writerow([ts]+[x.strip() for x in row[1:]]);f.flush();time.sleep(max(0,PERIOD-(time.time()-ts)))
print(OUT)
