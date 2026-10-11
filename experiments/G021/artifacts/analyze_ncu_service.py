#!/usr/bin/env python3
"""Summarize existing G021 service, NVML and bounded NCU CSV artifacts offline."""
import csv,json,statistics,sys,glob,os
BASE=os.path.dirname(__file__)

def quant(v,p):
 v=sorted(v)
 if not v:return None
 return v[min(len(v)-1,int(p*(len(v)-1)))]
for mode,file in [('S1','service-s1-all-cells.json'),('Candidate-dispatch-verified','service-candidate-dispatch-verified.json')]:
 d=json.load(open(os.path.join(BASE,file)))
 print('\n',mode,'sha256',d['binary_sha256'])
 for n in (1024,4096):
  for c in (1,4):
   waves=[x for x in d['results'] if x['prompt_tokens']==n and x['concurrency']==c]
   req=[q for w in waves for q in w['requests']]
   ttft=[q['first_token_s']*1000 for q in req]; e2e=[q['wall_s']*1000 for q in req]
   itl=[q['itl_ms_p50'] for q in req]; itl95=[q['itl_ms_p95'] for q in req]
   print(n,c,'tok/s',round(statistics.mean(w['aggregate_tps'] for w in waves),2),'TTFTms p50/p95',round(quant(ttft,.5),1),round(quant(ttft,.95),1),'ITLms-request-p50/p95',round(quant(itl,.5),2),round(quant(itl95,.95),2),'E2Ems p50/p95',round(quant(e2e,.5),1),round(quant(e2e,.95),1),'samewave',all(w['same_ids_within_wave'] for w in waves))
keys=('gpu__time_duration.sum','sm__throughput.avg','sm__cycles_active.avg','sm__issue_active.avg','sm__inst_executed_pipe_fp16.avg','sm__inst_executed_pipe_fma.avg','sm__inst_executed_pipe_tensor.avg','sm__pipe_tensor_cycles_active.avg','dram__bytes_read.avg','dram__bytes_write.avg','dram__throughput.avg','dram__cycles_active.avg','lts__t_bytes.avg','lts__throughput.avg','smsp__average_warps_active_per_issue_active.ratio','smsp__warps_issue_stalled_long_scoreboard.avg')
for f in sorted(glob.glob(os.path.join(BASE,'ncu-*.csv'))):
 rows=list(csv.reader(line for line in open(f) if line.startswith('"0"'))); d={r[12]:r[14] for r in rows if len(r)>14 and r[12] in keys}
 print('\n',os.path.basename(f),d)
for f in sorted(glob.glob(os.path.join(BASE,'nvml-*-matrix.csv'))):
 r=list(csv.DictReader(open(f))); print('\n',os.path.basename(f),'rows',len(r),'window',r[0]['timestamp'],r[-1]['timestamp'])
 for k in ('utilization.gpu','utilization.memory','memory.used','clocks.current.sm','clocks.current.memory','power.draw','temperature.gpu'):
  a=[float(x[k]) for x in r];print(k,'mean',round(statistics.mean(a),2),'p95',quant(a,.95),'max',max(a))
 print('active throttle masks',sorted(set(x['clocks_throttle_reasons.active'] for x in r)))
