#!/usr/bin/env python3
"""Summarize the existing bench_one_batch Kineto traces for one decode step."""
import gzip,json,collections,hashlib,sys

def bucket(name):
 n=name.lower()
 if 'sm70_marlin_u4b8' in n: return 'Marlin'
 if '_fwd_grouped_kernel_stage' in n or '_fwd_kernel_stage' in n or 'attention' in n and 'kernel' in n: return 'Attention'
 if 'memcpy' in n or 'memset' in n: return 'Copy/memset'
 return 'Other GPU'
for path in sys.argv[1:]:
 d=json.load(gzip.open(path,'rt')); events=[e for e in d['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X' and e.get('args',{}).get('device')==0]
 if not events: raise SystemExit(f'no CUDA kernel events in {path}')
 start=min(e['ts'] for e in events); end=max(e['ts']+e['dur'] for e in events)
 cats=collections.defaultdict(lambda:[0,0.0]);
 for e in events:
  x=cats[bucket(e['name'])];x[0]+=1;x[1]+=e['dur']
 out={'path':path,'sha256':hashlib.sha256(open(path,'rb').read()).hexdigest(),'kernel_events':len(events),'gpu_activity_union_ms':(end-start)/1000,'kernel_duration_sum_ms':sum(e['dur'] for e in events)/1000,'categories':{k:{'calls':v[0],'duration_sum_ms':v[1]/1000,'share_of_sum_pct':100*v[1]/sum(x[1] for x in cats.values())} for k,v in cats.items()},'graph_node_ids_present':any('graphNode' in str(e.get('args',{})) for e in events)}
 print(json.dumps(out,sort_keys=True))
