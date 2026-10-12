#!/usr/bin/env python3
"""Join temporary RID-correlated G021 lifecycle client rows to SGLang ReqTimeStats logs."""
import json,re,sys
from pathlib import Path
client=Path(sys.argv[1]); logs=Path(sys.argv[2]);
pat=re.compile(r'ReqTimeStats\(rid=(?P<rid>[^,]+), input_len=(?P<input>\d+), cached_input_len=(?P<cached>\d+), output_len=(?P<output>\d+), type=(?P<type>[^)]+)\): queue_duration=(?P<queue>[0-9.]+)ms, forward_duration=(?P<forward>[0-9.]+)ms, entry_time=(?P<entry>[0-9.]+)')
stats={}
for line in logs.read_text(errors='replace').splitlines():
 m=pat.search(line)
 if m: stats[m['rid']]={k:(float(v) if k in ('queue','forward','entry') else int(v) if k in ('input','cached','output') else v) for k,v in m.groupdict().items() if k!='rid'}
for line in client.read_text().splitlines():
 d=json.loads(line)
 print(f"\nctx={d['prompt_tokens']} c={d['concurrency']} client_wave_tps={d['wave_tps']:.2f}")
 for r in d['requests']:
  s=stats.get(r['rid'])
  print(json.dumps({'rid':r['rid'],'ttft_ms':round(r['ttft_s']*1000,2),'client_wall_ms':round(r['wall_s']*1000,2),'event_count':r['event_count'],'server':s},sort_keys=True))
print(f'\nmatched={sum(1 for line in client.read_text().splitlines() for r in json.loads(line)["requests"] if r["rid"] in stats)} / {sum(len(json.loads(line)["requests"]) for line in client.read_text().splitlines())}')
