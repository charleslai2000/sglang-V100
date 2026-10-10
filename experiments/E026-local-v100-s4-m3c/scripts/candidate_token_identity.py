#!/usr/bin/env python3
"""Final-candidate greedy parity matrix against frozen-S1 reference IDs."""
import argparse, concurrent.futures, json, threading, time, requests
from pathlib import Path
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);a=p.parse_args()
base='http://127.0.0.1:30003'; model='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'; tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=True)
q=tok.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False); fill=tok.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def prompt(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
def request(ids):
 st=time.perf_counter(); r=requests.post(base+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True},'stream':True},stream=True,timeout=300); r.raise_for_status(); out=[]
 for line in r.iter_lines(decode_unicode=True):
  if line and line.startswith('data: '):
   data=line[6:]
   if data=='[DONE]':break
   cur=json.loads(data).get('output_ids',[]);out.extend(cur[len(out):])
 return {'wall_s':time.perf_counter()-st,'ids':out}
baseline=json.load(open(a.baseline)); rows=[]
for n in (1024,4096):
 ids=prompt(n)
 for c in (1,2,4):
  ref=next(r for r in baseline['results'] if r['prompt_tokens']==n and r['concurrency']==c and r['repeat']==1)['requests']
  for rep in (1,2,3):
   barrier=threading.Barrier(c)
   def run(_): barrier.wait(); return request(ids)
   with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool: cur=list(pool.map(run,range(c)))
   assert all(len(x['ids'])==256 for x in cur)
   checks=[{'request':i,'same_as_frozen_s1':cur[i]['ids']==ref[i]['output_ids']} for i in range(c)]
   row={'prompt_tokens':n,'concurrency':c,'repeat':rep,'within_wave_same':all(x['ids']==cur[0]['ids'] for x in cur),'requests':checks}
   rows.append(row); print(json.dumps(row),flush=True)
with open(a.out,'w') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
if len(rows)!=18 or not all(r['within_wave_same'] and all(x['same_as_frozen_s1'] for x in r['requests']) for r in rows): raise SystemExit('final candidate parity failed')
