#!/usr/bin/env python3
"""Three paired c57/FP64 service repeats; requires a serially switched service."""
import concurrent.futures, hashlib, json, os, statistics, threading, time
from pathlib import Path
import requests
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003'; MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
OUT=Path(os.environ['E035_SERVICE_OUT']); ROUTE=os.environ['E035_ROUTE']; SHA=os.environ['E035_EXPECTED_SHA']
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False); fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def ids(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
def wave(prompt,c,rep):
 bar=threading.Barrier(c)
 def req(_):
  bar.wait();st=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':prompt,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True,'sampling_seed':31337},'stream':True,'return_logprob':True},stream=True,timeout=240);r.raise_for_status();events=[]
  for line in r.iter_lines(decode_unicode=True):
   if line and line.startswith('data: '):
    body=line[6:]
    if body=='[DONE]':break
    events.append((time.perf_counter(),json.loads(body)))
  out=[];ts=[];meta={}
  for when,o in events:
   cur=o.get('output_ids',[]);delta=cur[len(out):]
   if delta:out.extend(delta);ts.extend([when]*len(delta))
   meta=o.get('meta_info',meta)
  assert len(out)==256,(len(out),meta)
  itl=[(ts[i]-ts[i-1])*1000 for i in range(1,len(ts))];itl.sort()
  return {'wall_s':time.perf_counter()-st,'output_ids':out,'meta_info':meta,'ttft_s':ts[0]-st,'itl_ms_p50':statistics.median(itl),'itl_ms_p95':itl[int(.95*(len(itl)-1))]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:rows=list(ex.map(req,range(c)))
 return {'prompt_tokens':len(prompt),'client_concurrency':c,'repeat':rep,'same_ids_within_wave':all(r['output_ids']==rows[0]['output_ids'] for r in rows),'aggregate_tps':c*256/max(x['wall_s'] for x in rows),'wave_wall_s':max(x['wall_s'] for x in rows),'requests':rows}
rows=[]
for n in (1024,4096):
 p=ids(n)
 for c in (1,2,4):
  wave(p,c,0)
  for r in range(1,4):rows.append(wave(p,c,r))
with open(OUT,'w') as f:json.dump({'route':ROUTE,'expected_sha':SHA,'model_revision':'52036ae497a2c0c253e17f98cc8652f2f7082ae3','sampling':'greedy temperature=0, seed=31337, ignore_eos, 256 tokens','results':rows},f,indent=2)
print(json.dumps({'out':str(OUT),'route':ROUTE,'results':len(rows)}))
