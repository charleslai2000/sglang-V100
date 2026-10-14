#!/usr/bin/env python3
"""Local frozen S1 c1/c2/c4 baseline. Never run under a profiler."""
import concurrent.futures,json,threading,time,requests,statistics
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003'; MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'; t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False); fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False); p=[]
while len(p)+len(q)<1024:p+=fill
ids=(p[:1024-len(q)]+q)[:1024]
def wave(c,rep):
 bar=threading.Barrier(c)
 def one(_):
  bar.wait(); start=time.perf_counter(); r=requests.post(BASE+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0.0,'max_new_tokens':256,'ignore_eos':True}},timeout=180); elapsed=time.perf_counter()-start; r.raise_for_status();d=r.json(); m=d.get('meta_info',{}); out=d.get('output_ids',[]); assert out and len(out)<=256 and all(isinstance(x,int) and x>=0 for x in out); return {'wall_s':elapsed,'output_ids':out,'text':d.get('text',''),'meta_info':m}
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex: rows=list(ex.map(one,range(c)))
 return rows
out=[]
for c in (1,2,4):
 for rep in range(1,4):
  ts=time.perf_counter(); rows=wave(c,rep); wall=time.perf_counter()-ts
  if c>1: assert len({r['text'] for r in rows})==1, 'same greedy prompt diverged across batch'
  out.append({'c':c,'repeat':rep,'wave_wall_s':wall,'requests':rows})
print(json.dumps({'prompt_tokens':1024,'max_new_tokens':256,'temperature':0,'results':out},indent=2,ensure_ascii=False))
