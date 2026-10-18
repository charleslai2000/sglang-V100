#!/usr/bin/env python3
"""Three-wave, native /generate service benchmark for isolated T011 vs exact c57."""
import concurrent.futures,json,os,statistics,threading,time
from pathlib import Path
import requests
from transformers import AutoTokenizer
BASE=os.environ.get('T011_BASE_URL','http://127.0.0.1:30003');MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4';OUT=Path(os.environ['T011_SERVICE_OUT']);ROUTE=os.environ['T011_ROUTE'];SHA=os.environ['T011_EXPECTED_SHA']
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True);q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False);fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def ids(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
def wave(prompt,c,rep):
 bar=threading.Barrier(c)
 def req(_):
  bar.wait();start=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':prompt,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True,'sampling_seed':31337},'stream':False},timeout=300);r.raise_for_status();body=r.json();wall=time.perf_counter()-start;ids=body['output_ids'];meta=body.get('meta_info',{});return {'wall_s':wall,'output_ids':ids,'meta_info':meta,'e2e_s':meta.get('e2e_latency')}
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:rs=list(pool.map(req,range(c)))
 return {'prompt_tokens':len(prompt),'client_concurrency':c,'repeat':rep,'same_ids_within_wave':all(r['output_ids']==rs[0]['output_ids'] for r in rs),'aggregate_output_tps':sum(len(r['output_ids']) for r in rs)/max(r['wall_s'] for r in rs),'wave_wall_s':max(r['wall_s'] for r in rs),'requests':rs}
rows=[]
for n in (1024,4096):
 prompt=ids(n)
 for c in (1,2,4):
  warm=wave(prompt,c,0)
  for r in range(1,4):rows.append(wave(prompt,c,r))
OUT.write_text(json.dumps({'route':ROUTE,'expected_sha':SHA,'model_revision':'52036ae497a2c0c253e17f98cc8652f2f7082ae3','sampling':'temperature=0 seed 31337 ignore_eos 256 output tokens','results':rows},indent=2)+'\n');print(json.dumps({'route':ROUTE,'output':str(OUT),'cells':len(rows)}))
