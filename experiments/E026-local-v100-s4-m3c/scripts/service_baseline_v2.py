#!/usr/bin/env python3
"""Same-workload client TPS, streamed TTFT and ITL harness for G017 A/B."""
import concurrent.futures,json,threading,time,requests,statistics,hashlib,os
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003'; MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'; REV='52036ae497a2c0c253e17f98cc8652f2f7082ae3'
BIN=os.environ['SGLANG_G017_MEASURED_EXTENSION']; tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=tok.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False); fill=tok.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def ids(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
def wave(prompt,c,rep):
 barrier=threading.Barrier(c)
 def req(_):
  barrier.wait(); st=time.perf_counter(); r=requests.post(BASE+'/generate',json={'input_ids':prompt,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True},'stream':True},stream=True,timeout=300); r.raise_for_status(); events=[]
  for line in r.iter_lines(decode_unicode=True):
   if line and line.startswith('data: '):
    data=line[6:]
    if data=='[DONE]':break
    events.append((time.perf_counter(),json.loads(data)))
  out=[]; ts=[]
  for t,obj in events:
   cur=obj.get('output_ids',[]); delta=cur[len(out):]
   if delta:out.extend(delta);ts.extend([t]*len(delta))
  assert len(out)==256
  itl=sorted((ts[i]-ts[i-1])*1000 for i in range(1,len(ts)))
  return {'wall_s':time.perf_counter()-st,'output_ids':out,'ttft_s':ts[0]-st,'itl_p50_ms':statistics.median(itl),'itl_p95_ms':itl[int(.95*(len(itl)-1))]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex: rows=list(ex.map(req,range(c)))
 return {'prompt_tokens':len(prompt),'concurrency':c,'repeat':rep,'same_ids_within_wave':all(r['output_ids']==rows[0]['output_ids'] for r in rows),'aggregate_tps':c*256/max(r['wall_s'] for r in rows),'wave_wall_s':max(r['wall_s'] for r in rows),'requests':rows}
runs=[]
for n in (1024,4096):
 p=ids(n)
 for c in (1,2,4):
  wave(p,c,0)
  for rep in range(1,int(os.environ.get('G017_SERVICE_REPEATS','3'))+1):runs.append(wave(p,c,rep))
print(json.dumps({'model_revision':REV,'binary':BIN,'binary_sha256':hashlib.sha256(open(BIN,'rb').read()).hexdigest(),'output_tokens':256,'sampling':'greedy temperature=0 ignore_eos=true','results':runs},indent=2))
