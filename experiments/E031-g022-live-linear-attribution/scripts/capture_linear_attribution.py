#!/usr/bin/env python3
import concurrent.futures, json, os, pathlib, threading, time, requests
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003'; MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
OUT=pathlib.Path(os.environ['G022_TRACE_DIR']); OUT.mkdir(parents=True,exist_ok=True)
C=int(os.environ['G022_CONCURRENCY']); N=int(os.environ['G022_PROMPT_TOKENS'])
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False); fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
p=[]
while len(p)+len(q)<N:p.extend(fill)
ids=(p[:N-len(q)]+q)[:N]
# warm a full greedy response before profiling to isolate Decode
r=requests.post(BASE+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True}},timeout=240); r.raise_for_status()
steps=C+2
body={'output_dir':str(OUT),'num_steps':steps,'activities':['CPU','GPU','CUDA_PROFILER'],'with_stack':False,'record_shapes':False,'profile_by_stage':True,'profile_stages':['decode'],'profile_range':'decode','profile_prefix':f'g022-e031-c{C}-{N}'}
r=requests.post(BASE+'/start_profile',json=body,timeout=30); r.raise_for_status()
bar=threading.Barrier(C)
def one(_):
 bar.wait(); start=time.perf_counter(); rr=requests.post(BASE+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':4,'ignore_eos':True},'stream':True},stream=True,timeout=240); rr.raise_for_status(); out=[]
 for line in rr.iter_lines(decode_unicode=True):
  if line and line.startswith('data: '):
   x=line[6:]
   if x=='[DONE]':break
   out=json.loads(x).get('output_ids',out)
 return {'elapsed_s':time.perf_counter()-start,'tokens':len(out)}
with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool: result=list(pool.map(one,range(C)))
for _ in range(240):
 traces=list(OUT.glob('g022-e031-*.trace.json.gz'))
 if traces: break
 time.sleep(.25)
else: requests.post(BASE+'/stop_profile',timeout=30); traces=list(OUT.glob('g022-e031-*.trace.json.gz'))
print(json.dumps({'concurrency':C,'prompt_tokens':N,'requests':result,'traces':[str(p) for p in traces]},indent=2))
