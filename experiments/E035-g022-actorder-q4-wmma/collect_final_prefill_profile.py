#!/usr/bin/env python3
"""Use the established /start_profile profile_range API for short EXTEND traces."""
import json,os,time,requests
from pathlib import Path
BASE='http://127.0.0.1:30003'; OUT=Path(os.environ['E035_TRACE_DIR']);OUT.mkdir(parents=True,exist_ok=True)
from transformers import AutoTokenizer
T=AutoTokenizer.from_pretrained('/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4',local_files_only=True,trust_remote_code=True)
q=T.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False);fill=T.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def prompt(n):
 x=[]
 while len(x)+len(q)<n:x+=fill
 return (x[:n-len(q)]+q)[:n]
rows=[]
for n in (1024,4096):
 p=prompt(n)
 # request an isolated prefill profiler window, then a warm decode profile on another request
 for label,body,request in [
  ('prefill',{'output_dir':str(OUT),'num_steps':3,'activities':['CPU','GPU','CUDA_PROFILER'],'with_stack':False,'record_shapes':False,'profile_by_stage':True,'profile_stages':['extend'],'profile_range':'prefill','profile_prefix':f'fp64-prefill-{n}'},{'input_ids':p,'sampling_params':{'temperature':0,'max_new_tokens':1,'ignore_eos':True}}),
  ('decode',{'output_dir':str(OUT),'num_steps':4,'activities':['CPU','GPU','CUDA_PROFILER'],'with_stack':False,'record_shapes':False,'profile_by_stage':True,'profile_stages':['decode'],'profile_range':'decode','profile_prefix':f'fp64-decode-{n}'},{'input_ids':p,'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True}})]:
  rr=requests.post(BASE+'/generate',json={'input_ids':p,'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True}},timeout=240);rr.raise_for_status()
  t=requests.post(BASE+'/start_profile',json=body,timeout=30);t.raise_for_status();start=time.time()
  rr=requests.post(BASE+'/generate',json=request,timeout=300);rr.raise_for_status();elapsed=rr.json().get('meta_info',{}).get('e2e_latency')
  traces=[]
  for _ in range(240):
   traces=list(OUT.glob(f"{body['profile_prefix']}*.trace.json.gz"))
   if traces:break
   time.sleep(.5)
  if not traces:requests.post(BASE+'/stop_profile',timeout=30)
  rows.append({'context':n,'range':label,'client_wall_s':time.time()-start,'e2e_latency_s':elapsed,'traces':[str(x) for x in traces]})
print(json.dumps(rows,indent=2))
