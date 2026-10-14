#!/usr/bin/env python3
import concurrent.futures,json,threading,time,requests
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003'; MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True);q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False);fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def make_ids(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
baseline=json.load(open('/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-service-provisional.json'));rows=[]
for n in (1024,4096):
 prompt=make_ids(n)
 for c in (1,2,4):
  old=next(x for x in baseline['results'] if x['prompt_tokens']==n and x['concurrency']==c and x['repeat']==1)
  for rep in range(1,4):
   bar=threading.Barrier(c)
   def req(i):
    bar.wait();st=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':prompt,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True},'stream':True},stream=True,timeout=300);r.raise_for_status();out=[]
    for line in r.iter_lines(decode_unicode=True):
     if line and line.startswith('data: '):
      b=line[6:]
      if b=='[DONE]':break
      o=json.loads(b);cur=o.get('output_ids',[]);out.extend(cur[len(out):])
    return {'wall_s':time.perf_counter()-st,'ids':out}
   with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:cur=list(ex.map(req,range(c)))
   ref=old['requests'];cmp=[{'request':i,'n':len(a['ids']),'same_as_s1':a['ids']==ref[i]['output_ids'],'first_diff':next((j for j,(u,v) in enumerate(zip(a['ids'],ref[i]['output_ids'])) if u!=v),None),'wall_s':a['wall_s']} for i,a in enumerate(cur)]
   row={'prompt_tokens':n,'c':c,'repeat':rep,'candidate_within_wave_same':all(a['ids']==cur[0]['ids'] for a in cur),'vs_frozen_s1':cmp};rows.append(row);print(json.dumps(row),flush=True)
print(json.dumps({'results':rows},indent=2))
