#!/usr/bin/env python3
import concurrent.futures,json,threading,time,requests,statistics
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003';MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False);fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def ids(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
def wave(prompt,c,rep):
 bar=threading.Barrier(c)
 def req(_):
  bar.wait();st=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':prompt,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True},'stream':True},stream=True,timeout=240);r.raise_for_status();outs=[];ts=[]
  for line in r.iter_lines(decode_unicode=True):
   if line and line.startswith('data: '):
    body=line[6:]
    if body=='[DONE]':break
    o=json.loads(body);cur=o.get('output_ids',[]);delta=cur[len(outs):]
    if delta:outs.extend(delta);ts.extend([time.perf_counter()]*len(delta))
  assert len(outs)==256,len(outs); d=sorted((ts[i]-ts[i-1])*1000 for i in range(1,len(ts)))
  return {'wall_s':time.perf_counter()-st,'output_ids':outs,'p50_itl_ms':statistics.median(d),'p95_itl_ms':d[int(.95*(len(d)-1))]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:rows=list(ex.map(req,range(c)))
 return {'prompt_tokens':len(prompt),'concurrency':c,'repeat':rep,'same_ids':all(r['output_ids']==rows[0]['output_ids'] for r in rows),'aggregate_tps':c*256/max(r['wall_s'] for r in rows),'requests':rows}
rows=[]
for n in (1024,4096):
 p=ids(n)
 for c in (1,2,4):
  wave(p,c,0)
  for r in range(1,4):rows.append(wave(p,c,r))
print(json.dumps({'candidate_service_smoke':True,'results':rows},indent=2))
