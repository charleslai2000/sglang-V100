#!/usr/bin/env python3
"""Frozen source/build baseline: c1/c2/c4 1K/4K, paired repeats, streamed ITL."""
import concurrent.futures,json,threading,time,requests,statistics,hashlib,os
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003';MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4';MODEL_REV='52036ae497a2c0c253e17f98cc8652f2f7082ae3';BIN=os.environ.get('SGLANG_G017_MEASURED_EXTENSION','/home/charles/miniconda3/envs/sglang-v100/lib/python3.12/site-packages/sgl_kernel/sm70/common_ops.abi3.so')
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False);fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def ids(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
def wave(prompt,c,rep):
 bar=threading.Barrier(c)
 def req(_):
  bar.wait();st=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':prompt,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True},'stream':True,'return_logprob':True},stream=True,timeout=240);r.raise_for_status();ev=[]
  for line in r.iter_lines(decode_unicode=True):
   if line and line.startswith('data: '):
    body=line[6:]
    if body=='[DONE]':break
    ev.append((time.perf_counter(),json.loads(body)))
  out=[];ts=[];meta={}
  for when,obj in ev:
   cur=obj.get('output_ids',[]);delta=cur[len(out):]
   if delta:out.extend(delta);ts.extend([when]*len(delta))
   meta=obj.get('meta_info',meta)
  assert len(out)==256,(len(out),meta)
  itl=[(ts[i]-ts[i-1])*1000 for i in range(1,len(ts))]
  itl.sort()
  return {'wall_s':time.perf_counter()-st,'output_ids':out,'meta_info':meta,'itl_ms_p50':statistics.median(itl),'itl_ms_p95':itl[int(.95*(len(itl)-1))],'first_token_s':ts[0]-st,'last_token_s':ts[-1]-st}
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:rows=list(ex.map(req,range(c)))
 equal=all(r['output_ids']==rows[0]['output_ids'] for r in rows)
 wall=max(r['wall_s'] for r in rows)
 return {'prompt_tokens':len(prompt),'concurrency':c,'repeat':rep,'same_ids_within_wave':equal,'aggregate_tps':c*256/wall,'wave_wall_s':wall,'requests':rows}
runs=[]
for n in (1024,4096):
 prompt=ids(n)
 for c in (1,2,4):
  wave(prompt,c,0)  # explicit warmup
  for rep in range(1,int(os.environ.get('G017_SERVICE_REPEATS','3'))+1):runs.append(wave(prompt,c,rep))
print(json.dumps({'model_revision':MODEL_REV,'binary':BIN,'binary_sha256':hashlib.sha256(open(BIN,'rb').read()).hexdigest(),'output_tokens':256,'sampling':'greedy temperature=0 ignore_eos=true','results':runs},indent=2))
