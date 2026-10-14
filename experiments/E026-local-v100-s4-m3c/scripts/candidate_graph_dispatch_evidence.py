#!/usr/bin/env python3
"""Run minimal graph-size/service route smoke and collect server dispatch logs."""
import concurrent.futures,json,threading,time,requests,statistics
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003';MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=t.encode('Question: What is 6 times 7? Explain briefly and answer clearly.',add_special_tokens=False);fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
def prompt(n):
 p=[]
 while len(p)+len(q)<n:p+=fill
 return (p[:n-len(q)]+q)[:n]
def run_wave(n,c,rep):
 ids=prompt(n);bar=threading.Barrier(c)
 def req(_):
  bar.wait();st=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':32,'ignore_eos':True},'stream':True},stream=True,timeout=240);r.raise_for_status();out=[];events=[]
  for line in r.iter_lines(decode_unicode=True):
   if line and line.startswith('data: '):
    body=line[6:]
    if body=='[DONE]':break
    obj=json.loads(body);cur=obj.get('output_ids',[]);delta=cur[len(out):]
    if delta:out.extend(delta);events.append((time.perf_counter(),len(delta)))
  assert len(out)==32,(len(out),out)
  itls=[]
  for dt,k in events:itls.extend([1000*0.0]*0)
  return {'ids':out,'elapsed_s':time.perf_counter()-st,'tokens':len(out),'events':len(events)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:rows=list(ex.map(req,range(c)))
 return {'prompt_tokens':n,'c':c,'repeat':rep,'ids_equal':all(x['ids']==rows[0]['ids'] for x in rows),'aggregate_tps':c*32/max(r['elapsed_s'] for r in rows),'request_rows':rows}
results=[]
for n,c in ((1024,1),(1024,2),(1024,4),(4096,1),(4096,2),(4096,4)):
 results.append(run_wave(n,c,1))
print(json.dumps({'dispatch_test':'candidate opt-in enabled by SGLANG_G017_Q4_FP32_CANDIDATE=1; see server log for CUDA graph=True and candidate wrapper NVTX/log evidence','results':results},indent=2))
