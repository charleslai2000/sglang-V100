#!/usr/bin/env python3
"""State isolation test. S1 vs candidate run with same prompts, seeds and server flags."""
import concurrent.futures,hashlib,json,requests,time,threading
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003';MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=t.encode('Continue the story naturally. End with one sentence.',add_special_tokens=False)
fill=t.encode('A traveler walks beside a quiet river under the morning sky. ',add_special_tokens=False)
def make(n,tag):
 ids=[]
 while len(ids)+len(q)<n:ids+=fill
 tagids=t.encode(tag,add_special_tokens=False)
 ids[:len(tagids)]=tagids
 return (ids[:n-len(q)]+q)[:n]
P={'A':make(1024,'REQUEST ALPHA ID 101'),'B':make(1024,'REQUEST BRAVO ID 202'),'C':make(1024,'REQUEST CHARLIE ID 303')}
def run(label, key, rid):
 start=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':P[key],'rid':rid+str(time.time_ns()),'sampling_params':{'temperature':0,'max_new_tokens':128,'ignore_eos':True,'sampling_seed':343434}},timeout=240);r.raise_for_status();j=r.json();return {'label':label,'key':key,'rid':rid,'output_ids':j['output_ids'],'n':len(j['output_ids']),'wall_s':time.perf_counter()-start}
# Each request's own solo reference, then overlapping A+B; continuation/new request after both release.
solo={k:run(k+'-solo',k,'g017-'+k+'-solo') for k in P}
bar=threading.Barrier(2)
def conc(k):bar.wait();return run(k+'-concurrent',k,'g017-'+k+'-concurrent')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: concurrent_rows=list(ex.map(conc,['A','B']))
follow=[run('C-after-A-B','C','g017-C-follow'),run('A-continuation','A','g017-A-cont')]
reused=[run('B-after-release','B','g017-B-reuse'),run('A-after-release','A','g017-A-reuse')]
rows=list(solo.values())+concurrent_rows+follow+reused
summary={}
for k in P:
 ref=solo[k]['output_ids'];items=[x for x in rows if x['key']==k]
 summary[k]={'prompt_sha256':hashlib.sha256(bytes(str(P[k]),'ascii')).hexdigest(),'prompt_prefix':P[k][:12],'all_same_as_solo':all(x['output_ids']==ref for x in items),'all_128':all(x['n']==128 for x in items),'labels':[x['label'] for x in items]}
print(json.dumps({'summary':summary,'records':rows},indent=2))
