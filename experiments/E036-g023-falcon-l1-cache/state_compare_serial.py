#!/usr/bin/env python3
"""Serial exact prompt output reference/hit comparison. Warm checkpoint is a normal request."""
import hashlib,json,requests,time
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003';MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4';t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
shared=t.encode('A quiet lighthouse stands above a cold northern sea. ',add_special_tokens=False);heads=[t.encode(x,add_special_tokens=False) for x in ('Write about branch alpha: ','Write about branch bravo: ')];tails=[t.encode(x,add_special_tokens=False) for x in ('The first ship turns west at dawn. ','The second ship waits until midnight. ')];q=t.encode('Explain the scene in a short paragraph.',add_special_tokens=False)
def prompt(i): return ((shared*80)[:768]+heads[i]+tails[i]+q)[:1024]
P=[prompt(0),prompt(1)]
def req(label,ids):
 r=requests.post(BASE+'/generate',json={'input_ids':ids,'rid':label+str(time.time_ns()),'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True,'sampling_seed':908172}},timeout=240);r.raise_for_status();j=r.json();return {'label':label,'prompt_sha256':hashlib.sha256(bytes(str(ids),'ascii')).hexdigest(),'ids':j['output_ids'],'cached_tokens':j.get('meta_info',{}).get('cached_tokens'),'meta':j.get('meta_info')}
warm=req('ordinary-warm-',P[0]);records=[warm]
for i,p in enumerate(P):
 records.append(req(f'branch-{i}-A-',p));records.append(req(f'branch-{i}-B-',p))
print(json.dumps({'records':records},indent=2))
