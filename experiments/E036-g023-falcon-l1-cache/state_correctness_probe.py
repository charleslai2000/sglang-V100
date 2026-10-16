#!/usr/bin/env python3
"""Real-request cache-enabled Falcon branching/continuation comparison against no-cache decode."""
import concurrent.futures,hashlib,json,threading,time,requests
from transformers import AutoTokenizer
BASE='http://127.0.0.1:30003'; MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
shared=t.encode('A quiet lighthouse stands above a cold northern sea. ',add_special_tokens=False)
heads=[t.encode(x,add_special_tokens=False) for x in ('Write about branch alpha: ','Write about branch bravo: ')]
tails=[t.encode(x,add_special_tokens=False) for x in ('The first ship turns west at dawn. ','The second ship waits until midnight. ')]
question=t.encode('Explain the scene in a short paragraph.',add_special_tokens=False)
def make(branch,turn=0):
 prefix=(shared*80)[:768]
 suffix=heads[branch]+tails[branch]+([t.encode('Continue with new details. ',add_special_tokens=False)[0]]*turn)+question
 ids=(prefix+suffix)[:1024]
 return ids
P=[make(i) for i in range(2)]
def req(name,ids):
 st=time.perf_counter();r=requests.post(BASE+'/generate',json={'input_ids':ids,'rid':name+'-'+str(time.time_ns()),'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True,'sampling_seed':908172},'return_logprob':True},timeout=240);r.raise_for_status();j=r.json();return {'name':name,'prompt_sha':hashlib.sha256(bytes(str(ids),'ascii')).hexdigest(),'out':j['output_ids'],'meta':j.get('meta_info',{}),'wall_s':time.perf_counter()-st}
# Warm checkpoint by normal completed request from a shared-prefix input; branch suffixes are distinct.
warm=req('warm-branch-alpha',P[0])
solo=[req(f'uncached-reference-{i}',p) for i,p in enumerate(P)]
bar=threading.Barrier(2)
def request_branch(i): bar.wait();return req(f'concurrent-branch-{i}',P[i])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: branches=list(ex.map(request_branch,range(2)))
# New multi-turn requests exercise matching a continuation containing generated output.
cont=req('continuation-after-shared',P[0]+warm['out'][:8])
# Reissue branches after completion to exercise request-slot reuse and a repeat hit.
reuse=[req(f'branch-reuse-{i}',P[i]) for i in range(2)]
rows=[warm,*solo,*branches,cont,*reuse]
print(json.dumps({'cache_mode':'extra_buffer radix enabled','outputs_256':all(len(x['out'])==256 for x in rows if x['name']!='continuation-after-shared'),'baseline_id_sequence_equal_for_same_prompt':{str(i):branches[i]['out']==solo[i]['out'] and reuse[i]['out']==solo[i]['out'] for i in range(2)},'same_branch_sequences_equal':branches[0]['out']==branches[1]['out'],'warm_token_counts':{x['name']:x['meta'].get('cached_tokens') for x in rows},'records':rows},indent=2))
