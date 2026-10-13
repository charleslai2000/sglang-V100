#!/usr/bin/env python3
"""Decode captured frozen S1 forward outputs to locate GPTQ propagation effects.

Uses local V100, deterministic greedy decoding, no source patching/instrumentation.
"""
import json,time,requests
from transformers import AutoTokenizer
MODEL='/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'; BASE='http://127.0.0.1:30003'
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
prompt='Question: What is 6 times 7? Explain briefly and answer clearly.'
q=t.encode(prompt,add_special_tokens=False); fill=t.encode('The quick brown fox walks through a garden. ',add_special_tokens=False)
ids=[]
while len(ids)+len(q)<1024: ids+=fill
ids=(ids[:1024-len(q)]+q)[:1024]
outs=[]
for c in (1,2,4):
 for rep in range(1,4):
  batch=[ids]*c; st=time.perf_counter()
  r=requests.post(BASE+'/generate',json={'input_ids':batch if c>1 else ids,'sampling_params':{'temperature':0.0,'max_new_tokens':256,'ignore_eos':True}},timeout=180);r.raise_for_status();d=r.json();
  rows=d if isinstance(d,list) else d.get('output_ids',[])
  # SGLang batch endpoint returns list of individual generation objects when input_ids is list.
  if rows and isinstance(rows[0],int): rows=[d]
  toks=[x.get('output_ids',[]) for x in rows]
  outs.append({'c':c,'repeat':rep,'wall_s':time.perf_counter()-st,'tokens':toks})
print(json.dumps({'prompt_tokens':1024,'outputs':outs},indent=2))
