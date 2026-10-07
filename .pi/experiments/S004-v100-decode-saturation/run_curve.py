#!/usr/bin/env python3
import concurrent.futures as cf, json, statistics, threading, time
import requests
from pathlib import Path
from transformers import AutoTokenizer
BASE=__import__('os').environ.get('S4_BASE','http://127.0.0.1:30003'); MODEL='/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
OUT=Path(__import__('os').environ['S4_OUT'])
if not str(OUT).startswith('/hy-tmp/sglang-logs/s4-v100-') or OUT.exists(): raise RuntimeError('unsafe/existing S4_OUT')
OUT.parent.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
q=tok.encode('\nQuestion: What is 6 times 7? Answer with the number only:',add_special_tokens=False); f=tok.encode('Context filler sentence. ',add_special_tokens=False)
def prompt(n):
 a=[]
 while len(a)+len(q)<n:a.extend(f)
 return (a[:n-len(q)]+q)[:n]
def snapshot():
 d=json.loads(requests.get(BASE+'/server_info',timeout=5).text); s=d['internal_states'][0]; return (s.get('decode_batch_hist',{}),s.get('memory_usage',{}))
def one(ids,n,bar):
 t0=time.perf_counter(); ts=[]; final=[]; err=None; bar.wait()
 try:
  with requests.post(BASE+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0.0,'max_new_tokens':n,'ignore_eos':True},'stream':True},stream=True,timeout=1800) as r:
   if r.status_code!=200:err=f'HTTP {r.status_code}: {r.text[:500]}'
   else:
    for line in r.iter_lines(decode_unicode=True):
     if not line or not line.startswith('data: '):continue
     z=line[6:]
     if z=='[DONE]':break
     x=json.loads(z).get('output_ids',[])
     if len(x)>len(final):ts += [time.perf_counter()]*(len(x)-len(final));final=x
 except Exception as e:err=repr(e)
 itl=[ts[i]-ts[i-1] for i in range(1,len(ts))]
 return {'n':len(final),'nonzero':sum(bool(x) for x in final),'ttft':ts[0]-t0 if ts else None,'itl_mean':statistics.mean(itl) if itl else None,'itl_p95':sorted(itl)[int(.95*(len(itl)-1))] if itl else None,'error':err}
def wave(c,rep,n=256):
 ids=prompt(1024); bar=threading.Barrier(c); pre,_=snapshot(); t=time.perf_counter()
 with cf.ThreadPoolExecutor(max_workers=c) as ex:rs=list(ex.map(lambda _:one(ids,n,bar),range(c)))
 wall=time.perf_counter()-t; post,mem=snapshot(); keys=set(pre)|set(post); hist={k:int(post.get(k,0))-int(pre.get(k,0)) for k in keys}; ok=all(r['n']==n and r['nonzero'] and not r['error'] for r in rs)
 d={'c':c,'rep':rep,'wall_s':wall,'aggregate_tps':sum(r['n'] for r in rs)/wall,'per_req_tps':sum(r['n'] for r in rs)/(wall*c),'mean_itl_ms':1000*statistics.mean(r['itl_mean'] for r in rs if r['itl_mean'] is not None),'p95_itl_ms':1000*statistics.mean(r['itl_p95'] for r in rs if r['itl_p95'] is not None),'correct':ok,'batch_hist':hist,'memory_usage':mem,'request':[{k:v for k,v in r.items() if k!='n'} for r in rs]}
 with OUT.open('a') as f:f.write(json.dumps(d)+'\\n')
 print(json.dumps(d),flush=True)
 if not ok:raise RuntimeError(f'correctness failed c={c} rep={rep}')
 return d

if __name__ == '__main__':
    c = int(__import__('os').environ['S4_C'])
    for rep in range(int(__import__('os').environ.get('S4_START', '1')), 4):
        wave(c, rep)
