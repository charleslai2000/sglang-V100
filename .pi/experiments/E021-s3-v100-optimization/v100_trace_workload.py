import concurrent.futures as cf, json, os, statistics, threading, time
from pathlib import Path
import requests
from transformers import AutoTokenizer
BASE=os.environ.get('S2_BASE','http://127.0.0.1:30003')
MODEL='/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4'
OUT=os.environ.get('S3_OUT')
if not OUT or not OUT.startswith('/hy-tmp/sglang-logs/s3-v100-'):
 raise RuntimeError('S3_OUT must be a dedicated /hy-tmp/sglang-logs/s3-v100-* path')
if Path(OUT).exists():
 raise RuntimeError(f'refusing to overwrite existing workload output: {OUT}')
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
def ids_for_len(n):
 q='\nQuestion: What is 6 times 7? Answer with the number only:'; qi=tok.encode(q,add_special_tokens=False); f=tok.encode('Context filler sentence. ',add_special_tokens=False); b=[]
 while len(b)+len(qi)<n:b.extend(f)
 return (b[:n-len(qi)]+qi)[:n]
def one(n, outlen, barrier=None):
 ids=ids_for_len(n)
 if barrier:barrier.wait()
 t0=time.perf_counter(); arrivals=[]; final=[]; status=None; err=None
 try:
  with requests.post(BASE+'/generate',json={'input_ids':ids,'sampling_params':{'temperature':0.0,'max_new_tokens':outlen,'ignore_eos':True},'stream':True},stream=True,timeout=1200) as r:
   status=r.status_code
   if status!=200:err=r.text[:1000]
   else:
    for line in r.iter_lines(decode_unicode=True):
     if not line or not line.startswith('data: '):continue
     d=line[6:]
     if d=='[DONE]':break
     x=json.loads(d).get('output_ids',[])
     if x and len(x)>len(final):arrivals.extend([time.perf_counter()]*(len(x)-len(final)));final=x
 except Exception as e:err=repr(e)
 itl=[arrivals[i]-arrivals[i-1] for i in range(1,len(arrivals))]
 return {'http':status,'error':err,'prompt':n,'requested':outlen,'output':len(final),'nonzero':sum(v!=0 for v in final),'ttft':arrivals[0]-t0 if arrivals else None,'mean_itl':statistics.mean(itl) if itl else None,'ids':final}
def group(name,n,outlen,c):
 Path(OUT).parent.mkdir(parents=True,exist_ok=True)
 b=threading.Barrier(c);t=time.perf_counter()
 with cf.ThreadPoolExecutor(max_workers=c) as ex:r=list(ex.map(lambda _:one(n,outlen,b),range(c)))
 wall=time.perf_counter()-t
 d={'tag':name,'prompt':n,'output':outlen,'concurrency':c,'wall_s':wall,'aggregate_tps':sum(x['output'] for x in r)/wall,'correct':all(x['http']==200 and x['output']==outlen and x['nonzero']>0 and not x['error'] for x in r),'requests':r}
 with open(OUT,'a') as f:f.write(json.dumps(d)+'\n')
 print(json.dumps({k:v for k,v in d.items() if k!='requests'}),flush=True)
 return d
# correctness + warmup: two rounds with state reset at new request boundary; then 3 measured 90s-ish waves
assert group('control',5,16,1)['correct']
for i in range(2):group('warmup',1024,32,8)
for i in range(3):
 r=group('decode-steady',1024,256,8)
 if not r['correct']:raise RuntimeError('decode correctness failed')
# Separate prefill trace cases: 8 requests one wave per exact prompt bucket, 8 generated tokens
for n in (1024,4096,7680):
 r=group('prefill',n,8,1)
 if not r['correct']:raise RuntimeError('prefill correctness failed')
