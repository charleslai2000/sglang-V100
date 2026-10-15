#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
from gpu_uuid_lock import acquire,release,assert_target_idle
p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--c57',required=True);p.add_argument('--fp64',required=True);p.add_argument('--dot2',required=True);p.add_argument('--arms',default='c57,fp64,dot2');p.add_argument('--out',required=True);p.add_argument('--pairs',type=int,default=3);p.add_argument('--iters',type=int,default=40);p.add_argument('--only-fixture',action='append');p.add_argument('--only-M',type=int,action='append');a=p.parse_args()
u='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';assert_target_idle(u);fd=acquire(u);rid=str(uuid.uuid4());root=Path(a.fixtures);dirs=sorted(d for d in root.iterdir() if d.is_dir() and (d/'manifest.json').exists());fixtures=[json.loads((d/'manifest.json').read_text())['prefix'] for d in dirs]
if a.only_fixture:fixtures=[f for f in fixtures if f in a.only_fixture]
if not fixtures:raise SystemExit(f'fixture filter matched none: {a.only_fixture}')
ms=a.only_M or [1,2,3,4]
if any(m not in (1,2,3,4) for m in ms):raise SystemExit(f'invalid M selection {ms}')
arms=[v.strip() for v in a.arms.split(',') if v.strip()];binaries={'c57':Path(a.c57).resolve(),'fp64':Path(a.fp64).resolve(),'dot2':Path(a.dot2).resolve()}
if not arms or any(v not in binaries for v in arms):raise SystemExit(f'bad arms {arms}')
raw=Path(a.out).with_name(Path(a.out).stem+f'.{rid}.jsonl');log=raw.with_suffix('.run.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 with log.open('w') as lf,raw.open('w') as rf:
  def rec(f,x):f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
  rec(lf,{'event':'start','run_id':rid,'uuid':u,'arms':{k:{'binary':str(binaries[k]),'sha256':sha(binaries[k])} for k in arms},'started_ns':time.time_ns(),'pid':os.getpid()})
  for fixture in fixtures:
   for M in ms:
    for pair in range(a.pairs):
     order=arms if pair%2==0 else list(reversed(arms));results={}
     for arm in order:
      if int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])<8388608:raise RuntimeError('MemAvailable below 8GiB')
      env={**os.environ,'CUDA_VISIBLE_DEVICES':u,'E035_GPU_LOCK_FD':str(fd),'E035_RUN_ID':rid}
      cmd=[sys.executable,str(Path(__file__).with_name('test_numerical_fix.py')),'--fixture',a.fixtures,'--binary',str(binaries[arm]),'--arm',arm,'--M',str(M),'--arm-fixture',fixture,'--iters',str(a.iters)]
      start=time.time_ns();cp=subprocess.run(cmd,env=env,pass_fds=(fd,),text=True,capture_output=True);end=time.time_ns()
      rec(lf,{'event':'arm','fixture':fixture,'M':M,'pair':pair+1,'arm':arm,'start_ns':start,'end_ns':end,'exit':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr,'binary_sha256':sha(binaries[arm])})
      if cp.returncode:
       try:detail=json.loads(cp.stdout.splitlines()[-1])
       except Exception:detail={'stdout':cp.stdout}
       rec(rf,{'kind':'failure','fixture':fixture,'M':M,'pair':pair+1,'arm':arm,'result':detail,'stderr':cp.stderr});raise RuntimeError(f'{arm} failed {fixture} M{M}: {cp.stderr[-400:]}')
      row=json.loads(cp.stdout.strip().splitlines()[-1]);results[arm]=row;rec(rf,{'kind':'result','fixture':fixture,'M':M,'pair':pair+1,'arm':arm,'result':row})
     for left in arms:
      for right in arms:
       if left<right and results[left]['stress'] and results[right]['stress']:
        if results[left]['stress']['input_sha256']!=results[right]['stress']['input_sha256'] or results[left]['stress']['reference_sha256']!=results[right]['stress']['reference_sha256']:raise RuntimeError('paired stress input/reference identity mismatch')
     rec(rf,{'kind':'pair_complete','fixture':fixture,'M':M,'pair':pair+1,'order':order,'arms':results})
     print(json.dumps({'fixture':fixture,'M':M,'pair':pair+1,'order':order,'time_us':{k:{'eager':results[k]['eager_median_us'],'graph':results[k]['graph_median_us']} for k in order},'bad':{k:results[k]['eager']['strict_bad'] for k in order}}),flush=True)
  rec(lf,{'event':'end','status':0,'run_id':rid,'ended_ns':time.time_ns(),'raw':str(raw)})
 print(json.dumps({'run_id':rid,'raw':str(raw),'runlog':str(log)}))
finally:release(fd)
