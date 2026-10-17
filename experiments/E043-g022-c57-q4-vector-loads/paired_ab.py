#!/usr/bin/env python3
import argparse,hashlib,importlib.machinery,json,os,statistics,subprocess,sys
from pathlib import Path
ROOT=Path('/home/charles/Workspaces/3rdparty/sglang-V100'); SCR=Path(__file__).with_name('test_operator.py'); OUT=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E043/export');OUT.mkdir(parents=True,exist_ok=True)
PY='/home/charles/miniconda3/envs/sglang-v100/bin/python'; UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';
def main():
 ap=argparse.ArgumentParser();ap.add_argument('projection',choices=['gate_up','down']);ap.add_argument('M',type=int,choices=[1,2,4]);ap.add_argument('--pairs',type=int,default=3);args=ap.parse_args(); assert args.pairs>=3
 # Ensure fixed known device and host guard before any new child.
 assert 'MemAvailable:' in open('/proc/meminfo').read() and int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))) >= 8388608
 vis=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True);assert UUID in vis
 rows=[]
 for pair in range(args.pairs):
  order=['c57','candidate'] if pair%2==0 else ['candidate','c57']
  pairrow={'pair':pair,'order':order}
  for arm in order:
   env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':UUID,'PYTHONPATH':str(ROOT/'sgl-kernel/python')+':'+str(ROOT/'python'),'CUDA_HOME':'/usr/local/cuda-12.8','CUDACXX':'/usr/local/cuda-12.8/bin/nvcc','PATH':'/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:/usr/bin:/bin','SGLANG_G017_Q4_FP32_CANDIDATE':'0','SGLANG_G017_Q4_FP32_CANDIDATE_PROBE':'0'})
   cmd=[PY,str(SCR),args.projection,str(args.M),'--arm',arm,'--mode','bench']
   out=subprocess.check_output(cmd,env=env,text=True);rec=json.loads(out.strip().splitlines()[-1]);pairrow[arm]=rec
  for mode in ('eager','graph'):
   a=pairrow['c57'][mode+'_median_us'];b=pairrow['candidate'][mode+'_median_us'];pairrow[mode+'_ratio']=b/a
  rows.append(pairrow)
  print(json.dumps(pairrow,sort_keys=True),flush=True)
 result={'event':'E043_PAIRED_COMPLETE_LINEAR','projection':args.projection,'M':args.M,'pairs':rows,'median_ratio_eager':statistics.median(r['eager_ratio'] for r in rows),'median_ratio_graph':statistics.median(r['graph_ratio'] for r in rows),'strict_pair_order':True}
 p=OUT/f'{args.projection}-m{args.M}-paired.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'summary':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'median_ratio_eager':result['median_ratio_eager'],'median_ratio_graph':result['median_ratio_graph']},sort_keys=True))
if __name__=='__main__':main()
