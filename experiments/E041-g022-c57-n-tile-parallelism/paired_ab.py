#!/usr/bin/env python3
"""Three cyclic fresh-process complete operator A/B pairs, native-FP16 c57 vs NTile."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/charles/Workspaces/3rdparty/sglang-V100');E=ROOT/'experiments/E041-g022-c57-n-tile-parallelism'
PY='/home/charles/miniconda3/envs/sglang-v100/bin/python';ENV=os.environ.copy();ENV['CUDA_VISIBLE_DEVICES']='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';ENV['PYTHONPATH']=str(ROOT/'sgl-kernel/python')+':'+str(ROOT/'python')
def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--M',type=int,required=True);p.add_argument('--candidate',choices=['ntile128','ntile64','ntile32'],required=True);a=p.parse_args()
 out=[]
 for rep in range(1,4):
  for arm in ('c57',a.candidate):
   cmd=[PY,str(E/'test_ntile.py'),'--fixture',a.fixture,'--M',str(a.M),'--arm',arm,'--ab']
   t=time.time();r=subprocess.run(cmd,env=ENV,text=True,capture_output=True,timeout=300)
   if r.returncode: print(r.stdout);print(r.stderr,file=sys.stderr);raise SystemExit(r.returncode)
   j=json.loads(r.stdout.strip().splitlines()[-1]);j.update({'rep':rep,'order':len(out),'wall_s':time.time()-t});out.append(j)
 print(json.dumps({'fixture':a.fixture,'M':a.M,'candidate':a.candidate,'records':out},sort_keys=True))
if __name__=='__main__':main()
