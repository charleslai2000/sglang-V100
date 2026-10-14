#!/usr/bin/env python3
"""Run selected down fixture pairs serially through paired_isolated, one V100 owner."""
import argparse,json,subprocess,sys,os
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('fixtures','r1','parallel','c57','out'):p.add_argument('--'+key,required=True)
p.add_argument('--iters',type=int,default=20);p.add_argument('--pairs',type=int,default=3);p.add_argument('--prefix',action='append',required=True);a=p.parse_args()
script=Path(__file__).with_name('paired_isolated.py');out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
for prefix in a.prefix:
 for M in (1,2,4):
  cmd=[sys.executable,str(script),'--fixtures',a.fixtures,'--old','','--r1',a.r1,'--parallel',a.parallel,'--c57',a.c57,'--out',str(out),'--iters',str(a.iters),'--pairs',str(a.pairs),'--only-prefix',prefix,'--only-M',str(M)]
  cp=subprocess.run(cmd,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189')})
  if cp.returncode:raise SystemExit(f'failed prefix={prefix} M={M} exit={cp.returncode}')
