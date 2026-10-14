#!/usr/bin/env python3
"""Locked selected-down-matrix wrapper; paired_isolated owns the UUID lock."""
import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('fixtures','r1','parallel','c57','out'):p.add_argument('--'+key,required=True)
p.add_argument('--gpu-uuid',default='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189')
p.add_argument('--iters',type=int,default=20);p.add_argument('--pairs',type=int,default=3);p.add_argument('--prefix',action='append',required=True);a=p.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'',a.gpu_uuid):raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
script=Path(__file__).with_name('paired_isolated.py')
cmd=[sys.executable,str(script),'--fixtures',a.fixtures,'--old','','--r1',a.r1,'--parallel',a.parallel,'--c57',a.c57,'--gpu-uuid',a.gpu_uuid,'--out',a.out,'--iters',str(a.iters),'--pairs',str(a.pairs)]
for prefix in a.prefix:cmd.extend(['--only-prefix',prefix])
cp=subprocess.run(cmd,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid})
raise SystemExit(cp.returncode)
