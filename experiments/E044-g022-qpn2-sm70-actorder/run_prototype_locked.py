#!/usr/bin/env python3
"""E044 target-only V100 lock and resource-gated prototype launcher."""
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'E035-g022-actorder-q4-wmma'))
from gpu_uuid_lock import acquire, assert_target_idle, release
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
assert_target_idle(UUID)
mem_kb=int(next(line.split()[1] for line in open('/proc/meminfo') if line.startswith('MemAvailable:')))
if mem_kb < 8*1024*1024: raise SystemExit(f'abort: MemAvailable {mem_kb}kB < 8GiB')
fd=acquire(UUID)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=UUID;env['MAX_JOBS']='1';env['CUDA_HOME']='/usr/local/cuda-12.8'
try:
 p=subprocess.run(['/home/charles/miniconda3/envs/sglang-v100/bin/python',str(Path(__file__).with_name('prototype_qpn.py')),'--stress'],env=env,pass_fds=(fd,),text=True)
 if p.returncode:raise SystemExit(p.returncode)
finally:release(fd)
