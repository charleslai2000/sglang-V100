#!/usr/bin/env python3
"""Run E044 six-fixture qualification with cooperative V100 lock and memory gate."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'E035-g022-actorder-q4-wmma'))
from gpu_uuid_lock import acquire,assert_target_idle,release
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
assert_target_idle(UUID)
mem=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))
if mem<8*1024*1024:raise SystemExit(f'abort: MemAvailable={mem} kB')
fd=acquire(UUID);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,CUDA_HOME='/usr/local/cuda-12.8',MAX_JOBS='1')
try:
 p=subprocess.run(['/home/charles/miniconda3/envs/sglang-v100/bin/python',str(ROOT/'qualify_six.py')],env=env,pass_fds=(fd,),text=True)
 if p.returncode:raise SystemExit(p.returncode)
finally:release(fd)
