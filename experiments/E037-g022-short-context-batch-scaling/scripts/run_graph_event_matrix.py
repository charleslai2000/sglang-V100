#!/usr/bin/env python3
"""UUID-locked isolated Graph-event runs; sample GPU/host memory during each process."""
import argparse, os, subprocess, time, csv
from pathlib import Path
ROOT=Path('/home/charles/Workspaces/3rdparty/sglang-V100'); ART=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E037'); UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'; PY='/home/charles/miniconda3/envs/sglang-v100/bin/python'
p=argparse.ArgumentParser();p.add_argument('--batches',default='1,2,4,8,16,32');p.add_argument('--reps',type=int,default=3);a=p.parse_args()
for b in map(int,a.batches.split(',')):
 for rep in range(1,a.reps+1):
  host=int(subprocess.check_output("awk '/^MemAvailable:/ {print $2}' /proc/meminfo",shell=True,text=True)); assert host>=8388608, f'host floor: {host} kB'
  gpu=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=uuid,memory.free,memory.used','--format=csv,noheader'],text=True).strip();assert gpu.startswith(UUID) and int(gpu.split(',')[1].split()[0])>15000,gpu
  frac='0.99' if b==32 else ('0.95' if b>=16 else '0.85');max_tokens=max(8192,b*256) if b>=16 else max(4096,b*256);out_len=8
  args=[PY,str(ROOT/'experiments/E037-g022-short-context-batch-scaling/scripts/e037_graph_event_runner.py'),'--model-path','/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4','--trust-remote-code','--tp-size','1','--dtype','float16','--mem-fraction-static',frac,'--max-running-requests',str(b),'--max-total-tokens',str(max_tokens),'--attention-backend','tilelang_fa_v100','--disable-radix-cache','--mamba-scheduler-strategy','no_buffer','--batch-size',str(b),'--input-len','128','--output-len',str(out_len),'--page-size','1','--run-name',f'E037-GraphEvent-b{b}-rep{rep}','--result-filename',str(ART/f'graph-event-b{b}-rep{rep}.jsonl'),'--cuda-graph-max-bs',str(b),'--cuda-graph-bs',str(b),'--disable-piecewise-cuda-graph']
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,PYTHONPATH=f'{ROOT}/sgl-kernel/python:{ROOT}/python',CUDA_HOME='/usr/local/cuda-12.8',CUDACXX='/usr/local/cuda-12.8/bin/nvcc',SGLANG_SM70_JIT_HEADROOM_GB='0.25' if b==32 else ('1.0' if b==16 else '2.0'))
  log=ART/f'graph-event-b{b}-rep{rep}.log'; mon=ART/f'graph-event-b{b}-rep{rep}-resources.csv';print(f'START b={b} rep={rep} host_kB={host} gpu={gpu}',flush=True)
  with open(log,'w') as f,open(mon,'w',newline='') as mf:
   w=csv.writer(mf);w.writerow(['unix','memavailable_kB','gpu_uuid','memory_used_MiB','memory_free_MiB','gpu_util_pct','sm_clock_MHz','mem_clock_MHz'])
   q=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   while q.poll() is None:
    h=int(subprocess.check_output("awk '/^MemAvailable:/ {print $2}' /proc/meminfo",shell=True,text=True)); vals=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True).strip().split(',');w.writerow([time.time(),h,*[x.strip() for x in vals]]);mf.flush()
    if h<8388608:
     os.killpg(q.pid,15);time.sleep(3)
     if q.poll() is None:os.killpg(q.pid,9)
     raise SystemExit(f'ABORTED host floor breach: {h}kB')
    time.sleep(.25)
  rc=q.wait();print(f'END b={b} rep={rep} rc={rc} host_kB={int(subprocess.check_output("awk \'/^MemAvailable:/ {print $2}\' /proc/meminfo",shell=True,text=True).strip())} gpu='+subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.used,memory.free','--format=csv,noheader'],text=True).strip(),flush=True)
  if rc:raise SystemExit(rc)
