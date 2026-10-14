#!/usr/bin/env python3
"""Per-UUID locked Nsys capture of one representative full-call Eager Linear arm."""
import argparse,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
from gpu_uuid_lock import acquire,release,assert_target_idle,assert_cuda_uuid
from cost_capture_manifest import write_manifest_record,record_arm_start

p=argparse.ArgumentParser()
for k in ('fixtures','r1','parallel','c57','out'):p.add_argument('--'+k,required=True)
p.add_argument('--gpu-uuid',default='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189');a=p.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'',a.gpu_uuid):raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
assert_target_idle(a.gpu_uuid)
fd=acquire(a.gpu_uuid);rid=str(uuid.uuid4());base=Path(a.out);base.parent.mkdir(parents=True,exist_ok=True);manifest=base.with_name(base.name+'.'+rid+'.manifest.jsonl')
paths={'r1':Path(a.r1).resolve(),'parallel':Path(a.parallel).resolve(),'c57':Path(a.c57).resolve()}
try:
 with manifest.open('w') as f:
  write_manifest_record(f,{'event':'run_start','run_id':rid,'uuid':a.gpu_uuid,'pid':os.getpid(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_ns':time.time_ns()})
  for arm,binary in paths.items():
   # GPU UUID/resource preflight occurs before CUDA/PyTorch initialization.
   assert_target_idle(a.gpu_uuid)
   if int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0]) < 8388608:
    raise RuntimeError('MemAvailable fell below the 8-GiB hard floor')
   rep=base.with_name(base.name+f'.{rid}.{arm}')
   cmd=['/usr/local/cuda/bin/nsys','profile','--force-overwrite=true','--trace=cuda,nvtx','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','-o',str(rep),'--','/home/charles/miniconda3/envs/sglang-v100/bin/python',str(Path(__file__).with_name('cost_capture.py')),'--arm',arm,'--binary',str(binary),'--fixtures',a.fixtures,'--c57',str(paths['c57']),'--gpu-uuid',a.gpu_uuid,'--lock-fd',str(fd)]
   env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'E035_GPU_LOCK_FD':str(fd),'E035_RUN_ID':rid}
   # Validate the runtime-visible ordinal UUID before Nsight launches the workload.
   import torch
   cuda_uuid=assert_cuda_uuid(torch,a.gpu_uuid)
   del torch
   record_arm_start(f,rid,arm,a.gpu_uuid,cuda_uuid,binary,hashlib.sha256(binary.read_bytes()).hexdigest(),rep,time.time_ns())
   st=time.time_ns();cp=subprocess.run(cmd,text=True,capture_output=True,env=env,pass_fds=(fd,));en=time.time_ns()
   rec={'event':'arm_end','run_id':rid,'arm':arm,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'report_prefix':str(rep),'started_ns':st,'ended_ns':en,'exit_status':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
   write_manifest_record(f,rec)
   if cp.returncode:raise RuntimeError(f'nsys failed arm={arm}: {cp.stderr}')
   report=Path(str(rep)+'.nsys-rep')
   if not report.exists():raise RuntimeError(f'missing expected report {report}')
   for report_kind,args in [('gpu_trace',['cuda_gpu_trace','--format','csv']),('kern_sum',['cuda_gpu_kern_sum','--format','csv'])]:
    stats=Path(str(rep)+f'.{report_kind}.csv')
    sp=subprocess.run(['/usr/local/cuda/bin/nsys','stats','--force-export=true','--report',args[0],'--format','csv','--output',str(stats),str(report)],text=True,capture_output=True)
    write_manifest_record(f,{'event':'stats','arm':arm,'kind':report_kind,'exit_status':sp.returncode,'file':str(stats),'stdout':sp.stdout,'stderr':sp.stderr})
    if sp.returncode:raise RuntimeError(f'nsys stats failed: {sp.stderr}')
  write_manifest_record(f,{'event':'run_end','run_id':rid,'exit_status':0,'ended_ns':time.time_ns()})
 print(json.dumps({'run_id':rid,'manifest':str(manifest)}),flush=True)
finally:release(fd)
