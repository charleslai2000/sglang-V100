#!/usr/bin/env python3
"""CPU-only lock protocol tests, including launcher duplicate rejection before workload hook."""
import os,subprocess,sys,tempfile
from pathlib import Path
from gpu_uuid_lock import acquire,release,assert_cuda_uuid
UUID='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
class FakeCuda:
 def __init__(self,uuid):self.uuid=uuid
 def device_count(self):return 1
 def get_device_properties(self,index):
  assert index==0
  return type('Device',(),{'uuid':self.uuid})()
class FakeTorch:
 def __init__(self,uuid):self.cuda=FakeCuda(uuid)

def main():
 moddir=str(Path(__file__).resolve().parent)
 with tempfile.TemporaryDirectory() as d:
  env={**os.environ,'PYTHONPATH':moddir+os.pathsep+os.environ.get('PYTHONPATH',''),'E035_GPU_LOCK_DIR':d,'CUDA_VISIBLE_DEVICES':''}
  os.environ['E035_GPU_LOCK_DIR']=d
  # The test launcher takes its lock before invoking a fake workload hook.
  parent_fd=acquire(UUID);marker=Path(d)/'workload.started'
  fake="from gpu_uuid_lock import acquire; acquire(%r); open(%r,'w').write('started')"%(UUID,str(marker))
  cmd=[sys.executable,'-c',fake]
  child=subprocess.run(cmd,env=env,capture_output=True,text=True)
  assert child.returncode!=0 and 'locked by another' in child.stderr and not marker.exists(),(child.returncode,child.stderr)
  # Child inherited lock remains held after abrupt owner-fd close.
  code="import os,time; time.sleep(.4); os.fstat(int(os.environ['TEST_LOCK_FD']))"
  inherited_env={**env,'TEST_LOCK_FD':str(parent_fd)}
  survivor=subprocess.Popen([sys.executable,'-c',code],env=inherited_env,pass_fds=(parent_fd,))
  os.close(parent_fd)
  duplicate=subprocess.run(cmd,env=env,capture_output=True,text=True)
  assert duplicate.returncode!=0 and 'locked by another' in duplicate.stderr,duplicate.stderr
  assert survivor.wait(timeout=3)==0
  fd=acquire(UUID);release(fd)
 assert_cuda_uuid(FakeTorch(UUID.removeprefix('GPU-')),UUID)
 try:assert_cuda_uuid(FakeTorch('deadbeef'),UUID)
 except RuntimeError as e:assert 'UUID mismatch' in str(e)
 else:raise AssertionError('wrong CUDA UUID was accepted')
 print('PASS: duplicate rejected before fake workload; surviving child retains flock; final owner releases; CUDA UUID allow/mismatch tested')
if __name__=='__main__':main()
