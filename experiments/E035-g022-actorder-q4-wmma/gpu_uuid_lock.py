#!/usr/bin/env python3
"""Cooperative advisory lock for one explicitly named GPU UUID.

The launcher parent keeps this fd for the complete run and supervises every
child. flock is released by the kernel only when all inherited/open copies
close; children are spawned with the lock fd explicitly preserved. CPU-only
unit tests can exercise the protocol without importing CUDA/PyTorch.
"""
from __future__ import annotations
import fcntl, os, re, subprocess
from pathlib import Path
from typing import Optional

def lock_path(uuid: str) -> Path:
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+',uuid):
        raise ValueError(f'invalid GPU UUID: {uuid!r}')
    directory=Path(os.environ.get('E035_GPU_LOCK_DIR','/tmp/e035-gpu-locks'))
    directory.mkdir(parents=True,exist_ok=True)
    return directory/(uuid+'.lock')

def assert_cuda_uuid(torch, uuid: str):
    """Bind CUDA's runtime-visible ordinal zero to the requested physical UUID."""
    if torch.cuda.device_count()!=1:raise RuntimeError(f'expected exactly one CUDA-visible device, got {torch.cuda.device_count()}')
    actual=str(torch.cuda.get_device_properties(0).uuid).lower()
    expected=uuid.lower().removeprefix('gpu-')
    if actual!=expected:raise RuntimeError(f'CUDA ordinal 0 UUID mismatch: {actual} != {expected}')
    return actual

def acquire(uuid: str):
    """Return an exclusively locked fd; fail immediately if another cohort owns it."""
    path=lock_path(uuid)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f'target GPU is locked by another cooperative E035 run: {uuid}')
    os.ftruncate(fd,0)
    os.write(fd,f'pid={os.getpid()} uuid={uuid}\n'.encode())
    os.fsync(fd)
    return fd

def assert_inherited(fd: int, uuid: str):
    """Verify this process inherited the lock file's held open file description."""
    path=lock_path(uuid)
    st=os.fstat(fd);pst=os.stat(path)
    if (st.st_dev,st.st_ino)!=(pst.st_dev,pst.st_ino):raise RuntimeError('inherited UUID lock fd points to wrong lock file')
    probe=os.open(path,os.O_RDWR)
    try:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(probe,fcntl.LOCK_UN)
        raise RuntimeError('inherited UUID lock fd does not hold an exclusive lock')
    finally:os.close(probe)

def assert_target_idle(uuid: str):
    """Reject visible compute PIDs on exactly the target UUID; never inspect/alter other GPUs."""
    out=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True,stderr=subprocess.STDOUT)
    if out.strip():raise RuntimeError(f'target GPU has unrelated/current compute owner(s); refusing without termination: {out.strip()}')

def release(fd: Optional[int]):
    if fd is not None:
        try: fcntl.flock(fd,fcntl.LOCK_UN)
        finally: os.close(fd)
