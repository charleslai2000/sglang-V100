#!/usr/bin/env python3
"""Replay one captured real-operand GPTQ geometry repeatedly for targeted NCU.

This is an operator microbenchmark on exact tensors captured from production; it
is not a service-performance measurement. Fixture file hashes live in manifests.
"""
import argparse,json,torch,importlib.util
from pathlib import Path
from sgl_kernel.gemm import gptq_gemm
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--warmup',type=int,default=10);p.add_argument('--iters',type=int,default=100);p.add_argument('--candidate-module');a=p.parse_args();root=Path(a.fixture)
def load(n,dtype,shape):return torch.from_file(str(root/f'{n}.bin'),shared=False,size=int(torch.tensor(shape).prod()),dtype=dtype).view(*shape).to(device='cuda')
manifest=json.loads((root/'manifest.json').read_text());t=manifest['tensors'];x=load('x',torch.float16,t['x']['shape']);w=load('qweight',torch.uint32,t['qweight']['shape']);z=load('qzeros',torch.uint32,t['qzeros']['shape']);s=load('scales',torch.float16,t['scales']['shape']);g=load('g_idx',torch.int32,t['g_idx']['shape']);
if a.candidate_module:
 spec=importlib.util.spec_from_file_location('common_ops',a.candidate_module);candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
 def op():return torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(x,w.view(torch.int32),z.view(torch.int32),s,g)
 route='isolated candidate'
else:
 # Production route; use_shuffle=False keeps original GPTQ-v1 g_idx semantics.
 def op():return gptq_gemm(x,w,z,s,g,False,4)
 route='production use_shuffle=False'
y=op();torch.cuda.synchronize();actual_len=y.numel(); ref=load('s1_output',torch.float16,manifest['s1_output']['shape']);print(json.dumps({'prefix':manifest['prefix'],'x':list(x.shape),'qweight':list(w.shape),'qzeros':list(z.shape),'scales':list(s.shape),'g_idx':list(g.shape),'output':list(y.shape),'max_abs_vs_s1':(y-ref).abs().max().item(),'max_abs':y.abs().max().item(),'finite':bool(torch.isfinite(y).all().item())}))
for _ in range(a.warmup):op()
torch.cuda.synchronize();starts=torch.cuda.Event(enable_timing=True);ends=torch.cuda.Event(enable_timing=True);starts.record()
for _ in range(a.iters):op()
ends.record();torch.cuda.synchronize();print(json.dumps({'prefix':manifest['prefix'],'iters':a.iters,'elapsed_ms':starts.elapsed_time(ends),'mean_ms':starts.elapsed_time(ends)/a.iters,'m':x.shape[0],'n':w.shape[1],'k':x.shape[1],'route':route}))
