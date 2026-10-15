#!/usr/bin/env python3
"""Strict FP64/Dot2 versus exact-c57 test; one fresh process per arm."""
import argparse,hashlib,importlib.util,json,os,sys,torch
from pathlib import Path
from gpu_uuid_lock import assert_inherited,assert_cuda_uuid
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--binary',required=True);p.add_argument('--arm',required=True);p.add_argument('--M',type=int,required=True);p.add_argument('--arm-fixture',required=True);p.add_argument('--iters',type=int,default=40);a=p.parse_args()
u='GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189';fd=int(os.environ['E035_GPU_LOCK_FD']);assert_inherited(fd,u);assert_cuda_uuid(torch,u)
bin=Path(a.binary).resolve();module_name=('common_ops' if a.arm=='c57' else ('g022_fp64_optin' if 'optin' in bin.name else f'g022_numerical_fix_{"fp64" if "fp64" in bin.name else "dot2"}'));sp=importlib.util.spec_from_file_location(module_name,str(bin));mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
assert Path(mod.__file__).resolve()==bin
if a.arm=='c57':
 fn=lambda xx,qw_,qz_,sc_,g_,fp32: torch.ops.sgl_kernel.gptq_q4_fp32_m4_candidate(xx,qw_,qz_,sc_,g_,fp32)
else:
 fn=(torch.ops.sgl_kernel.gptq_q4_fp32_m4_numerical_fix if 'optin' in bin.name else torch.ops.numerical_fix.gptq)
root=Path(a.fixture);d=next(q for q in root.iterdir() if q.is_dir() and json.loads((q/'manifest.json').read_text())['prefix']==a.arm_fixture)
m=json.loads((d/'manifest.json').read_text());t=m['tensors']
def ld(n,dt):
 f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dt).element_size(),dtype=dt).view(t[n]['shape']).cuda()
x4=ld('x',torch.float16);qw=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1);x=x4[:a.M].contiguous();K,N=x.shape[1],qw.shape[1]
q=torch.empty((K,N),device='cuda',dtype=torch.int32)
for b in range(8):q[b::8]=(qw>>(4*b))&15
z=torch.empty((qz.shape[0],N),device='cuda',dtype=torch.int32)
for b in range(8):z[:,b::8]=(qz>>(4*b))&15
w64=(q.double()-(z[g.long()]+1).double())*sc.double()[g.long()];ref=(x.double()@w64).float();tol=1e-4+5e-5*ref.abs()

def check(y,reference=ref,limits=tol):
 torch.cuda.synchronize();diff=(y-reference).abs();ratio=diff/limits;bad=(diff>limits);coords=bad.nonzero().cpu().tolist()
 return {'strict_bad':int(bad.sum()),'max_abs':float(diff.max()),'max_tolerance_ratio':float(ratio.max()),'finite':bool(torch.isfinite(y).all()),'bad_points':[{'m':i,'n':j,'actual':float(y[i,j]),'reference':float(reference[i,j]),'error':float(diff[i,j]),'allowed':float(limits[i,j])} for i,j in coords]}
y32=fn(x,qw,qz,sc,g,True);stats=check(y32)
strict_gate_bad = (a.arm != 'c57' and stats['strict_bad'])
# Check final eager cast, repeat and independent scratch.
y=fn(x,qw,qz,sc,g,False);torch.cuda.synchronize();repeat=fn(x,qw,qz,sc,g,False);torch.cuda.synchronize();ind=fn(x.clone().contiguous(),qw,qz,sc,g,False);torch.cuda.synchronize();independent_fp32=fn(x.clone().contiguous(),qw,qz,sc,g,True);torch.cuda.synchronize();independent_fp32_stats=check(independent_fp32)
cast_fp32_bad=int((y.float()-y32).abs().gt(0).sum().item())
# Actual CUDA Graph FP32 oracle scratch and final cast graph.
gx=x.clone().contiguous();graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph): go=fn(gx,qw,qz,sc,g,True)
graph.replay();torch.cuda.synchronize();gstats=check(go)
# Separate captured FP16 execution.
gx2=x.clone().contiguous();graphh=torch.cuda.CUDAGraph()
with torch.cuda.graph(graphh): gy=fn(gx2,qw,qz,sc,g,False)
graphh.replay();torch.cuda.synchronize()
# 4x L43/down stress, on all arms: strict failure is fatal for repaired routes.
stress=None
if a.arm_fixture=='43.mlp.down_proj' and a.M==4:
 xs=(x*4).half().contiguous();rs=(xs.double()@w64).float();ts=1e-4+5e-5*rs.abs();ys=fn(xs,qw,qz,sc,g,True);stress_stats=check(ys,rs,ts);torch.cuda.synchronize();dd=(ys-rs).abs();rr=dd/ts;mask=dd>ts
 stress={'strict_bad':int(mask.sum()),'max_abs':float(dd.max()),'max_tolerance_ratio':float(rr.max()),'input_max_abs':float(xs.abs().max()),'input_sha256':hashlib.sha256(xs.cpu().contiguous().numpy().tobytes()).hexdigest(),'fp32_output_sha256':hashlib.sha256(ys.cpu().contiguous().numpy().tobytes()).hexdigest(),'reference_sha256':hashlib.sha256(rs.cpu().contiguous().numpy().tobytes()).hexdigest(),'bad_points':[{'m':i,'n':j,'actual':float(ys[i,j]),'reference':float(rs[i,j]),'error':float(dd[i,j]),'allowed':float(ts[i,j])} for i,j in mask.nonzero().cpu().tolist()]}
if a.arm!='c57' and (stats['strict_bad'] or gstats['strict_bad'] or independent_fp32_stats['strict_bad'] or (stress and stress['strict_bad'])):
 print(json.dumps({'arm':a.arm,'fixture':a.arm_fixture,'M':a.M,'binary':str(bin),'binary_sha256':hashlib.sha256(bin.read_bytes()).hexdigest(),'eager':stats,'graph':gstats,'stress':stress},indent=2),file=sys.stdout,flush=True);raise SystemExit(1)
assert torch.equal(y,repeat) and torch.equal(y,ind) and torch.equal(gy,y), 'cast/repeat/independent/Graph mismatch'
def bench(f):
 for _ in range(8):f()
 torch.cuda.synchronize();v=[]
 for _ in range(a.iters):
  s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();f();e.record();e.synchronize();v.append(s.elapsed_time(e)*1000)
 return sorted(v)
eager=bench(lambda:fn(x,qw,qz,sc,g,False));graph_times=bench(graphh.replay)
print(json.dumps({'arm':a.arm,'fixture':a.arm_fixture,'M':a.M,'K':K,'N':N,'binary':str(bin),'binary_sha256':hashlib.sha256(bin.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'eager':stats,'graph':gstats,'independent_scratch_fp32':independent_fp32_stats,'independent_scratch_fp32_sha256':hashlib.sha256(independent_fp32.cpu().contiguous().numpy().tobytes()).hexdigest(),'eager_cast_vs_fp32_nonzero':cast_fp32_bad,'graph_cast_vs_fp32_nonzero':int((gy.float()-go).abs().gt(0).sum().item()),'graph_fp32_sha256':hashlib.sha256(go.cpu().contiguous().numpy().tobytes()).hexdigest(),'stress':stress,'fp16_cast_finite':bool(torch.isfinite(y).all()),'graph_fp16_equal_eager':bool(torch.equal(gy,y)),'repeat_equal':bool(torch.equal(y,repeat)),'independent_scratch_equal':bool(torch.equal(y,ind)),'eager_median_us':eager[len(eager)//2],'graph_median_us':graph_times[len(graph_times)//2],'eager_samples_us':eager,'graph_samples_us':graph_times,'pid':os.getpid()}),flush=True)
