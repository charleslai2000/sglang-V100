#!/usr/bin/env python3
"""Correctness/Graph/high-amplitude qualification for isolated group-parallel candidate."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True);p.add_argument('--build',required=True);p.add_argument('--module');p.add_argument('--only-prefix');p.add_argument('--stress',action='store_true');a=p.parse_args()
if a.module:
 s=importlib.util.spec_from_file_location('e035_grouped_q4_parallel_sm70',a.module);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
else:
 os.environ['MAX_JOBS']='1'
 mod=load(name='e035_grouped_q4_parallel_sm70',sources=[str(Path(__file__).with_name('grouped_q4_sm70_parallel.cu'))],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory=a.build,verbose=True)
root=Path(a.fixtures);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);rows=[]
for d in sorted(root.iterdir()):
 if not d.is_dir() or not (d/'manifest.json').exists():continue
 m=json.loads((d/'manifest.json').read_text())
 if a.only_prefix and m['prefix']!=a.only_prefix:continue
 t=m['tensors']
 def ld(n,dtype):
  f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(t[n]['shape']).cuda()
 x4=ld('x',torch.float16);qw0=ld('qweight',torch.int32);qz=ld('qzeros',torch.int32);sc0=ld('scales',torch.float16);g=ld('g_idx',torch.int32).reshape(-1)
 K,N=qw0.shape[0]*8,qw0.shape[1];G=sc0.shape[0]
 perm=torch.argsort(g,stable=True).int();q=torch.empty((K,N),device='cuda',dtype=torch.int32)
 for b in range(8):q[b::8]=(qw0>>(4*b))&15
 qg=q.index_select(0,perm);packed=torch.zeros((K//8,N),device='cuda',dtype=torch.int32)
 for b in range(8):packed|=qg[b::8]<<(4*b)
 z=torch.empty((qz.shape[0],qz.shape[1]*8),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>>(4*b))&15
 w64=(q.double()-(z[g.long()]+1).double())*sc0.double()[g.long()]
 for M in (1,2,4):
  x=x4[:M].contiguous();sc=sc0.contiguous();partial=torch.empty((G,M,N),device='cuda',dtype=torch.float32);gx=torch.empty_like(x);outbuf=torch.empty((M,N),device='cuda',dtype=torch.float32)
  def run(fp32=True,xx=x,pp=partial,gg=gx,oo=outbuf):return mod.grouped_q4_sm70_parallel(xx,packed,sc,perm,pp,gg,oo,fp32)
  y=run(True);torch.cuda.synchronize();ref=(x.double()@w64).float();tol=1e-4+5e-5*ref.abs();bad=int(((y-ref).abs()>tol).sum().item())
  cast=y.half();finite=bool(torch.isfinite(y).all().item() and torch.isfinite(cast).all().item())
  yr=run(True);repeat=bool(torch.equal(y,yr))
  p2=torch.empty_like(partial);g2=torch.empty_like(gx);o2=torch.empty_like(outbuf);y2=run(True,x,p2,g2,o2);independent=bool(torch.equal(y,y2))
  graph=[];xg=x.clone().contiguous();gp=torch.empty_like(partial);gg=torch.empty_like(gx);go=torch.empty_like(outbuf)
  st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(st):
   gr=torch.cuda.CUDAGraph()
   with torch.cuda.graph(gr,stream=st):gy=mod.grouped_q4_sm70_parallel(xg,packed,sc,perm,gp,gg,go,False)
   gr.replay();torch.cuda.synchronize()
  torch.cuda.current_stream().wait_stream(st)
  graph_bad=int(((go-ref).abs()>tol).sum().item());cast_bad=int((gy!=go.half()).sum().item());equal=bool(torch.equal(gy,y.half()))
  graph.append({'graph_key':M,'actual_M':M,'fp32_go_strict_bad':graph_bad,'fp32_go_finite':bool(torch.isfinite(go).all().item()),'fp16_cast_bad':cast_bad,'graph_eager_cast_equal':equal,'output_dtype':str(gy.dtype)})
  stress=[]
  if a.stress and m['prefix']=='43.mlp.down_proj' and M==4:
   for factor in (1.,4.):
    xs=(x*factor).half().contiguous();ss=torch.empty_like(partial);sg=torch.empty_like(gx);so=torch.empty_like(outbuf)
    ys=mod.grouped_q4_sm70_parallel(xs,packed,sc,perm,ss,sg,so,True);rr=(xs.double()@w64).float();tt=1e-4+5e-5*rr.abs();hs=ys.half()
    stress.append({'factor':factor,'input_max_abs':float(xs.abs().max().item()),'strict_bad':int(((ys-rr).abs()>tt).sum().item()),'fp32_finite':bool(torch.isfinite(ys).all().item()),'fp16_finite':bool(torch.isfinite(hs).all().item())})
  row={'fixture':m['prefix'],'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'M':M,'K':K,'N':N,'G':G,'binary':str(Path(mod.__file__).resolve()),'binary_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),'oracle':'FP64 GPTQ matmul -> FP32; abs(error)<=1e-4+5e-5*abs(ref)','fp32_strict_bad':bad,'fp32_max_abs_error':float((y-ref).abs().max().item()),'fp32_finite':finite,'fp16_finite':bool(torch.isfinite(cast).all().item()),'same_scratch_repeat_equal':repeat,'independent_scratch_equal':independent,'graph':graph,'partial_elements':G*M*N,'partial_bytes':G*M*N*4,'stress':stress}
  rows.append(row);print(json.dumps(row),flush=True)
out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
