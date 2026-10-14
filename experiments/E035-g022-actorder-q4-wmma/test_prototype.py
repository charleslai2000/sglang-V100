#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np, torch
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True);p.add_argument('--build',required=True);p.add_argument('--module');p.add_argument('--only-prefix');p.add_argument('--iters',type=int,default=40);p.add_argument('--stress',action='store_true');a=p.parse_args()
root=Path(a.fixtures);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if a.module:
 spec=importlib.util.spec_from_file_location('e035_grouped_q4_sm70',a.module);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
else:
 os.environ['MAX_JOBS']='1'
 mod=load(name='e035_grouped_q4_sm70',sources=[str(Path(__file__).with_name('grouped_q4_sm70.cu'))],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_70,code=sm_70'],extra_cflags=['-O2'],build_directory=a.build,verbose=True)
rows=[]
for d in sorted(root.iterdir()):
 if not d.is_dir() or not (d/'manifest.json').exists():continue
 m=json.loads((d/'manifest.json').read_text())
 if a.only_prefix and m['prefix']!=a.only_prefix:continue
 def load_t(n,dtype):
  td=m['tensors'][n];f=d/(n+'.bin');return torch.from_file(str(f),shared=False,size=f.stat().st_size//torch.empty((),dtype=dtype).element_size(),dtype=dtype).view(td['shape']).cuda()
 x4=load_t('x',torch.float16);qw0=load_t('qweight',torch.int32);qz=load_t('qzeros',torch.int32);s0=load_t('scales',torch.float16);g=load_t('g_idx',torch.int32).reshape(-1)
 # deterministic group sort and Q4 repack; group API tensor transformations stay GPU-side.
 perm=torch.argsort(g,stable=True).to(torch.int32);gs=g[perm]
 if int(g.min())<0 or int(g.max())>=s0.shape[0] or not torch.equal(gs,torch.arange(s0.shape[0],device='cuda').repeat_interleave(128)):
  raise RuntimeError('g_idx does not cover exactly 128 positions/group')
 z=torch.empty((qz.shape[0],qz.shape[1]*8),device='cuda',dtype=torch.int32)
 for b in range(8):z[:,b::8]=(qz>> (4*b))&15
 if not torch.all(z==7):raise RuntimeError('qzero fastpath qualification failed')
 q=torch.empty((x4.shape[1],qw0.shape[1]),device='cuda',dtype=torch.int32)
 for b in range(8):q[b::8]=(qw0>> (4*b))&15
 qg=q.index_select(0,perm);qw=torch.zeros((qg.shape[0]//8,qg.shape[1]),device='cuda',dtype=torch.int32)
 for b in range(8):qw|=qg[b::8]<<(4*b)
 scales=s0.index_select(0,torch.arange(s0.shape[0],device='cuda'))
 K,N=x4.shape[1],qw.shape[1];G=scales.shape[0]
 # Validate grouped repack exactly reconstructs source Q and g-index linkage, with boundary samples.
 qback=torch.empty_like(qg)
 for b in range(8):qback[b::8]=(qw>> (4*b))&15
 code_bad=int((qback!=qg).sum().item());
 if code_bad:raise RuntimeError(f'Q4 roundtrip mismatch {code_bad}')
 for M in (1,2,4):
  x=x4[:M].contiguous();part=torch.empty((M,N),device='cuda',dtype=torch.float32);gx=torch.empty_like(x);outbuf=torch.empty_like(part)
  y=mod.grouped_q4_sm70(x,qw,scales,perm,part,gx,outbuf,True);torch.cuda.synchronize()
  # actual output accumulator from prototype; build FP64 GPTQ reference.
  w64=(q.double()-(z[g.long()]+1).double())*s0.double()[g.long()]
  ref=(x.double()@w64).float();lim=1e-4+5e-5*ref.abs();bad=int(((y-ref).abs()>lim).sum().item());maxerr=float((y-ref).abs().max().item())
  yh=y.half();cast_bad=int((yh.float()!=y.half().float()).sum().item());finite=bool(torch.isfinite(y).all().item())
  stress=[]
  if a.stress and m['prefix']=='43.mlp.down_proj' and M==4:
   for factor in (1.,4.):
    xs=(x*factor).to(torch.float16).contiguous(); gsx=torch.empty_like(xs); scratch=torch.empty_like(outbuf); so=torch.empty_like(outbuf)
    ys=mod.grouped_q4_sm70(xs,qw,scales,perm,scratch,gsx,so,True)
    wr=(q.double()-(z[g.long()]+1).double())*s0.double()[g.long()]
    rs=(xs.double()@wr).float();ls=1e-4+5e-5*rs.abs()
    hs=ys.half()
    stress.append({'factor':factor,'input_max_abs':float(xs.abs().max().item()),'fp32_bad':int(((ys-rs).abs()>ls).sum().item()),'fp32_max_abs_err':float((ys-rs).abs().max().item()),'fp32_finite':bool(torch.isfinite(ys).all().item()),'fp16_finite':bool(torch.isfinite(hs).all().item()),'fp16_max_abs':float(hs.abs().max().item())})
  # deterministic repeat with same and distinct scratch; fixed-shape op is graph capturable.
  y2=mod.grouped_q4_sm70(x,qw,scales,perm,part,gx,outbuf,True);same_scratch_equal=bool(torch.equal(y,y2))
  part2=torch.empty_like(part);gx2=torch.empty_like(gx);out2=torch.empty_like(outbuf);y3=mod.grouped_q4_sm70(x,qw,scales,perm,part2,gx2,out2,True);independent_equal=bool(torch.equal(y,y3))
  graph=[]
  for key in (1,2,4):
   if M!=key:continue
   x_graph=x.clone().contiguous()
   gpart=torch.empty_like(part);ggx=torch.empty_like(gx);go=torch.empty_like(outbuf)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    gr=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr,stream=stream):gy=mod.grouped_q4_sm70(x_graph,qw,scales,perm,gpart,ggx,go,True)
    gr.replay();gr.replay()
   torch.cuda.current_stream().wait_stream(stream);graph.append({'graph_key':key,'actual_M':M,'matches_accumulator':bool(torch.equal(gy,y)),'matches_oracle_strict_bad':int(((gy-ref).abs()>lim).sum().item())})
  # Complete operator latency includes gather, group WMMA, scale/reduce, final cast and fixed scratch supplied above.
  def timed(dtype_out):
   fn=lambda:mod.grouped_q4_sm70(x,qw,scales,perm,part,gx,outbuf,dtype_out)
   for _ in range(5):fn()
   torch.cuda.synchronize();vals=[]
   for _ in range(a.iters):
    st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();fn();en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000)
   return {'median_us':float(torch.tensor(vals).median()),'p10_us':float(torch.quantile(torch.tensor(vals),.1)),'p90_us':float(torch.quantile(torch.tensor(vals),.9)),'n':len(vals)}
  row={'fixture':m['prefix'],'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'M':M,'K':K,'N':N,'groups':G,'zero_stored_7_all':True,'gidx_group_sorted_exact':True,'q4_repack_bad_codes':code_bad,'fp32_bad':bad,'max_abs_error':maxerr,'fp16_cast_finite':bool(torch.isfinite(yh).all().item()),'fp32_finite':finite,'same_scratch_repeat_bitwise':same_scratch_equal,'independent_scratch_bitwise':independent_equal,'graph':graph,'timing_fp32':timed(True),'timing_fp16_cast':timed(False),'partial_bytes':part.numel()*part.element_size(),'gather_bytes':gx.numel()*gx.element_size(),'high_amplitude_stress':stress}
  rows.append(row);print(json.dumps(row),flush=True)
(out/'correctness-timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
