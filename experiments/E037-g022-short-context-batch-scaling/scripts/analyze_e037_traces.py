#!/usr/bin/env python3
"""Offline E037 trace/event reconciliation; no CUDA imports or device access."""
import collections, gzip, hashlib, json, statistics
from pathlib import Path

EXT = Path('/mnt/data/pc01/sglang-v100-migration/experiments/E037')
G022 = Path('/mnt/data/pc01/sglang-v100-migration/experiments/G022')
UUID = 'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'

def union(intervals):
    out=[]
    for a,b in sorted((float(a),float(b)) for a,b in intervals if b>a):
        if out and a<=out[-1][1]: out[-1]=(out[-1][0],max(out[-1][1],b))
        else: out.append((a,b))
    return out

def owner(name,cat,args):
    args=args or {}; grid=args.get('grid') or []; n=name
    if cat in ('gpu_memcpy','gpu_memset') or n.startswith(('memcpy','memset')): return 'memcpy/memset'
    if 'gemm_half_q_half_alt_4bit_kernel' in n or 'reconstruct_gptq_kernel' in n or 'volta_h884gemm_' in n: return 'GPTQ gate_up/down'
    if 'gemmSN_TN_kernel<float' in n: return 'lm_head FP32 GEMV'
    if 'gemv2T_kernel_val' in n:
        # Exact source correlation for E037 eager b1 is not encoded in this compact analyzer.
        return 'nonquant GEMV (E031 source-mapped QKV at M1; otherwise UNKNOWN)'
    if 'cutlass::Kernel2<' in n or 'cublasLt::splitKreduce_kernel<' in n: return 'nonquant GEMM/CUTLASS'
    if n in ('_fwd_grouped_kernel_stage1','_fwd_kernel_stage2') or any(x in n for x in ('fused_rope_kernel<','store_kvcache<')): return 'Attention'
    if n=='_selective_scan_update_kernel' or 'causal_conv1d_update_kernel<' in n: return 'Mamba'
    if any(x in n for x in ('elementwise_kernel<','vectorized_elementwise_kernel<','unrolled_elementwise_kernel<','FusedAddRMSNormKernel<','RMSNormKernel<','act_and_mul_kernel<')): return 'Norm/elementwise'
    return 'UNKNOWN'

def trace_row(path,b):
    with gzip.open(path,'rt') as f: doc=json.load(f)
    ev=doc['traceEvents']; st=next(x for x in ev if x.get('cat')=='gpu_user_annotation' and f'DECODE bs={b}' in x.get('name',''))
    lo=float(st['ts']); hi=lo+float(st['dur']); acts=[]; raw=collections.Counter(); sums=collections.Counter(); events=collections.defaultdict(list)
    for e in ev:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset') or not isinstance(e.get('ts'),(int,float)) or not isinstance(e.get('dur'),(int,float)) or e['dur']<0 or e['ts']>=hi or e['ts']+e['dur']<=lo: continue
        a=max(lo,float(e['ts'])); z=min(hi,float(e['ts'])+float(e['dur'])); o=owner(e.get('name',''),e.get('cat',''),e.get('args'))
        acts.append((a,z,o)); events[o].append((a,z)); sums[o]+=float(e['dur']); raw[o]+=1
    allu=union([(a,z) for a,z,_ in acts]); owners={o:union(v) for o,v in events.items()}
    points=[]
    for o,iv in owners.items():
        for a,z in iv: points.extend(((a,1,o),(z,-1,o)))
    points.sort(key=lambda x:(x[0],x[1])); active=set(); prev=None; ex=collections.Counter()
    for t,d,o in points:
        if prev is not None and t>prev and len(active)==1: ex[next(iter(active))]+=t-prev
        if d<0: active.discard(o)
        else: active.add(o)
        prev=t
    gptq=[e for e in ev if e.get('cat')=='kernel' and lo<=e.get('ts',-1) and e.get('ts',-1)+e.get('dur',0)<=hi and any(k in e.get('name','') for k in ('gemm_half_q_half_alt_4bit_kernel','reconstruct_gptq_kernel','volta_h884gemm_'))]
    c=collections.Counter(e['name'].split('(')[0] for e in gptq)
    return {'batch':b,'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'annotation_ms':st['dur']/1000,'gpu_activity_union_ms':sum(z-a for a,z in allu)/1000,'activity_coverage_pct':100*sum(z-a for a,z in allu)/float(st['dur']),'activity_interval_gaps_ms':(float(st['dur'])-sum(z-a for a,z in allu))/1000,'owner_interval_union_ms':{o:sum(z-a for a,z in iv)/1000 for o,iv in owners.items()},'owner_exclusive_ms':{o:v/1000 for o,v in ex.items()},'owner_kernel_duration_sum_ms':{o:v/1000 for o,v in sums.items()},'owner_calls':dict(raw),'gptq_kernel_calls_by_symbol':dict(c)}

def main():
    out={'schema':'E037-offline-attribution-v1','device_uuid':UUID,'interpretation':'kineto GPU interval scope only; annotation duration is not production Graph Event latency; kernel sums are not additive wall time'}
    rows=[]
    for b in (1,2,4,8,16,32):
        p=next(EXT.glob(f'prof-b{b}_batch{b}_input128_output8_decode.trace.json.gz')); rows.append(trace_row(p,b))
    out['eager_instrumented_traces']=rows
    ge=[]
    for b in (1,2,4,8,16,32):
        reps=[]
        for r in (1,2,3):
            log=(EXT/f'graph-event-b{b}-rep{r}.log').read_text(errors='replace')
            line=next(l for l in log.splitlines() if 'e037_graph_decode_cuda_events' in l); ev=json.loads(line[line.index('{'):])['e037_graph_decode_cuda_events']; v=[e['t']*1000 for e in ev if e['timer_key']>=5]
            cap=next(l for l in log.splitlines() if 'Capture cuda graph bs' in l)
            med=statistics.median(v); dev=[abs(x-med) for x in v]; ordered=sorted(v)
            reps.append({'repeat':r,'graph_key':int(cap.rsplit('[',1)[1].split(']',1)[0]),'steady_events_n':len(v),'median_ms':med,'min_ms':min(v),'max_ms':max(v),'mad_ms':statistics.median(dev),'p90_ms':ordered[int(.9*(len(ordered)-1))],'gpu_tok_s':b/(med/1000),'log_sha256':hashlib.sha256((EXT/f'graph-event-b{b}-rep{r}.log').read_bytes()).hexdigest()})
        medians=[x['median_ms'] for x in reps]
        ge.append({'batch':b,'repeats':reps,'pooled_median_ms':statistics.median(medians),'across_repeat_mad_ms':statistics.median(abs(x-statistics.median(medians)) for x in medians)})
    out['graph_cuda_events']=ge
    # Capture the full node graph trace boundary (actual b4 only) for cross-mode attribution.
    gp=EXT/'graph-node-b4_batch4_input128_output8_decode.trace.json.gz'
    if gp.exists(): out['existing_graph_node_b4']=trace_row(gp,4)
    path=Path(__file__).resolve().parents[1]/'e037-analysis.json'
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(path,hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
