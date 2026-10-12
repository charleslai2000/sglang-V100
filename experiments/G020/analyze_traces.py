#!/usr/bin/env python3
"""Offline attribution of archived G019 PyTorch/CUPTI Chrome traces. No CUDA use."""
import argparse, collections, gzip, hashlib, json, os, glob
from pathlib import Path

ROOT = Path('/mnt/data/pc01/sglang-v100-migration/experiments/G019')
OWNER_ORDER = ['GPTQ gate_up', 'GPTQ down', 'Attention', 'Mamba', 'Other GEMM', 'Elementwise / normalization', 'Memcpy / memset', 'UNKNOWN']

def classify(name, cat, args=None):
    args = args or {}; grid = args.get('grid') or []
    is_gptq = name.startswith('sglang::gptq::gemm_half_q_half_alt_4bit_kernel') or name.startswith('(anonymous namespace)::packed_fp32_partial') or name.startswith('void (anonymous namespace)::packed_fp32_reduce')
    if is_gptq and grid:
        if grid[0] in (192, 96, 384): return 'GPTQ gate_up'
        if grid[0] in (24, 12, 48): return 'GPTQ down'
    if is_gptq: return 'UNKNOWN'
    if cat in ('gpu_memcpy','gpu_memset') or name.startswith(('memcpy','memset')): return 'Memcpy / memset'
    if name in ('_fwd_grouped_kernel_stage1','_fwd_kernel_stage2') or 'fused_rope_kernel<' in name or 'store_kvcache<' in name: return 'Attention'
    if name == '_selective_scan_update_kernel' or 'causal_conv1d_update_kernel<' in name: return 'Mamba'
    # Generic CUBLAS GEMV deliberately stays UNKNOWN: callsite is not identified.
    if any(x in name for x in ('cutlass::Kernel2<','cublasLt::splitKreduce_kernel<')): return 'Other GEMM'
    if any(x in name for x in ('elementwise_kernel<','vectorized_elementwise_kernel<','unrolled_elementwise_kernel<','FusedAddRMSNormKernel<','RMSNormKernel<','act_and_mul_kernel<')): return 'Elementwise / normalization'
    return 'UNKNOWN'

def interval_union(intervals):
    xs=sorted((float(a),float(b)) for a,b in intervals if b>a); out=[]; start=end=None
    for a,b in xs:
        if start is None: start,end=a,b
        elif a<=end: end=max(end,b)
        else: out.append((start,end)); start,end=a,b
    if start is not None: out.append((start,end))
    return out

def overlap_duration(a,b):
    i=j=total=0
    while i<len(a) and j<len(b):
        total+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return total

def analyze(path):
    with gzip.open(path,'rt') as f: doc=json.load(f)
    events=doc['traceEvents']; kernels=[e for e in events if e.get('cat')=='kernel' and isinstance(e.get('ts'),(int,float)) and isinstance(e.get('dur'),(int,float)) and e['dur']>=0]
    steps=sorted((e for e in events if e.get('cat')=='gpu_user_annotation' and e.get('name','').startswith('step[DECODE bs=')),key=lambda x:x['ts'])
    output=[]
    for si,step in enumerate(steps):
        lo=float(step['ts']); hi=lo+float(step['dur'])
        contained=[e for e in kernels if lo<=e['ts'] and e['ts']+e['dur']<=hi]
        activities=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and isinstance(e.get('ts'),(int,float)) and isinstance(e.get('dur'),(int,float)) and e['dur']>=0 and e['ts']<hi and e['ts']+e['dur']>lo]
        records=collections.defaultdict(list); owner_intervals=collections.defaultdict(list)
        for e in activities:
            owner=classify(e.get('name',''),e.get('cat',''),e.get('args',{})); records[owner].append(e)
            owner_intervals[owner].append((max(lo,float(e['ts'])),min(hi,float(e['ts'])+float(e['dur']))))
        # Kernel-only category unions prevent copy activity names from entering owner overlap.
        kernel_unions={o:interval_union((max(lo,float(e['ts'])),min(hi,float(e['ts'])+float(e['dur']))) for e in records[o] if e.get('cat')=='kernel') for o in OWNER_ORDER}
        all_kernel=interval_union((e['ts'],e['ts']+e['dur']) for e in contained)
        all_activity=interval_union((max(lo,float(e['ts'])),min(hi,float(e['ts'])+float(e['dur']))) for e in activities)
        category_union={o:interval_union(owner_intervals[o]) for o in records}
        lengths=lambda iv:sum(b-a for a,b in iv)
        overlaps={}; kinds=[o for o in OWNER_ORDER if kernel_unions[o]]
        for i,a in enumerate(kinds):
            for b in kinds[i+1:]: overlaps[f'{a} <> {b}']=overlap_duration(kernel_unions[a],kernel_unions[b])
        # Exclusivity across all activity categories, swept as active owner set.
        points=[]
        for owner,iv in category_union.items():
            for a,b in iv: points.extend(((a,1,owner),(b,-1,owner)))
        points.sort(key=lambda x:(x[0],x[1])); active=set(); prev=None; exclusive=collections.Counter()
        for t,delta,owner in points:
            if prev is not None and t>prev and len(active)==1: exclusive[next(iter(active))]+=t-prev
            if delta<0: active.discard(owner)
            else: active.add(owner)
            prev=t
        summary={}
        for owner in OWNER_ORDER:
            arr=records[owner]
            if arr: summary[owner]={'calls':len(arr),'duration_sum_us':sum(float(e['dur']) for e in arr),'interval_union_us':lengths(category_union[owner]),'exclusive_us':exclusive[owner],'names':dict(collections.Counter(e.get('name','') for e in arr))}
        span=float(step['dur'])
        output.append({'step_index':si,'decode_batch_annotation':step.get('name'),'step_external_id':step.get('args',{}).get('External id'),'annotation_start_us':lo,'annotation_end_us':hi,'annotation_span_us':span,'kernel_calls':len(contained),'gpu_activity_calls':len(activities),'kernel_duration_sum_us':sum(float(e['dur']) for e in contained),'gpu_activity_duration_sum_us':sum(float(e['dur']) for e in activities),'all_kernel_union_us':lengths(all_kernel),'all_gpu_activity_union_us':lengths(all_activity),'all_kernel_span_us':max((e['ts']+e['dur'] for e in contained),default=lo)-min((e['ts'] for e in contained),default=lo),'all_kernel_coverage_of_annotation':lengths(all_kernel)/span if span else None,'owner_union_overlap_excess_us':sum(lengths(v) for v in category_union.values())-lengths(all_activity),'pairwise_owner_overlap_us':overlaps,'exclusive_owner_us':dict(exclusive),'owners':summary,'unknown_kernel_calls':sum(1 for e in records['UNKNOWN'] if e.get('cat')=='kernel'),'unknown_duration_sum_us':sum(float(e['dur']) for e in records['UNKNOWN'] if e.get('cat')=='kernel')})
    return {'file':str(path),'sha256':hashlib.sha256(open(path,'rb').read()).hexdigest(),'trace_device':doc.get('deviceProperties'),'cuda_runtime':doc.get('cuda_runtime_version'),'cuda_driver':doc.get('cuda_driver_version'),'trace_event_counts':dict(collections.Counter(e.get('cat','<none>') for e in events)),'kernel_events_total':len(kernels),'graph_launch_count':sum(e.get('name')=='cudaGraphLaunch' for e in events),'step_windows':output}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(ROOT)); ap.add_argument('--out',required=True); args=ap.parse_args()
    files=sorted(glob.glob(os.path.join(args.root,'*','*DECODE.trace.json.gz')))
    result={'schema':'g020-offline-trace-analysis-v1','analysis_mode':'offline only','input_root':args.root,'traces':[analyze(p) for p in files]}
    with open(args.out,'w') as f: json.dump(result,f,indent=2)
    print(json.dumps({'traces':len(files),'out':args.out}))
if __name__=='__main__': main()
