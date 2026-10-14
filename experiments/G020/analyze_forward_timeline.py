#!/usr/bin/env python3
"""Offline CPU/GPU interval analysis for bounded SGLang Kineto Chrome traces.

Uses the per-forward G022_CLOCK_SYNC_START/END CPU record_function spans to map
host monotonic_ns onto Kineto's CPU trace clock. Kineto's CUDA activities share
the trace timestamp domain. No GPU use or FPM contract changes.
"""
import argparse, collections, gzip, hashlib, json, math
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
OWNER_ORDER = ["GPTQ gate_up", "GPTQ down", "Other GEMM", "vocab GEMV", "Attention", "Mamba", "Elementwise / norm", "Memcpy / memset", "UNKNOWN"]

def union(intervals):
    xs=sorted((float(a),float(b)) for a,b in intervals if a is not None and b is not None and b>a)
    out=[]
    for a,b in xs:
        if out and a<=out[-1][1]: out[-1]=(out[-1][0],max(out[-1][1],b))
        else: out.append((a,b))
    return out

def total(iv): return sum(b-a for a,b in iv)

def classify(e):
    name=e.get('name',''); cat=e.get('cat',''); args=e.get('args') or {}; grid=args.get('grid') or []
    if cat in ('gpu_memcpy','gpu_memset') or name.startswith(('memcpy','memset')): return 'Memcpy / memset'
    if 'sglang::gptq::gemm_half_q_half_alt_4bit_kernel' in name or 'packed_fp32_partial' in name or 'packed_fp32_reduce' in name:
        if grid and grid[0] in (192,96,384): return 'GPTQ gate_up'
        if grid and grid[0] in (24,12,48): return 'GPTQ down'
    if 'gemmSN_TN_kernel<float' in name: return 'vocab GEMV'
    if 'cutlass::Kernel2<' in name or 'cublasLt::splitKreduce_kernel<' in name: return 'Other GEMM'
    if name in ('_fwd_grouped_kernel_stage1','_fwd_kernel_stage2') or any(x in name for x in ('fused_rope_kernel<','store_kvcache<')): return 'Attention'
    if name=='_selective_scan_update_kernel' or 'causal_conv1d_update_kernel<' in name: return 'Mamba'
    if any(x in name for x in ('elementwise_kernel<','vectorized_elementwise_kernel<','unrolled_elementwise_kernel<','FusedAddRMSNormKernel<','RMSNormKernel<','act_and_mul_kernel<')): return 'Elementwise / norm'
    return 'UNKNOWN'

def read_sidecar(path):
    records=collections.defaultdict(list)
    if not path: return records
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r=json.loads(line); records[int(r['forward_iter'])].append(r)
    return records

def analyze(trace_path, sidecar_path=None):
    with gzip.open(trace_path,'rt') as f: doc=json.load(f)
    side=read_sidecar(sidecar_path) if sidecar_path else {}
    ev=doc.get('traceEvents',[]); idx=collections.defaultdict(list)
    for e in ev:
        if e.get('ph')=='X': idx[e.get('cat')].append(e)
    # Fit affine mono_ns -> Kineto_us from profiler timestamp sync annotations.
    sync=[]
    for e in idx['user_annotation']:
        name=e.get('name','')
        if not name.startswith('G022_CLOCK_SYNC_'): continue
        try: fi=int(name.split(' forward_iter=',1)[1])
        except Exception: continue
        is_start='_START ' in name
        item=next((r for r in side.get(fi,[]) if r.get('record_type')=='forward_snapshot'),None)
        if not item: continue
        ns0=item.get('clock_sync_monotonic_start_ns') if is_start else item.get('completion_clock_sync_start_ns')
        ns1=item.get('clock_sync_monotonic_end_ns') if is_start else item.get('completion_clock_sync_end_ns')
        if ns0 is not None and ns1 is not None:
            sync.append(((ns0+ns1)/2e3,e['ts']+e['dur']/2))
    # Every CPU profiler marker is matched to its sidecar monotonic bracket.
    if not sync:
        for e in idx['user_annotation']:
            name=e.get('name','')
            if not name.startswith('G022_CLOCK_SYNC_'): continue
            try: fi=int(name.split(' forward_iter=',1)[1])
            except Exception: continue
            item=next((r for r in side.get(fi,[]) if r.get('record_type')=='forward_snapshot'),None)
            if not item: continue
            is_start='_START ' in name
            ns0=item.get('clock_sync_monotonic_start_ns') if is_start else item.get('completion_clock_sync_start_ns')
            ns1=item.get('clock_sync_monotonic_end_ns') if is_start else item.get('completion_clock_sync_end_ns')
            if ns0 is not None and ns1 is not None: sync.append(((ns0+ns1)/2e3,e['ts']+e['dur']/2))
    if len(sync)<2: raise RuntimeError(f'{trace_path}: need >=2 sidecar-correlated clock markers; got {len(sync)}')
    # Unit slope; fit a robust offset to avoid poor conditioning at large epoch times.
    slope=1.0
    offsets=sorted(y-x for x,y in sync)
    intercept=offsets[len(offsets)//2]
    def mono_to_trace_ns(ns): return slope*(ns/1000.0)+intercept
    anns=sorted([e for e in idx['gpu_user_annotation'] if e.get('name','').startswith('step[')],key=lambda e:e['ts'])
    all_gpu=[e for cat in GPU_CATS for e in idx[cat]]
    rows=[]
    for i,a in enumerate(anns):
        lo=float(a['ts']); hi=lo+float(a['dur']); clipped=[]
        for e in all_gpu:
            s=float(e['ts']); t=s+float(e['dur'])
            if s<hi and t>lo: clipped.append((max(lo,s),min(hi,t),e))
        merged=union((s,t) for s,t,e in clipped)
        gaps=[]; cur=lo
        for s,t in merged:
            if s>cur: gaps.append((cur,s))
            cur=max(cur,t)
        if cur<hi: gaps.append((cur,hi))
        owners=collections.defaultdict(list)
        for s,t,e in clipped: owners[classify(e)].append((s,t))
        sums={o:sum(float(e['dur']) for s,t,e in clipped if classify(e)==o and e.get('cat')=='kernel') for o in OWNER_ORDER}
        kernel_env=union((s,t) for s,t,e in clipped if e.get('cat')=='kernel')
        kernel_internal_gaps=[(b-a) for (a,b),(c,d) in zip(kernel_env,kernel_env[1:])]
        # Synchronized scheduler boundaries are converted only with marker-fit clock.
        try: fi=int(a['name'].split('iter=',1)[1].split()[0])
        except Exception: fi=None
        side_records=side.get(fi,[]) if sidecar_path and fi is not None else []
        snap=next((r for r in side_records if r.get('record_type')=='forward_snapshot'),{})
        timer=next((r for r in side_records if r.get('record_type')=='gpu_event_completion'),{})
        done=next((r for r in side_records if r.get('record_type')=='scheduler_completion'),{})
        cpu_start=mono_to_trace_ns(snap['scheduler_start_monotonic_ns']) if snap.get('scheduler_start_monotonic_ns') else None
        cpu_batch_result=mono_to_trace_ns(done['batch_result_start_monotonic_ns']) if done.get('batch_result_start_monotonic_ns') else None
        cpu_end=mono_to_trace_ns(done['scheduler_end_monotonic_ns']) if done.get('scheduler_end_monotonic_ns') else None
        between= (lo-rows[-1]['gpu_envelope_end_trace_ns']) if rows and lo>=rows[-1]['gpu_envelope_end_trace_ns'] else None
        rows.append({'forward_iter':fi,'annotation':a['name'],'external_id':(a.get('args') or {}).get('External id'),'trace_start_ns':lo,'trace_end_ns':hi,'forward_elapsed_annotation_us':hi-lo,'activity_union_us':total(merged),'kernel_duration_sum_us':sum(float(e['dur']) for s,t,e in clipped if e.get('cat')=='kernel'),'memcpy_duration_sum_us':sum(float(e['dur']) for s,t,e in clipped if e.get('cat') in ('gpu_memcpy','gpu_memset')),'intra_forward_gpu_gaps_us':[b-a for a,b in gaps],'intra_forward_gpu_idle_total_us':sum(b-a for a,b in gaps),'inter_kernel_union_idle_gaps_us':kernel_internal_gaps,'kernel_union_internal_idle_us':sum(kernel_internal_gaps),'activity_envelope_start_trace_ns':merged[0][0] if merged else None,'gpu_envelope_end_trace_ns':merged[-1][1] if merged else None,'inter_forward_gap_us':between,'inter_forward_envelope_gap_us':(merged[0][0]-rows[-1]['gpu_envelope_end_trace_ns']) if rows and merged and merged[0][0]>=rows[-1]['gpu_envelope_end_trace_ns'] else None,'owners_kernel_sum_us':sums,'actual_batch':snap.get('actual_batch_size'),'actual_tokens':snap.get('actual_input_token_count'),'scheduled_prefill_tokens':snap.get('scheduled_prefill_tokens'),'request_ids':[r.get('rid') for r in snap.get('requests',[])],'graph_key':timer.get('graph_key'),'graph_bs':timer.get('graph_bs'),'cuda_event_elapsed_s':timer.get('cuda_event_elapsed_s'),'scheduler_start_trace_ns':cpu_start,'batch_result_start_trace_ns':cpu_batch_result,'scheduler_end_trace_ns':cpu_end,'prefill_eager_or_decode_graph':'Decode Graph if graph_key else Prefill EXTEND/Eager' if fi else None})
    return {'trace':str(trace_path),'sha256':hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),'trace_device':doc.get('deviceProperties'),'trace_clock_alignment':{'method':'Kineto CPU record_function sync marker paired with monotonic_ns bracket; unit-slope robust median offset','marker_count':len(sync),'slope':slope,'intercept_trace_us':intercept,'residual_us':math.sqrt(sum((y-(slope*x+intercept))**2 for x,y in sync)/len(sync))},'event_counts':dict(collections.Counter(e.get('cat','') for e in ev)),'forward_rows':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--sidecar');p.add_argument('--out',required=True);a=p.parse_args()
    result=analyze(a.trace,a.sidecar);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({'forwards':len(result['forward_rows']),'clock':result['trace_clock_alignment'],'out':a.out}))
if __name__=='__main__':main()
