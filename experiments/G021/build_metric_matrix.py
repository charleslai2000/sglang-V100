#!/usr/bin/env python3
"""Build the G021 R17 evidence matrix from retained, bounded source artifacts.

Offline only: reads repository JSON/CSV plus external capture outputs; no CUDA,
model loading, or reinterpretation of unsupported metrics.
"""
import csv
import hashlib
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = Path('/mnt/data/pc01/sglang-v100-migration')
OUT = ROOT / 'experiments/G021/artifacts'
UUID = 'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189'
S1_SHA = 'fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842'
MODEL_REV = '52036ae497a2c0c253e17f98cc8652f2f7082ae3'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def metric(value, unit, boundary, samples, source, status='MEASURED', reason=None):
    d = {'value': value, 'unit': unit, 'boundary': boundary, 'samples': samples,
         'source': source, 'status': status}
    if reason:
        d['reason'] = reason
    return d

service_inputs = [
    ('S1', ROOT / 'experiments/G021/artifacts/service-s1-all-cells.json'),
    ('Candidate', ROOT / 'experiments/G021/artifacts/service-candidate-dispatch-verified.json'),
]
rows = []
for route, service_path in service_inputs:
    service_doc = json.loads(service_path.read_text())
    service_rows = service_doc['results']
    for prompt in (1024, 4096):
      for concurrency in (1, 2, 4):
        cell = [x for x in service_rows if x['prompt_tokens'] == prompt and x['concurrency'] == concurrency]
        reqs = [q for x in cell for q in x['requests']]
        waves = [x['aggregate_tps'] for x in cell]
        quant = lambda key, scale=1: sorted(q[key] * scale for q in reqs)
        ttft = quant('first_token_s', 1000)
        e2e = quant('wall_s', 1000)
        itl_p50 = sorted(q['itl_ms_p50'] for q in reqs)
        itl_p95 = sorted(q['itl_ms_p95'] for q in reqs)
        # nearest-rank; explicitly report sample counts and do not call small-n p99.
        nr = lambda xs, p: xs[max(0, int((p * len(xs) + 0.999999999)) - 1)]
        rows.append({
            'run_id': f'{route.lower()}-service-{prompt}-{concurrency}-{len(cell)}waves',
            'target_uuid': UUID, 'source_commit': 'G017 accepted base + service capture artifact',
            'model_revision': MODEL_REV, 'route': route, 'extension_sha256': service_doc['binary_sha256'],
            'workload': {'prompt_tokens': prompt, 'output_tokens': 256, 'submitted_concurrency': concurrency,
                         'repeats': len(cell), 'requests': len(reqs), 'cache': 'radix disabled for S1; Candidate route/config as pinned in source JSON'},

            'actual_batch': {'status': 'PARTIAL', 'reason': 'service JSON lacks per-forward batch; use separately scoped G020 trace / FPM captures'},
            'metrics': {
                'output_tps': metric(statistics.mean(waves), 'output tokens/s', 'client wave aggregate, mean of 3 waves', len(waves), str(service_path)),
                'input_tps': metric(None, 'input tokens/s', 'not timed as a separate service interval', len(waves), str(service_path), 'NOT_MEASURED', 'Harness reports output TPS only.'),
                'rps': metric(None, 'requests/s', 'not exported as a separately bounded request rate', len(waves), str(service_path), 'NOT_MEASURED', 'Wave aggregate TPS cannot be relabelled RPS.'),
                'ttft_ms_p50': metric(nr(ttft,.50), 'ms', 'client request start to first streamed output', len(reqs), str(service_path)),
                'ttft_ms_p95': metric(nr(ttft,.95), 'ms', 'client request start to first streamed output', len(reqs), str(service_path)),
                'ttft_ms_p99': metric(None, 'ms', 'client request start to first streamed output', len(reqs), str(service_path), 'NOT_MEASURED', 'Only 3 c1 / 12 c4 requests per cell; p99 not statistically supported.'),
                'itl_ms_p50': metric(nr(itl_p50,.50), 'ms', 'streamed inter-token interval per request; median of per-request p50', len(reqs), str(service_path)),
                'itl_ms_p95': metric(nr(itl_p95,.95), 'ms', 'streamed inter-token interval; p95 of per-request p95', len(reqs), str(service_path)),
                'e2e_ms_p50': metric(nr(e2e,.50), 'ms', 'client request start to completion', len(reqs), str(service_path)),
                'e2e_ms_p95': metric(nr(e2e,.95), 'ms', 'client request start to completion', len(reqs), str(service_path)),
                'e2e_ms_p99': metric(None, 'ms', 'client request start to completion', len(reqs), str(service_path), 'NOT_MEASURED', 'Insufficient request samples for p99.'),
                'queue_admission_wait': metric(None, 'ms', 'server enqueue/admission timestamp pair', 0, str(service_path), 'NOT_MEASURED', 'Client trace has no server arrival/admission timestamps.'),
                'actual_running_queued': metric(None, 'requests', 'per-request server lifecycle', 0, str(service_path), 'PARTIAL', 'Periodic server logs show queue/running observations but are not aligned to these exact request IDs.'),
                'nvml_gpu_busy': metric(None, '%', 'per-request aligned device window', 0, 'artifacts/nvml-s1-matrix.csv', 'UNAVAILABLE', 'Existing whole-device window spans multiple cells and has no per-request tags; matrix aggregate is documented in report.'),
            },
            'status': 'PARTIAL',
        })

# FPM accepted-path bounded capture: accurately expose event sums vs active union.
sidecar_path = STORE / 'experiments/E026/fpm-request-sidecar.jsonl'
subscriber_path = STORE / 'experiments/E026/fpm-subscriber-delivery.json'
server_log = STORE / 'logs/g021-fpm-delivery-window.log'
request_path = STORE / 'experiments/E026/fpm-production-delivery-requests.json'
for p in (sidecar_path, subscriber_path, server_log, request_path):
    if not p.exists(): raise FileNotFoundError(p)
side = [json.loads(x) for x in sidecar_path.read_text().splitlines()]
forwards = [x for x in side if x.get('kind') != 'shutdown']
user = [x for x in forwards if not any(str(r.get('rid','')).startswith('HEALTH_CHECK_') for r in x.get('requests', []))]
ext = [x for x in user if x['mode'] == 'EXTEND']
dec = [x for x in user if x['mode'] == 'DECODE']
sub = json.loads(subscriber_path.read_text())
wire = sub['rows']
active = [x for x in wire if x.get('prefill_reqs', 0) or x.get('decode_reqs', 0)]
requests = json.loads(request_path.read_text())
assert len(forwards) == 2357 and len(active) == 2357 and len(wire) == 2506
scheduled_prefill = sum(r.get('scheduled_prefill_input_tokens', 0) for x in ext for r in x['requests'])
aggregate_user_prefill = sum(x['scheduled_prefill_tokens'] for x in ext)
health_extend = [x for x in forwards if x['mode'] == 'EXTEND' and any(str(r.get('rid','')).startswith('HEALTH_CHECK_') for r in x['requests'])]
all_extend_owner_sum = scheduled_prefill + sum(r.get('scheduled_prefill_input_tokens',0) for x in health_extend for r in x['requests'])
prefill_event = sum(x['gpu_elapsed_s'] for x in ext)
decode_steps = sum(x['decode_output_tokens'] for x in dec)
decode_event = sum(x['gpu_elapsed_s'] for x in dec)
seq_gaps = sum(b['transport_seq'] != a['transport_seq'] + 1 for a,b in zip(wire,wire[1:]))
shutdown = next(x for x in side if x.get('kind') == 'shutdown')
forwards_doc = {
    'run_id': 'fpm-s1-ready-before-start-20261011', 'target_uuid': UUID,
    'source_commit': '631cfc89c2644d398a4555fcfbcb509919275bef', 'model_revision': MODEL_REV,
    'route': 'S1', 'extension_sha256': S1_SHA,
    'workload': {'prompts': [1024,4096], 'requested_concurrency': [1,4], 'waves': len(requests),
                 'requests': sum(len(x['output_tokens']) for x in requests), 'outputs_each': 96,
                 'observed_cache': 'radix disabled; per-request cached_tokens_before_forward sum 0'},
    'actual_batch_distribution': dict(Counter(str(x['scheduled_batch_size']) for x in dec)),
    'decode_graph_key_distribution': dict(Counter(str(x['graph_key']) for x in dec)),
    'metrics': {
      'scheduled_prefill_tokens': metric(scheduled_prefill, 'tokens', 'sum of per-request extend_input_len snapshots across user EXTEND forwards; excludes health-check', sum(len(x['requests']) for x in ext if not any(str(r.get('rid','')).startswith('HEALTH_CHECK_') for r in x['requests'])), str(sidecar_path)),
      'aggregate_prefill_stats_tokens': metric(aggregate_user_prefill, 'tokens', 'sum of PrefillStats.log_input_tokens across 43 user EXTEND forwards (health-check excluded)', len(ext), str(sidecar_path), 'PARTIAL'),
      'prefill_token_definition_difference': metric(aggregate_user_prefill-scheduled_prefill, 'tokens', 'aggregate PrefillStats sum minus sum(req.extend_input_len) on same 43 user EXTEND forwards', len(ext), str(sidecar_path), 'PARTIAL'),
      'all_extend_request_owner_tokens_including_health': metric(all_extend_owner_sum, 'tokens', 'sum request extend_input_len across all EXTENDs including one health probe', len(ext)+len(health_extend), str(sidecar_path)),
      'health_probe_extend_forwards': metric(len(health_extend), 'forward', 'health endpoint EXTEND excluded from user-workload TPS', len(health_extend), str(sidecar_path)),
      'request_cached_tokens_before_forward_sum': metric(sum(r.get('cached_tokens_before_forward',0) for x in ext for r in x['requests']), 'tokens', 'sum of per-request pre-forward cached token snapshots including health probe', sum(len(x['requests']) for x in ext), str(sidecar_path)),
      'prefill_cuda_event_elapsed_sum': metric(prefill_event, 's', 'sum of one CUDA Event interval per EXTEND Forward; not active union', len(ext), str(sidecar_path)),
      'prefill_event_bound_tokens_per_second': metric(scheduled_prefill/prefill_event, 'tokens/s', 'scheduled Extend tokens divided by accumulated CUDA Event duration; diagnostic, overlapping intervals not unioned', len(ext), str(sidecar_path), 'PARTIAL'),
      'decode_output_token_steps': metric(decode_steps, 'tokens', 'one output-token step per scheduled non-speculative request-forward', len(dec), str(sidecar_path)),
      'decode_cuda_event_elapsed_sum': metric(decode_event, 's', 'sum of CUDA Event forward intervals; not GPU active union', len(dec), str(sidecar_path)),
      'decode_event_bound_steps_per_second': metric(decode_steps/decode_event, 'steps/s', 'token steps divided by accumulated Event intervals; diagnostic, not complete GPU Decode TPS', len(dec), str(sidecar_path), 'PARTIAL'),
      'decode_step_event_p50_ms': metric(statistics.median(x['gpu_elapsed_s']*1000 for x in dec), 'ms', 'single-Forward CUDA Event elapsed', len(dec), str(sidecar_path)),
      'decode_step_event_p95_ms': metric(sorted(x['gpu_elapsed_s']*1000 for x in dec)[int(.95*(len(dec)-1))], 'ms', 'single-Forward CUDA Event elapsed', len(dec), str(sidecar_path)),
      'scheduler_wall_sum_prefill': metric(sum(x['scheduler_wall_s'] for x in ext), 's', 'scheduler wall interval sum, separate from Event time', len(ext), str(sidecar_path)),
      'scheduler_wall_sum_decode': metric(sum(x['scheduler_wall_s'] for x in dec), 's', 'scheduler wall interval sum, separate from Event time', len(dec), str(sidecar_path)),
      'gpu_active_union': metric(None, 's', 'common device timestamp interval union', 0, str(sidecar_path), 'UNSUPPORTED', 'CUDA Event elapsed intervals have no shared absolute CUDA timestamp coordinate; interval sums are not a union.'),
      'queue_admission_and_gap': metric(None, 'ms', 'request arrival/enqueue/admission and worker scheduling gap', 0, str(sidecar_path), 'NOT_MEASURED', 'FPM snapshots are Forward-bound; no request lifecycle event chain.'),
      'cache_reused_tokens': metric(0, 'tokens', 'accepted S1 capture; radix disabled and cached_tokens_before_forward snapshots sum to zero', sum(len(x['requests']) for x in ext), str(sidecar_path), 'OBSERVED'),
    },
    'transport_ledger': {'scheduler_active_forward': len(forwards), 'publisher_sent': shutdown['publisher']['sent'],
       'subscriber_received': len(wire), 'subscriber_active': len(active), 'idle_records': len(wire)-len(active),
       'sequence_first_last': [wire[0]['transport_seq'],wire[-1]['transport_seq']], 'sequence_gaps': seq_gaps,
       'queue_drops': shutdown['queue_dropped'], 'send_drops': shutdown['publisher']['send_dropped'],
       'send_failed': shutdown['publisher']['send_failed'], 'pending_batches': shutdown['pending_batches'],
       'pending_gpu_intervals': shutdown['timer_pending']},
    'qualification_scope': 'PASS for bounded accepted Falcon S1 production capture; no further FPM gates.',
    'status': 'PARTIAL for R17 GPU throughput (event-bound diagnostic only)',
}

matrix = {
  'schema': 'g021-r17-evidence-matrix-v1',
  'generated_utc': datetime.now(timezone.utc).isoformat(),
  'goal_status': 'ACTIVE',
  'target_uuid': UUID,
  'identity': {'model': 'Falcon-H1-7B-Instruct-GPTQ-Int4','model_revision': MODEL_REV,
               'accepted_route': 'S1 default; Candidate remains exact-SHA opt-in', 's1_extension_sha256': S1_SHA,
               'g021_commit': 'bcd5a8cbf50f81e9169a412d30565ade5adbdc81',
               'fpm_commit': '631cfc89c2644d398a4555fcfbcb509919275bef'},
  'service_runs': rows,
  'fpm_capture': forwards_doc,
  'hardware_inventory': {
    'whole_device_nvml': {'status':'MEASURED_WINDOW_AGGREGATE','evidence':'experiments/G021/artifacts/nvml-s1-matrix.csv',
      'summary': {'GPU busy mean/median/p95':'95.25/99/100 %','Memory busy mean/p95':'38.43/43 %',
        'VRAM used':'11478 MiB','SM clock mean/p95':'1513/1530 MHz','memory clock':'877 MHz',
        'power mean/p95/max':'225.85/289.30/307.75 W','temperature max':'72 C'},
      'limitation':'Whole-device windows span six service cells, not per-request; NVML Busy is not SM throughput or HBM bandwidth.'},
    'ncu_actual_falcon_derived_kernel_fixtures': {'status':'MEASURED_KERNEL_SCOPE','evidence':['experiments/G021/artifacts/ncu-s1-gateup.csv','experiments/G021/artifacts/ncu-s1-down.csv'],
      'values': {'S1 gate_up':{'duration_ms':0.290496,'dram_read_MB':39.875,'dram_write_MB':2.342,'effective_DRAM_GB_s':145.3,'tensor_pipe_instructions':0},
        'S1 down':{'duration_ms':0.186112,'dram_read_MB':19.793,'dram_write_MB':1.709,'effective_DRAM_GB_s':115.5,'tensor_pipe_instructions':0}},
      'limitation':'Real Falcon L43 M4 operand fixture under NCU kernel replay, not live Graph. GV100 sm__throughput.avg, dram__throughput.avg, lts__throughput.avg are n/a in these reports; NCU queried metrics are supported but no new full Graph profile exists.'},
    'real_graph_full_model_hardware_counters': {'status':'NOT_COLLECTED','value':None,'reason':'Archived G019 CUDA activity traces do not carry Graph node identity or hardware counters. No directly attributable full-model Graph NCU measurement in repository.'},
  },
  'owner_ledger': {'status':'PARTIAL_TRACE_WINDOW','evidence':'experiments/G020/analysis.json',
    's1_1k_bs1_stable_step_ms':{'GPTQ gate_up':9.91,'GPTQ down':6.40,'Attention':2.46,'Mamba':0.69,'Other GEMM':0.0,'Elementwise_norm':2.19,'Memcpy_memset':0.26,'UNKNOWN':7.69,'kernel_union':29.58,'annotation_span':30.55},
    's1_1k_bs4_stable_step_ms':{'GPTQ gate_up':11.41,'GPTQ down':7.30,'Attention':2.91,'Mamba':1.79,'Other GEMM':7.20,'Elementwise_norm':2.82,'Memcpy_memset':0.27,'UNKNOWN':2.19,'kernel_union':35.90,'annotation_span':36.98},
    'limitation':'Three-step torch.profiler windows with per-step GPU annotations; owner durations are sums, can overlap; graph node IDs absent; UNKNOWN includes unowned GEMV/GEMM. Not uninstrumented service timing.'},
  'cache_capability': {'status':'SOURCE_SUPPORTED_BUT_NOT_RUN',
    'observed_service':'radix/hierarchical cache disabled; cached tokens 0 in measured S1 request set.',
    'source':'SGLang registry selects HiMambaRadixCache when enable_hierarchical_cache and hybrid SSM; Falcon config has all 44 layers with both full attention and Mamba. Host Mamba pool/restore implementation exists in memory_pool_host.py and hybrid_cache/hybrid_pool_assembler.py.',
    'L1_gpu':'Radix prefix cache available in source but disabled in measured service; cold/warm benefit NOT_RUN.',
    'L2_host':'HiCache host KV + Mamba state path present in source; Falcon-H1 runtime compatibility/correctness and measurement NOT_RUN.',
    'L3_disk':'Optional storage backend exists; no backend/path enabled, no restore bytes/time. DISABLED/NOT_RUN.',
    'state_pair':'Source has HiMambaRadixCache and host Mamba conv/temporal state pools, but complete Falcon joint prefix-frontier restore not demonstrated; correctness/effect UNSUPPORTED for this evidence set.'},
  'scheduler_lifecycle': {'status':'PARTIAL', 'fpm_actual_batch_counts':{'decode_bs1':2024,'decode_bs3':288,'decode_graph_key1':2024,'decode_graph_key4':288},
    'queue_wait':'NOT_MEASURED per request; server periodic queue logs are not ID-correlated.',
    'periodic_log_observation':'The FPM capture service log is periodic and shows #queue-req 0 at sampled Prefill/Decode logs, but samples are not exhaustive or joined to HTTP request IDs; cannot rule out short queue waits.',
    '4k_c4_tail':'Existing S1 service p95 TTFT 14021.0 ms (12 request samples across 3 waves); observed client TTFT includes queue. Root cause unresolved because request IDs are not joined to arrival/enqueue/admission/cache/first-token timestamps.',
    'saturation':'Only c1/c4 bounded 256-output S1/Candidate cells exist for common workload; c8+ scan NOT_RUN.'},
  'required_answers': [
    {'question':1,'status':'PARTIAL','answer':'Measured service output TPS is reported by workload; request-aligned Event ratios are diagnostics, not qualified GPU active TPS.'},
    {'question':2,'status':'PARTIAL','answer':'Service S1/Candidate TTFT/ITL/E2E p50/p95 measured at small n; p99 unsupported.'},
    {'question':3,'status':'PARTIAL','answer':'Whole-device NVML busy and memory-controller busy measured; full Graph SM/Tensor/HBM counters unavailable/not collected.'},
    {'question':4,'status':'PARTIAL','answer':'G020 trace-window owner table exists; UNKNOWN and no Graph-node mapping remain.'},
    {'question':5,'status':'NOT_MEASURED','answer':'No request-ID-correlated server Queue/Admission/GPU-gap lifecycle.'},
    {'question':6,'status':'NOT_RUN','answer':'Measured service disabled radix cache and observed zero cached tokens; causal cache-vs-prefill benefit not run.'},
    {'question':7,'status':'NOT_RUN','answer':'No Host/Disk restore bytes/time/benefit workload.'},
    {'question':8,'status':'NOT_RUN','answer':'No safe c8+ saturation scan; only c1/c4 measured.'},
    {'question':9,'status':'UNRESOLVED_COMPARABILITY','answer':'3090 historical workload/timing boundaries not aligned.'},
    {'question':10,'status':'UNRESOLVED','answer':'Partial G020 owner ledger has large UNKNOWN GEMV/GEMM; hardware counters/recoverable exclusive owner budget incomplete.'}
  ],
}
path = OUT / 'r17-evidence-matrix.json'
path.write_text(json.dumps(matrix, indent=2) + '\n')
# Hash all repository evidence and the critical large external captures.
inputs = [ROOT/'experiments/G021/artifacts/service-s1-all-cells.json', ROOT/'experiments/G021/artifacts/service-candidate-dispatch-verified.json',
          ROOT/'experiments/G021/artifacts/nvml-s1-matrix.csv', ROOT/'experiments/G021/artifacts/nvml-candidate-matrix.csv',
          ROOT/'experiments/G021/artifacts/ncu-s1-gateup.csv', ROOT/'experiments/G021/artifacts/ncu-s1-down.csv',
          ROOT/'experiments/G021/artifacts/ncu-candidate-gateup.csv', ROOT/'experiments/G021/artifacts/ncu-candidate-down.csv',
          ROOT/'experiments/G020/analysis.json', ROOT/'experiments/G020/analyze_traces.py',
          ROOT/'experiments/G021/report.md', ROOT/'experiments/G021/artifacts/fpm-source-gpu-event-audit.json',
          sidecar_path, subscriber_path, server_log, request_path]
manifest = {'schema':'g021-r17-artifact-manifest-v1','g021_commit':'bcd5a8cbf50f81e9169a412d30565ade5adbdc81',
 'fpm_commit':'631cfc89c2644d398a4555fcfbcb509919275bef','target_uuid':UUID,'model_revision':MODEL_REV,
 'matrix':'experiments/G021/artifacts/r17-evidence-matrix.json','generator':'experiments/G021/build_metric_matrix.py',
 'artifacts':[{'path':os.path.relpath(p,ROOT) if str(p).startswith(str(ROOT)) else str(p), 'sha256':sha(p), 'bytes':p.stat().st_size} for p in inputs],
 'status':'PARTIAL_G021_ACTIVE; FPM bounded accepted S1 capture PASS; no qualification claims beyond recorded boundaries'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'matrix':str(path),'service_cells':len(rows),'fpm_forward_rows':len(forwards),'manifest_artifacts':len(inputs)},indent=2))
