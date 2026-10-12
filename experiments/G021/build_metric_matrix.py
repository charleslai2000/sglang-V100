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
  'falcon_ncu_hotspot_measurements': {
    'status':'MEASURED_KERNEL_SCOPE','tool':'NVIDIA Nsight Compute 2025.1.1.0 root','target_uuid':UUID,
    'model_revision':MODEL_REV,'route':'accepted S1','extension_sha256':S1_SHA,
    'benchmark':'bench_one_batch S1 synthetic valid-vocabulary input, bs1, input1024, output64, graph keys [1,2,4], radix/piecewise disabled, static fraction .75; same warmed standalone run has separate Prefill and Decode profiler ranges. Captured 2026-10-12 on V100; NCU PID worker, helper subprocess tracked.',
    'capture':'--target-processes all --device 0 --profile-from-start off --nvtx --nvtx-include R17_MEASURED_FORWARD --graph-profiling node --replay-mode kernel; name-filtered actual Falcon calls. NVTX is start/end around benchmark stage; CUDA graph capture occurs before it. NCU selected the intended main worker (PID is recorded in report); child tokenizer process is tracked but has no selected matching kernels.',
    'scope':'Real full Falcon forward call path and actual kernel inputs, but values are selected kernel/grid records, not full Graph aggregate or whole Forward. Replayed NCU metrics are arithmetic means across 2-4 NCU replay records for identical kernel/grid. Decode captures CUDA Graph replay kernels from one real decode graph forward; no conversion to complete GPU-active union.',
    'metric_notes':{'gpu__time_duration.sum':'per-selected-kernel elapsed; Prefill CSV unit ms, Decode CSV unit us',
      'dram__bytes_read/write.sum':'per selected-kernel aggregate; NCU displays Mbyte',
      'dram__throughput.avg.pct_of_peak_sustained_elapsed':'device counter as % sustained peak; not bytes/time derived across different scope',
      'lts__t_sector_hit_rate.pct':'L2 sector hit ratio; supported and measured',
      'sm__throughput.avg.pct_of_peak_sustained_elapsed':'SM elapsed throughput; distinct from NVML GPU busy',
      'smsp__inst_executed_pipe_tensor.avg':'Tensor-pipe instruction activity; Tensor zero on decode selected kernels, nonzero on Prefill CUTLASS matmuls',
      'FP32 pipe':'GV100 NCU inventory did not expose smsp__inst_executed_pipe_fp32.avg in collected report; reported per-SASS fadd/fmul/ffma instruction counts as available, not relabeled FP32-only due mixed dtype instructions.',
      'NCU values':'Never sum time or throughput percentages across kernel types; rows are not simultaneous additive budgets.'},
    'artifacts':{'raw_prefill_ncu_rep':'/mnt/data/pc01/sglang-v100-migration/profiles/r17-ncu-tmp/s1-prefill-pipe-final.ncu-rep','raw_prefill_sha256':'9cdf1113bd5c6a69bafec9f6e96c516d8782a3f24aeae2a646e25c8507eda753',
      'raw_decode_ncu_rep':'/mnt/data/pc01/sglang-v100-migration/profiles/r17-ncu-tmp/s1-decode-pipe-final.ncu-rep','raw_decode_sha256':'8ce82c0243279464b898b75508b303fa494e99f949f0e2d609b9b98b8a3c3801',
      'normalized_csv':'/mnt/data/pc01/sglang-v100-migration/profiles/r17-ncu-tmp/s1-hotspots-normalized.csv'},
    'key_rows':{
      'prefill_cutlass_gate_up_projection':{'grid':'64,7,1','ncu_records':4,'duration_ms_mean':0.5432,'sm_throughput_pct_mean':73.8162,'tensor_inst_avg':260505.6,'fp16_inst_avg':339.2,'sass_ffma_inst_avg':21708.8,'dram_read_MiB_mean':74.3795,'dram_write_MiB_mean':14.3413,'dram_throughput_pct_peak_mean':18.2057,'l2_bytes_GiB_mean':0.6720,'l2_throughput_pct_mean':48.4630,'l2_sector_hit_pct_mean':88.9841,'active_warps_pct_mean':11.6623,'issue_active_pct_mean':52.1857,'long_scoreboard_pct_mean':3.2872},
      'prefill_mamba_chunk_scan':{'grid':'128,4,24','ncu_records':4,'duration_ms_mean':0.966152,'sm_throughput_pct_mean':29.9713,'tensor_inst_avg':0,'fp16_inst_avg':153.6,'sass_fadd_inst_avg':83558.4,'sass_fmul_inst_avg':312925.225,'sass_ffma_inst_avg':3863347.2,'dram_read_MiB_mean':27.0093,'dram_write_MiB_mean':12.8450,'dram_throughput_pct_peak_mean':4.6031,'l2_sector_hit_pct_mean':95.6287,'active_warps_pct_mean':55.0753,'issue_active_pct_mean':30.4237,'long_scoreboard_pct_mean':11.8747},
      'prefill_cutlass_attention_projection':{'grid':'64,24,1','ncu_records':4,'duration_ms_mean':1.672352,'sm_throughput_pct_mean':86.5988,'tensor_inst_avg':471859.2,'fp16_inst_avg':2784,'sass_ffma_inst_avg':0,'dram_read_MiB_mean':252.1127,'dram_write_MiB_mean':49.6076,'dram_throughput_pct_peak_mean':20.1006,'l2_bytes_GiB_mean':2.3545,'l2_throughput_pct_mean':55.0003,'l2_sector_hit_pct_mean':89.7800,'active_warps_pct_mean':12.4079,'issue_active_pct_mean':33.8392,'long_scoreboard_pct_mean':1.8616},
      'decode_gptq_gate_up':{'grid':'192,1,24','ncu_records':2,'duration_us_mean':254.304,'sm_throughput_pct_mean':42.9429,'tensor_inst_avg':0,'fp16_inst_avg':16712.1969,'sass_fp32_fadd_fmul_ffma_inst_avg':0,'dram_read_MiB_mean':39.4014,'dram_write_MiB_mean':1.8511,'dram_throughput_pct_peak_mean':18.3652,'l2_bytes_MiB_mean':184.5979,'l2_throughput_pct_mean':28.3962,'l2_sector_hit_pct_mean':78.3424,'active_warps_pct_mean':53.2410,'issue_active_pct_mean':44.4405,'long_scoreboard_pct_mean':61.5722},
      'decode_gptq_down':{'grid':'24,1,96','ncu_records':2,'duration_us_mean':166.656,'sm_throughput_pct_mean':33.5342,'tensor_inst_avg':0,'fp16_inst_avg':8494.3125,'sass_fp32_fadd_fmul_ffma_inst_avg':0,'dram_read_MiB_mean':19.7007,'dram_write_MiB_mean':1.7109,'dram_throughput_pct_peak_mean':14.6573,'l2_bytes_MiB_mean':112.2415,'l2_throughput_pct_mean':26.4049,'l2_sector_hit_pct_mean':82.0710,'active_warps_pct_mean':52.6812,'issue_active_pct_mean':36.6483,'long_scoreboard_pct_mean':67.7226},
      'decode_vocab_gemv_grid835':{'grid':'835,1,1','ncu_records':2,'duration_us_mean':62.08,'sm_throughput_pct_mean':28.9945,'tensor_inst_avg':0,'fp16_inst_avg':2515.4375,'sass_fadd_inst_avg':2317.125,'sass_fmul_inst_avg':8016,'sass_ffma_inst_avg':64128,'dram_read_MiB_mean':41.0605,'dram_write_MiB_mean':1.5334,'dram_throughput_pct_peak_mean':83.6718,'l2_sector_hit_pct_mean':51.1457,'active_warps_pct_mean':42.1014,'issue_active_pct_mean':27.9332,'long_scoreboard_pct_mean':45.6263},
      'decode_mamba_selective_scan':{'grid':'32,1,24','ncu_records':2,'duration_us_mean':13.824,'sm_throughput_pct_mean':34.0310,'tensor_inst_avg':0,'fp16_inst_avg':38.4,'sass_fadd_inst_avg':5529.6,'sass_fmul_inst_avg':6144,'sass_ffma_inst_avg':11673.6,'dram_read_MiB_mean':3.1666,'dram_write_MiB_mean':1.4614,'dram_throughput_pct_peak_mean':43.1014,'l2_sector_hit_pct_mean':51.8958,'active_warps_pct_mean':62.7737,'issue_active_pct_mean':40.7482,'long_scoreboard_pct_mean':32.0096}},
    'limitations':'Prefill first 1024-token iteration of actual EXTEND call; reports include selected attention and Mamba kernels. Decode NCU launch window profile_start_step=32, but Graph kernel nodes are replayed; call path is actual Falcon decode, and all selected grids/names occur in the no-NCU decode trace. NCU replay heavily perturbs benchmark timing (e.g. instrumented throughput ~9-11 tok/s); never use instrumented TPS as service TPS. Captures did not measure whole-Graph aggregate, nonselected kernel families, or device-active union.'},
  'standalone_speed_diagnostic': {
    'run_id':'r17-s1-valid-single-20261012','target_uuid':UUID,'route':'S1','extension_sha256':S1_SHA,
    'source_tree':'clean G017 worktree plus out-of-tree benchmark shim; no tracked source changes',
    'runtime_extensions':{'sm70_moe_sha256':'c64273b785dfc7854392170161f5e6d311be77a3cd5861b12a44144918beb172','sm70_dense_sha256':'b94235dac9713d28f8eee707b391f73bdec15b69317dc2e1fabd49329f763d09'},
    'workload':{'input_tokens':1024,'synthetic_vocab_valid_ids':True,'output_tokens_requested':64,'submitted_concurrency':1,'graph_keys':[1,2,4],'radix_cache':False,'piecewise_graph':False,'mem_fraction_static':0.75},
    'metrics':{'prefill_input_tps':{'value':3595.75,'unit':'input tokens/s','boundary':'1024 synthetic inputs / 0.28478 s synchronized standalone ModelRunner prefill','samples':1,'status':'DIAGNOSTIC'},
      'decode_output_tps_median':{'value':32.48,'unit':'output tokens/s','boundary':'median single-request decode step; 64 output tokens requested','samples':63,'status':'DIAGNOSTIC'},
      'prefill_profiled_cuda_total_ms':{'value':281.444,'unit':'ms','boundary':'single EXTEND bs1 toks1024 PyTorch profiler CUDA total','samples':1,'status':'DIAGNOSTIC'},
      'decode_profiled_cuda_self_sum_ms':{'value':246.441,'unit':'ms','boundary':'8 Graph decode steps Self CUDA sum, not active union','samples':8,'status':'DIAGNOSTIC'}},
    'artifacts':['/mnt/data/pc01/sglang-v100-migration/experiments/E026/r17-s1-valid.jsonl','/mnt/data/pc01/sglang-v100-migration/profiles/r17-s1-valid_batch1_input1024_output64_prefill.trace.json.gz','/mnt/data/pc01/sglang-v100-migration/profiles/r17-s1-valid_batch1_input1024_output64_decode.trace.json.gz'],
    'limitations':'One synthetic c1 standalone run, different request/cache/timing boundary from service matrix; not statistical service/GPU-active TPS. NCU selected real call-path kernels are recorded separately and are not whole Graph.'},
  'prior_full_graph_ncu_attempt':{'status':'ATTEMPT_REJECTED_NOT_SERVING_GRAPH','tool':'NVIDIA Nsight Compute 2025.1.1.0 root','target_uuid':UUID,
    'config':'--graph-profiling graph --replay-mode kernel --profile-from-start off --launch-count 1; Torch CUDA Profiler range around Prefill',
    'result':'NCU report contains only one prefill initialization unrolled_elementwise_kernel; 81 replay passes; no serving Forward Graph row, no counters promoted.',
    'report':'/mnt/data/pc01/sglang-v100-migration/profiles/r17-ncu-tmp/s1-prefill-1k-c1-range.ncu-rep',
    'inventory':'Root metric query lists GV100 SM throughput, Tensor instructions, DRAM bytes, L2 bytes and DRAM/L2 percent-of-peak metrics; inventory is not measurement.'},
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
    'real_falcon_kernel_hardware_counters': {'status':'MEASURED_KERNEL_SCOPE','evidence':'falcon_ncu_hotspot_measurements','limitation':'selected hot kernel records only; no full Graph aggregate or GPU active union'},
    'real_graph_full_model_hardware_counters': {'status':'NOT_COLLECTED','value':None,'reason':'Archived G019 CUDA activity traces do not carry Graph node identity or hardware counters. No directly attributable full-model Graph NCU measurement in repository.'},
  },
  'owner_ledger': {'status':'PARTIAL_TRACE_WINDOW','evidence':'experiments/G020/analysis.json',
    's1_1k_bs1_stable_step_ms':{'GPTQ gate_up':9.91,'GPTQ down':6.40,'Attention':2.46,'Mamba':0.69,'Other GEMM':0.0,'Elementwise_norm':2.19,'Memcpy_memset':0.26,'UNKNOWN':7.69,'kernel_union':29.58,'annotation_span':30.55},
    's1_1k_bs4_stable_step_ms':{'GPTQ gate_up':11.41,'GPTQ down':7.30,'Attention':2.91,'Mamba':1.79,'Other GEMM':7.20,'Elementwise_norm':2.82,'Memcpy_memset':0.27,'UNKNOWN':2.19,'kernel_union':35.90,'annotation_span':36.98},
    'limitation':'Three-step torch.profiler windows with per-step GPU annotations; owner durations are sums, can overlap; graph node IDs absent; UNKNOWN includes unowned GEMV/GEMM. Not uninstrumented service timing.'},
  'cache_capability': {'status':'SOURCE_SUPPORTED_BUT_NOT_RUN',
    'observed_service':'radix/hierarchical cache disabled; cached tokens 0 in measured S1 request set.',
    'source':'Accepted G017 tree source audit: mem_cache/registry.py selects HiMambaRadixCache for hierarchical cache + hybrid SSM; Falcon config has 44 full-attention/Mamba layers. hi_mamba_radix_cache.py builds distinct PoolName.MAMBA backup/restore transfers; hicache_storage.py documents v2 usable-prefix intersection across KV and auxiliary pools; host state pools are assembled in hybrid_cache/hybrid_pool_assembler.py.',
    'L1_gpu':'GPU Radix prefix cache source-supported; disabled in measured service and cached-token snapshots sum to zero. Cold/warm benefit NOT_RUN.',
    'L2_host':'Host KV and Mamba state backup/restore source-supported. Valid full Falcon prefix-frontier restore and runtime correctness were not executed; empirical benefit/cost NOT_RUN (not UNSUPPORTED).',
    'L3_disk':'Optional storage backend and HiMamba archive/prefetch hooks exist; no backend/path enabled. DISABLED/NOT_RUN; no restore bytes/time.',
    'state_pair':'Source exposes paired KV + Mamba state handling and prefix-hit intersection; this is capability evidence only. Full Falcon restore correctness and reuse are NOT_RUN.'},
   'remaining_metric_dispositions':{
     'full_graph_hardware_aggregate':'NOT_RUN; earlier NCU graph-profiling attempt captured initialization only and was rejected. Selected actual Falcon kernel NCU is KERNEL_SCOPE.',
     'gpu_active_union':'UNSUPPORTED by retained CUDA Event data: event durations lack shared absolute CUDA timestamps; do not sum intervals as union.',
     'pcie_h2d_d2h_bandwidth':'NOT_RUN; no valid per-workload directional PCIe byte/time record retained.',
     'cuda_launch_overhead':'NOT_RUN; no isolated/attributable launch-overhead measurement retained.',
     'safe_concurrency_above_c4':'NOT_RUN; accepted S1 max_num_reqs=4 and no validated higher running-request pool headroom/config was evidenced. No c8+ point claimed safe.',
     'request_p99':'NOT_RUN / insufficient sample size: service matrix has 3 requests at c1 and 12 at c4 per cell; no qualified p99.',
     'rtx_3090_comparison':'UNRESOLVED_COMPARABILITY; historical E021 path is external and raw workload/config dataset is unavailable in the current checkout/capture store.'},

  'scheduler_lifecycle': {'status':'PARTIAL', 'fpm_actual_batch_counts':{'decode_bs1':2024,'decode_bs3':288,'decode_graph_key1':2024,'decode_graph_key4':288},
    'queue_wait':'PARTIAL measured in one additional S1 service capture via existing --enable-request-time-stats-logging. Ten request IDs join client TTFT to scheduler queue_duration and forward_duration; one 4K/c4 wave shows TTFT tail tracking wait-queue residence. Admission subphases, GPU first-token duration, scheduling gap, cache wait and preemption remain NOT_RUN; this is not an uninstrumented quantile authority.',
    'measured_service_speed_diagnostic':{'context':'4K/c4','client_wave_tps':29.6806,'requests':4,'boundary':'single 4-request same-wave 256-output client aggregate', 'status':'DIAGNOSTIC_NOT_QUANTILE'},
    'lifecycle_capture':{'target_uuid':UUID,'route':'accepted S1','workload':'single 4K/c4, 4K/c1, 1K/c4, 1K/c1 wave, 256 greedy tokens/request, radix disabled','request_ids_joined':10,'client_artifact':'/mnt/data/pc01/sglang-v100-migration/experiments/E026/r17-s1-lifecycle-client.jsonl','client_sha256':'f68a885aae400052afebd82e1b293272a96eacd682e5659a69f14a22835a9115','server_log':'/mnt/data/pc01/sglang-v100-migration/logs/r17-lifecycle-s1-server.log','server_log_sha256':'168bd605dee93cd9cbfe823f6672ac653527a0574faf1c710c2cfdbb24b9cd8d','request_time_log':'/mnt/data/pc01/sglang-v100-migration/logs/r17-lifecycle-reqtime.log','request_time_log_sha256':'5472b51b6b3cd9e13a7c79b5f38e88cb4ade4959a69ca3d153e0db894d80e078','analysis_script':'experiments/G021/analyze_lifecycle_capture.py','boundary':'Scheduler wait_queue_entry_time to forward_entry_time and forward_entry_time to completion_time from ReqTimeStats. Not GPU duration. Client request-to-first-streamed output remains client TTFT. 4K/c4 queue values track the observed TTFT tail for this bounded wave.'},
    'periodic_log_observation':'The FPM capture service log is periodic and shows #queue-req 0 at sampled Prefill/Decode logs, but samples are not exhaustive or joined to HTTP request IDs; cannot rule out short queue waits.',
    '4k_c4_tail':{'baseline':'Existing S1 service p95 TTFT 14021.0 ms (12 request samples across 3 waves); observed TTFT includes queue.','follow_up_capture':'/mnt/data/pc01/sglang-v100-migration/experiments/E026/r17-s1-lifecycle-client.jsonl and /mnt/data/pc01/sglang-v100-migration/logs/r17-lifecycle-reqtime.log; 10/10 request IDs joined to ReqTimeStats. In a new 4K/c4 wave, queue durations were 0.20, 8367.65, 16991.85, 25620.53 ms; corresponding client TTFT 1191, 9815, 18452, 27070 ms. 4K/c1 matched queue 0.24ms and TTFT 1198ms. Cache disabled/0 cached tokens. This directly shows first-token tail tracks SGLang wait-queue residence in this one capture; forward_duration is request lifecycle span, not isolated GPU active time.', 'limits':'one 4K/c4 and one comparison wave; not a new quantile authority and not enough to explain all historical waves.'},
    'saturation':'Only c1/c2/c4 bounded 256-output S1/Candidate cells exist for common workload; c8+ scan NOT_RUN. Higher points were not treated as safe because no validated live KV/Mamba pool headroom or running-request limit above accepted max_num_reqs=4 was recorded.'},
  'required_answers': [
    {'question':1,'status':'PARTIAL','answer':'Measured service output TPS is reported by workload; request-aligned Event ratios are diagnostics, not qualified GPU active TPS.'},
    {'question':2,'status':'PARTIAL','answer':'Service S1/Candidate TTFT/ITL/E2E p50/p95 measured at small n; p99 unsupported.'},
    {'question':3,'status':'PARTIAL','answer':'Whole-device NVML busy measured and actual Falcon Prefill/Decode selected-kernel NCU counters include SM/Tensor/FP16/SASS/DRAM/L2/efficiency metrics; full-Graph aggregate and complete active time remain unavailable.'},
    {'question':4,'status':'PARTIAL','answer':'G020 trace-window owner table exists; UNKNOWN and no Graph-node mapping remain.'},
    {'question':5,'status':'PARTIAL','answer':'10/10 request-ID join and one 4K/c4 queue-residence capture; admission subphases, GPU-time separation, scheduling gaps/cache wait remain unmeasured.'},
    {'question':6,'status':'NOT_RUN','answer':'Measured service disabled radix cache and observed zero cached tokens; causal cache-vs-prefill benefit not run.'},
    {'question':7,'status':'NOT_RUN','answer':'No Host/Disk restore bytes/time/benefit workload.'},
    {'question':8,'status':'NOT_RUN','answer':'No safe c8+ saturation scan; only c1/c4 measured.'},
    {'question':9,'status':'UNRESOLVED_COMPARABILITY','answer':'No raw RTX 3090 workload/config dataset found in current checkout or searched capture store; historical E021 result is external to this environment. Workload, actual batch, cache and timing boundaries cannot be reconciled, so no direct ratio is calculated.'},
    {'question':10,'status':'UNRESOLVED','answer':'Selected actual Falcon hot-kernel NCU and G020 attribution do not form an exclusive uninstrumented whole-forward owner budget. No optimization owner or recoverable-time estimate is defensible; the single 4K/c4 queue-tail finding is bounded evidence, not a general optimization budget.'}
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
          ROOT/'experiments/G021/report.md', ROOT/'experiments/G021/metric-differential-notes.md',
          ROOT/'experiments/G021/parse_ncu_hotspots.py',
          ROOT/'experiments/G021/artifacts/fpm-source-gpu-event-audit.json',
          sidecar_path, subscriber_path, server_log, request_path,
          STORE/'experiments/E026/r17-s1-valid.jsonl',
          STORE/'profiles/r17-s1-valid_batch1_input1024_output64_prefill.trace.json.gz',
          STORE/'profiles/r17-s1-valid_batch1_input1024_output64_decode.trace.json.gz',
          STORE/'profiles/r17-ncu-tmp/s1-prefill-1k-c1-range.ncu-rep',
          STORE/'profiles/r17-ncu-tmp/s1-prefill-pipe-final.ncu-rep',
          STORE/'profiles/r17-ncu-tmp/s1-decode-pipe-final.ncu-rep',
          STORE/'profiles/r17-ncu-tmp/s1-hotspots-normalized.csv',
          STORE/'experiments/E026/r17-s1-lifecycle-client.jsonl',
          STORE/'logs/r17-lifecycle-s1-server.log', STORE/'logs/r17-lifecycle-reqtime.log',
          ROOT/'experiments/G021/analyze_lifecycle_capture.py']
manifest = {'schema':'g021-r17-artifact-manifest-v1','g021_commit':'f21156e0a73e515206817f216f6c5c9a6636cc87',
 'fpm_commit':'631cfc89c2644d398a4555fcfbcb509919275bef','target_uuid':UUID,'model_revision':MODEL_REV,
 'matrix':'experiments/G021/artifacts/r17-evidence-matrix.json','generator':'experiments/G021/build_metric_matrix.py',
 'artifacts':[{'path':os.path.relpath(p,ROOT) if str(p).startswith(str(ROOT)) else str(p), 'sha256':sha(p), 'bytes':p.stat().st_size} for p in inputs],
 'status':'PARTIAL_G021_ACTIVE; FPM bounded accepted S1 capture PASS; no qualification claims beyond recorded boundaries'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'matrix':str(path),'service_cells':len(rows),'fpm_forward_rows':len(forwards),'manifest_artifacts':len(inputs)},indent=2))
