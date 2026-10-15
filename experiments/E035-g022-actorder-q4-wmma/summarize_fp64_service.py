#!/usr/bin/env python3
"""Create bounded service comparison summary from frozen exact-identity JSONs."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E035')
FILES={'c57':'c57-exact-service-matched3.json','fp64':'fp64-service-matched3.json'}
EXPECTED={'c57':'c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9','fp64':'35998ff295f0a4fe27417886ff2fd86bc9a1c1030975435e8b34aa42fb26b5f6'}
def med(xs):return statistics.median(xs)
D={k:json.loads((ROOT/f).read_text()) for k,f in FILES.items()}
for k,d in D.items():
 assert d['route']==k and d['expected_sha']==EXPECTED[k] and len(d['results'])==18
 for r in d['results']:
  assert r['same_ids_within_wave'] and r['prompt_tokens'] in (1024,4096) and r['client_concurrency'] in (1,2,4)
rows=[]
for n in (1024,4096):
 for c in (1,2,4):
  cell={}
  for arm,d in D.items():
   rr=[r for r in d['results'] if r['prompt_tokens']==n and r['client_concurrency']==c]
   assert len(rr)==3
   req=[q for r in rr for q in r['requests']]
   cell[arm]={'repeats':3,'actual_client_concurrency':c,'actual_gpu_batch':'see scheduler logs; not FPM-joined to client RIDs','graph_key':'Decode profile graph replay count 4 at M1 decode window; service request graph key not per-RID joined','service_output_tps_median':med([r['aggregate_tps'] for r in rr]),'wave_wall_ms_median':1000*med([r['wave_wall_s'] for r in rr]),'request_ttft_ms_median':1000*med([q['ttft_s'] for q in req]),'request_itl_p50_ms_median':med([q['itl_ms_p50'] for q in req]),'request_itl_p95_ms_median':med([q['itl_ms_p95'] for q in req]),'request_e2e_ms_median':1000*med([q['wall_s'] for q in req]),'token_lengths':sorted({len(q['output_ids']) for q in req}),'token_hashes':sorted({hashlib.sha256(bytes(str(q['output_ids']),'ascii')).hexdigest() for q in req}),'same_ids_within_every_wave':all(r['same_ids_within_wave'] for r in rr)}
  a,b=cell['c57'],cell['fp64'];cell['fp64_output_tps_delta_pct']=100*(b['service_output_tps_median']/a['service_output_tps_median']-1)
  rows.append({'prompt_tokens':n,'client_concurrency':c,**cell})
out={'routes':{'c57':{'binary_sha256':EXPECTED['c57'],'service_artifact':str(ROOT/FILES['c57'])},'fp64':{'binary_sha256':EXPECTED['fp64'],'s1_base_sha256':'fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842','service_artifact':str(ROOT/FILES['fp64'])}},'conditions':{'model_revision':'52036ae497a2c0c253e17f98cc8652f2f7082ae3','prompt_token_counts':[1024,4096],'requested_output_tokens':256,'temperature':0,'sampling_seed':31337,'radix_cache':'disabled; both cold and warm prompt requests report cached_tokens=0','profile_instrumented_service_waves':'separate FPM matrices; not mixed into this uninstrumented throughput summary','paired_order':'serial c57 then FP64 final matched run, three repeats per cell; earlier contaminant artifacts excluded'},'cells':rows,'resource_sample':{'target_uuid':'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189','max_model_vram_used_mib':11354,'min_memavailable_kib':12847292,'temperature_c':45,'cache_residency':'model loaded; Mamba/KV state reset between request via new request; prefix cache disabled'}}
path=ROOT/'fp64-final-service-summary.json';path.write_text(json.dumps(out,indent=2)+'\n');print(path)
