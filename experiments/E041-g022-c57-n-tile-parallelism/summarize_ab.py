import glob,json,statistics,os
for p in sorted(glob.glob('/mnt/data/pc01/sglang-v100-migration/experiments/E041/export/[a-z]*-m[124]-ntile*.json')):
 d=json.load(open(p));rs=d['records'];vals={}
 for metric in ('eager_us','graph_replay_us'):
  cs=[];ns=[]
  for rep in (1,2,3):
   cs.append(statistics.median(next(r for r in rs if r['rep']==rep and r['arm']=='c57')[metric]))
   ns.append(statistics.median(next(r for r in rs if r['rep']==rep and r['arm']==d['candidate'])[metric]))
  vals[metric]={'c57_pair_medians_us':cs,'candidate_pair_medians_us':ns,'candidate_over_c57_median_pair_ratio':statistics.median(n/c for c,n in zip(cs,ns))}
 print(json.dumps({'file':os.path.basename(p),'projection': 'down' if d['fixture'].endswith('down_proj-call0000') else 'gate_up','M':d['M'],'tile':d['candidate'],'results':vals},sort_keys=True))
