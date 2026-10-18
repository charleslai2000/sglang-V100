#!/usr/bin/env python3
"""Exact token sequence comparison of stored service matrices, first mismatch included."""
import json,sys
c=json.load(open(sys.argv[1]));q=json.load(open(sys.argv[2]));idx={(r['prompt_tokens'],r['client_concurrency'],r['repeat']):r for r in q['results']};out=[]
for r in c['results']:
 k=(r['prompt_tokens'],r['client_concurrency'],r['repeat']);s=idx[k];a=r['requests'];b=s['requests']
 if len(a)!=len(b):raise SystemExit(f'request count mismatch {k}')
 for i,(x,y) in enumerate(zip(a,b)):
  xx=x['output_ids'];yy=y['output_ids'];d=next((j for j,(u,v) in enumerate(zip(xx,yy)) if u!=v),None)
  if d is None and len(xx)!=len(yy):d=min(len(xx),len(yy))
  out.append({'prompt_tokens':k[0],'concurrency':k[1],'repeat':k[2],'request':i,'tokens_equal':d is None,'token_count_c57':len(xx),'token_count_t011':len(yy),'first_mismatch_token':d,'c57_token':xx[d] if d is not None and d<len(xx) else None,'t011_token':yy[d] if d is not None and d<len(yy) else None})
print(json.dumps({'requests':len(out),'all_equal':all(x['tokens_equal'] for x in out),'rows':out},indent=2))
