#!/usr/bin/env python3
import glob,json,statistics,os
for p in sorted(glob.glob('/mnt/data/pc01/sglang-v100-migration/experiments/E043/export/*-paired.json')):
 d=json.load(open(p));print(os.path.basename(p),'M',d['M'],'graph',round(d['median_ratio_graph'],6),'eager',round(d['median_ratio_eager'],6))
