#!/usr/bin/env python3
"""Map per-layer NVTX ranges to CUPTI kernel correlation IDs for G002-L."""
import collections
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
strings = dict(con.execute("select id, value from StringIds"))
rows = con.execute(
    "select start,end,globalTid,text from NVTX_EVENTS "
    "where text like 'G002L_ATTENTION_LAYER:%' order by start"
).fetchall()
if not rows:
    raise SystemExit("No G002L layer ranges found")
runtime = con.execute(
    "select correlationId,start,end,globalTid,nameId from CUPTI_ACTIVITY_KIND_RUNTIME "
    "order by start"
).fetchall()
kernels = con.execute(
    "select start,end,correlationId,contextId,streamId,demangledName,gridX,gridY,gridZ "
    "from CUPTI_ACTIVITY_KIND_KERNEL order by start"
).fetchall()
for start, end, tid, text in rows:
    layer = text.split(":")
    if len(layer) < 3:
        raise SystemExit(f"Malformed NVTX layer label: {text}")
    layer_id = int(layer[1])
    dim = int(layer[2].removeprefix("D="))
    rts = [r for r in runtime if r[3] == tid and r[1] >= start and r[2] <= end]
    corr = {r[0] for r in rts}
    linked = [k for k in kernels if k[2] in corr]
    by_kernel = collections.defaultdict(list)
    for k in linked:
        by_kernel[strings[k[5]]].append(k)
    summaries = []
    for name, ks in by_kernel.items():
        summaries.append(f"{name} n={len(ks)} sum_ms={sum(k[1]-k[0] for k in ks)/1e6:.6f}")
    total = sum(k[1]-k[0] for k in linked)
    print(f"layer={layer_id} dim={dim} nvtx_ms={(end-start)/1e6:.6f} runtime={len(rts)} corr={len(corr)} kernels={len(linked)} kernel_sum_ms={total/1e6:.6f} streams={sorted(set((k[3],k[4]) for k in linked))}")
    for s in summaries:
        print("  " + s)
