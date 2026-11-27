"""P1: unprofiled hot-prefix extend loop with NVML sampling (GPU busy vs idle)."""
import json, os, subprocess, sys, time, urllib.request, statistics

BASE="http://127.0.0.1:30177"
CASES=json.load(open('/var/tmp/lfm3_ref/input_ids.json')); BASE_IDS=CASES["k4"][:1024]
DELTAS=(5,8,10); REPS=40

def post(path,payload,timeout=900):
    r=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=timeout) as f: body=f.read().decode()
    try: return json.loads(body)
    except Exception: return {"raw":body}

UUID=subprocess.run(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],capture_output=True,text=True).stdout.strip().split("\n")[0]
out={}
post("/flush_cache",{})
post("/generate",{"input_ids":BASE_IDS,"sampling_params":{"max_new_tokens":1,"temperature":0}})
for d in DELTAS:
    p=BASE_IDS+CASES["k4"][1024:1024+d]
    csv_path=f"/var/tmp/p1_nvml_d{d}.csv"
    sampler=subprocess.Popen(["/data/t011/venv/sglang/bin/python","/data/workspace/3rdparty/sglang-V100/benchmark/perf_telemetry/gpu_sampler.py",UUID,csv_path,"240","0.05"],stdout=subprocess.DEVNULL)
    time.sleep(1.5)
    walls=[];cached=None
    t0=time.perf_counter()
    for _ in range(REPS):
        s=time.perf_counter()
        r=post("/generate",{"input_ids":p,"sampling_params":{"max_new_tokens":1,"temperature":0},"return_meta_info":True})
        walls.append((time.perf_counter()-s)*1000)
        cached=r.get("meta_info",{}).get("cached_tokens",cached)
    elapsed=time.perf_counter()-t0
    sampler.terminate(); sampler.wait(timeout=30)
    import csv as _csv
    rows=list(_csv.DictReader(open(csv_path)))
    utils=[float(x["utilization.gpu"]) for x in rows]
    mean_util=sum(utils)/max(1,len(utils))
    busy=mean_util/100.0*elapsed
    out[str(d)]={"delta":d,"reps":REPS,"prompt_tokens":len(p),"cached_tokens":cached,
        "loop_s":round(elapsed,3),"wall_median_ms":round(statistics.median(walls),3),
        "wall_min_ms":round(min(walls),3),"wall_max_ms":round(max(walls),3),
        "gpu_util_median_pct":round(statistics.median(utils),1),
        "mean_util_pct":round(mean_util,1),
        "gpu_busy_per_req_ms":round(busy/REPS*1000,2),
        "gpu_idle_per_req_ms":round(statistics.median(walls)-busy/REPS*1000,2)}
    print(out[str(d)])
json.dump(out,open("/var/tmp/p1_extend_nvml.json","w"),indent=2)
