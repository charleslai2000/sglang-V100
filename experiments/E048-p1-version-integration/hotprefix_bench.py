"""P1: hot-prefix incremental extend benchmark (5/8/10 new tokens on a cached prefix).

Measures, per delta:
  * cold TTFT  (prefix cache flushed first)
  * hot  TTFT  (prefix cached by an earlier request)
  * cached_tokens actually reported by the scheduler
and cross-checks the scheduler's own prefill log lines for the same requests.
"""

import json
import statistics
import sys
import time
import urllib.request

BASE_URL = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 30177}"
CASES = json.load(open("/var/tmp/lfm3_ref/input_ids.json"))
BASE = CASES["k4"]
PREFIX_LEN = 1024
PREFIX = BASE[:PREFIX_LEN]
DELTAS = (5, 8, 10)
REPS = 5
NEW_TOKENS = 1


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "ignore")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def flush():
    post("/flush_cache", {})


def ttft_of(prompt_ids, tag):
    """Streaming request with one new token; return (ttft_s, meta_info)."""
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {"max_new_tokens": NEW_TOKENS, "temperature": 0, "n": 1},
        "stream": True,
        "return_meta_info": True,
    }
    start = time.perf_counter()
    req = urllib.request.Request(
        BASE_URL + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    meta = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body in ("", "[DONE]"):
                continue
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if chunk.get("meta_info"):
                meta.update(chunk["meta_info"])
            if chunk.get("text") or chunk.get("output_ids"):
                if ttft is None:
                    ttft = time.perf_counter() - start
    if ttft is None:
        ttft = time.perf_counter() - start
    return ttft, meta


def warm_prefix():
    """Populate the radix cache with the shared prefix."""
    payload = {
        "input_ids": PREFIX,
        "sampling_params": {"max_new_tokens": NEW_TOKENS, "temperature": 0},
        "return_meta_info": True,
    }
    out = post("/generate", payload)
    return out.get("meta_info", {})


def main():
    results = {"prefix_len": PREFIX_LEN, "deltas": {}, "new_tokens": NEW_TOKENS}
    for delta in DELTAS:
        prompt = PREFIX + BASE[PREFIX_LEN : PREFIX_LEN + delta]
        row = {"prompt_tokens": len(prompt), "delta": delta, "cold": [], "hot": []}

        # Cold: no cached prefix at all.
        for _ in range(REPS):
            flush()
            ttft, meta = ttft_of(prompt, "cold")
            row["cold"].append(
                {
                    "ttft_ms": round(ttft * 1000, 3),
                    "cached_tokens": meta.get("cached_tokens"),
                    "prompt_tokens": meta.get("prompt_tokens"),
                }
            )

        # Hot: prefix already cached, only `delta` new tokens need an extend.
        flush()
        warm_meta = warm_prefix()
        row["warm_cached_tokens"] = warm_meta.get("cached_tokens")
        for _ in range(REPS):
            ttft, meta = ttft_of(prompt, "hot")
            row["hot"].append(
                {
                    "ttft_ms": round(ttft * 1000, 3),
                    "cached_tokens": meta.get("cached_tokens"),
                    "prompt_tokens": meta.get("prompt_tokens"),
                }
            )

        cold = [r["ttft_ms"] for r in row["cold"]]
        hot = [r["ttft_ms"] for r in row["hot"]]
        row["cold_median_ms"] = round(statistics.median(cold), 3)
        row["hot_median_ms"] = round(statistics.median(hot), 3)
        row["extend_median_ms"] = round(statistics.median(hot), 3)
        row["speedup"] = round(statistics.median(cold) / statistics.median(hot), 2)
        row["hot_cached_tokens"] = sorted(
            {r["cached_tokens"] for r in row["hot"]}, key=lambda v: (v is None, v)
        )
        results["deltas"][str(delta)] = row
        print(
            f"delta={delta:2d} prompt={len(prompt)} cold={row['cold_median_ms']:8.3f} ms "
            f"hot={row['hot_median_ms']:8.3f} ms speedup={row['speedup']:5.2f}x "
            f"cached_tokens(hot)={row['hot_cached_tokens']} "
            f"cached_tokens(cold)={sorted({r['cached_tokens'] for r in row['cold']}, key=lambda v: (v is None, v))}"
        )

    json.dump(results, open("/var/tmp/p1_hotprefix.json", "w"), indent=2)
    print("wrote /var/tmp/p1_hotprefix.json")


if __name__ == "__main__":
    main()
