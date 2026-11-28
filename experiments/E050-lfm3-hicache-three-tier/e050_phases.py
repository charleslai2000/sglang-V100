"""E050 phase tests against a live HiCache server:
  l2        device eviction -> host(L2) restore
  concurrency  8 concurrent requests on one shared prefix
  repeat    5x (flush -> restore) with host-pool reclaim checks
Metrics come from /metrics (upstream HiCache counters), not from guesses.
"""

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request

CASES = json.load(open("/var/tmp/lfm3_ref/input_ids.json"))
K4 = CASES["k4"]

METRICS = [
    "sglang:hicache_backup_bytes_total",
    "sglang:hicache_backup_tokens_total",
    "sglang:load_back_bytes_total",
    "sglang:load_back_tokens_total",
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    "sglang:hicache_host_mamba_used_states",
    "sglang:hicache_host_mamba_total_states",
    "sglang:prefetched_tokens_total",
    "sglang:backuped_tokens_total",
    "sglang:cached_tokens_total",
]


class Server:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def req(self, path, payload=None, timeout=600):
        data = json.dumps(payload).encode() if payload is not None else b"{}"
        r = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(r, timeout=timeout) as f:
                body = f.read().decode()
        except urllib.error.HTTPError as exc:
            return {"http_error": exc.code, "body": exc.read().decode()[:300]}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body[:200]}

    def flush(self, attempts=60):
        last = None
        for _ in range(attempts):
            last = self.req("/flush_cache", {})
            body = json.dumps(last)
            if "flushed" in body.lower() and "not flushed" not in body.lower():
                return last
            time.sleep(1.0)
        raise RuntimeError(f"/flush_cache never succeeded: {last}")

    def generate(self, ids, max_new=8):
        start = time.perf_counter()
        out = self.req(
            "/generate",
            {
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
                "return_meta_info": True,
                "return_cached_tokens_details": True,
            },
        )
        wall = (time.perf_counter() - start) * 1000
        meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
        details = meta.get("cached_tokens_details") or {}
        return {
            "wall_ms": round(wall, 1),
            "cached": meta.get("cached_tokens"),
            "device": details.get("device", 0),
            "host": details.get("host", 0),
            "storage": details.get("storage") or 0,
            "ids": out.get("output_ids") or [],
        }

    def snapshot(self):
        """Sum counters by name; gauges kept as-is (last value wins)."""
        out = {}
        with urllib.request.urlopen(self.base + "/metrics", timeout=30) as f:
            for line in f.read().decode().splitlines():
                if line.startswith("#"):
                    continue
                for name in METRICS:
                    if line.startswith(name):
                        value = float(line.rsplit(" ", 1)[1])
                        out[name] = out.get(name, 0.0) + value
        return out


def diff(after, before):
    return {k: round(after.get(k, 0.0) - before.get(k, 0.0), 3) for k in after}


def flood(srv, count, length=1024):
    for i in range(count):
        ids = [1000 + i] + [((t + 4096 * (i % 12) + 1) % 60000) for t in K4[: length - 1]]
        srv.generate(ids, 1)


def phase_l2(srv, length, flood_count):
    prompt = K4[:length]
    srv.flush()
    warm = srv.generate(prompt)
    before = srv.snapshot()
    flood(srv, flood_count, 1024)
    after_flood = srv.snapshot()
    hit = srv.generate(prompt)
    after = srv.snapshot()
    return {
        "prompt": length,
        "flood_count": flood_count,
        "warm": warm,
        "after_flood": hit,
        "flood_metrics": diff(after_flood, before),
        "hit_metrics": diff(after, after_flood),
    }


def phase_concurrency(srv, length, concurrency=8):
    prompt = K4[:length]
    srv.flush()
    results = [None] * concurrency
    before = srv.snapshot()

    def worker(i):
        ids = prompt + [11 + i]  # one token differs: shared 1024-token prefix
        results[i] = srv.generate(ids, 8)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.perf_counter() - t0) * 1000
    after = srv.snapshot()
    ids_equal = len({tuple(r["ids"]) for r in results if r}) == 1
    return {
        "prompt": length,
        "concurrency": concurrency,
        "wall_ms": round(wall, 1),
        "levels": [[r["device"], r["host"], r["storage"]] for r in results if r],
        "cached": [r["cached"] for r in results if r],
        "ids_equal_across_requests": ids_equal,
        "metrics": diff(after, before),
    }


def phase_repeat(srv, length, rounds=5):
    prompt = K4[:length]
    out = []
    for i in range(rounds):
        srv.flush()
        # Gauges refresh on the next scheduler iteration, so sample after a beat.
        time.sleep(2.0)
        idle = srv.snapshot()
        time.sleep(1.0)
        idle = srv.snapshot()
        before = srv.snapshot()
        rec = srv.generate(prompt)
        after = srv.snapshot()
        rec["round"] = i
        rec["host_used_after_flush"] = round(idle.get("sglang:hicache_host_used_tokens", -1), 1)
        rec["mamba_used_after_flush"] = round(
            idle.get("sglang:hicache_host_mamba_used_states", -1), 1
        )
        rec["metrics"] = diff(after, before)
        out.append(rec)
    return {"prompt": length, "rounds": out}


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 30177
    ours = sys.argv[2] if len(sys.argv) > 2 else "all"
    srv = Server(port)
    out = {}
    if ours in ("all", "l2"):
        out["l2_1024"] = phase_l2(srv, 1024, 200)
        print("l2_1024", out["l2_1024"]["after_flood"], flush=True)
        out["l2_4096"] = phase_l2(srv, 4096, 200)
        print("l2_4096", out["l2_4096"]["after_flood"], flush=True)
    if ours in ("all", "conc"):
        out["concurrency_1024"] = phase_concurrency(srv, 1024, 8)
        print("conc_1024", out["concurrency_1024"]["levels"], flush=True)
        out["concurrency_4096"] = phase_concurrency(srv, 4096, 8)
        print("conc_4096", out["concurrency_4096"]["levels"], flush=True)
    if ours in ("all", "repeat"):
        out["repeat_1024"] = phase_repeat(srv, 1024, 5)
        print(
            "repeat_1024",
            [(r["storage"], r["wall_ms"], r["host_used_after_flush"]) for r in out["repeat_1024"]["rounds"]],
            flush=True,
        )
    json.dump(out, open(f"/var/tmp/e050_phases_{ours}.json", "w"), indent=1)
    print("wrote", f"/var/tmp/e050_phases_{ours}.json")


if __name__ == "__main__":
    main()
