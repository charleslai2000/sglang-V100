"""E051 verification: HiCache restore correctness, liveness and capacity behaviour.

Scenarios
  matrix    1K/4K/8K prefixes x {5,8,10} appends: ids + per-token logprobs, for the
            HiCache paths and for a no-HiCache baseline (pure recompute).
  missing   delete a slice of the L3 files under wait_complete: a partial chain
            must not wedge admission (E050 defect) and the ids must still match.
  pressure  c8/c32 concurrency against a host pool that cannot hold the working
            set: exercises the staging crop / capacity retry ladder.
  cancel    abort requests mid-prefetch, then verify the following requests and
            that host occupancy does not ratchet up.
  repeat    /flush_cache -> restore, five rounds: every round must report the
            same storage span and identical ids.

Usage: e051_verify.py <scenario> [--port N] [--out FILE]
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

CASES = json.load(open("/var/tmp/lfm3_ref/input_ids.json"))
K4 = CASES["k4"]
SHIFTED = [(t + 1237) % 60000 for t in K4]
LONG = (K4 + SHIFTED)[:8192]
L3 = os.environ.get("E051_L3", "/data/t011/tmp/e051_l3")


def prompt(length):
    return LONG[:length]


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
        except Exception as exc:
            return {"timeout": type(exc).__name__, "detail": str(exc)[:160]}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body[:300]}

    def flush(self, attempts=60):
        last = None
        for _ in range(attempts):
            last = self.req("/flush_cache", {})
            body = json.dumps(last).lower()
            if "flushed" in body and "not flushed" not in body:
                return last
            time.sleep(1.0)
        raise RuntimeError(f"/flush_cache never succeeded: {last}")

    def generate(self, ids, max_new=8, timeout=600, logprob=True):
        return self.req(
            "/generate",
            {
                "input_ids": list(ids),
                "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
                "return_logprob": logprob,
            },
            timeout=timeout,
        )

    def metrics(self):
        out = {}
        try:
            with urllib.request.urlopen(self.base + "/metrics", timeout=30) as f:
                body = f.read().decode()
        except Exception:
            return out
        for line in body.splitlines():
            if not line.startswith("sglang:"):
                continue
            name = line.split(" ")[0]
            if "hicache" in name or "storage_prefetch" in name:
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return out


def meta(out):
    m = out.get("meta_info") or {}
    return {
        "cached": m.get("cached_tokens"),
        "details": m.get("cached_tokens_details"),
        "ids": out.get("output_ids"),
        "lp": (m.get("output_token_logprobs") or None),
    }


def record(cases, length, append, out, path):
    m = meta(out)
    cases.append(
        {
            "len": length,
            "append": append,
            "path": path,
            "cached": m["cached"],
            "details": m["details"],
            "ids": m["ids"],
            "lp": [x[0] for x in m["lp"]] if m["lp"] else None,
            "raw": None if m["ids"] is not None else out,
        }
    )


def scenario_matrix(srv, baseline_cases):
    cases = []
    for length in (1024, 4096, 8192):
        for append in (5, 8, 10):
            ids = prompt(length) + [7] * append
            srv.flush()
            r1 = srv.generate(ids)
            record(cases, length, append, r1, "cold")
            r2 = srv.generate(ids)
            record(cases, length, append, r2, "l1")
            r3 = srv.generate(ids)
            record(cases, length, append, r3, "l1b")
            # Flood the device with unrelated prefixes to push the span to host.
            for i in range(12):
                srv.generate(prompt(1024)[: 512 + 37 * i] + [999] * 3, max_new=1, logprob=False)
            r4 = srv.generate(ids)
            record(cases, length, append, r4, "l2")
            srv.flush()
            r5 = srv.generate(ids)
            record(cases, length, append, r5, "l3")
    return cases


def scenario_baseline(srv):
    cases = []
    for length in (1024, 4096, 8192):
        for append in (5, 8, 10):
            ids = prompt(length) + [7] * append
            srv.flush()
            record(cases, length, append, srv.generate(ids), "baseline")
    return cases


def scenario_missing(srv):
    """wait_complete + a hole in the L3 chain: must terminate, must match ids."""
    out = {"rounds": []}
    ids = prompt(4096) + [7] * 8
    for i in range(3):
        srv.generate(ids)
    srv.flush()
    ref = srv.generate(ids)
    out["reference"] = {"cached": meta(ref)["cached"], "ids": meta(ref)["ids"]}
    files = sorted(os.listdir(L3))
    victims = files[len(files) // 2 : len(files) // 2 + max(1, len(files) // 6)]
    for v in victims:
        p = os.path.join(L3, v)
        if os.path.isfile(p):
            os.unlink(p)
    out["deleted"] = len(victims)
    out["files_left"] = len(os.listdir(L3))
    srv.flush()
    t0 = time.time()
    after = srv.generate(ids, timeout=180)
    dt = time.time() - t0
    m = meta(after)
    out["rounds"].append(
        {
            "path": "after_delete",
            "elapsed_s": dt,
            "cached": m["cached"],
            "details": m["details"],
            "ids": m["ids"],
            "match": m["ids"] == out["reference"]["ids"],
            "raw": None if m["ids"] is not None else after,
        }
    )
    out["metrics"] = srv.metrics()
    return out


def _one(srv, ids, results, idx, max_new=8):
    t0 = time.time()
    r = srv.generate(ids, max_new=max_new, logprob=False, timeout=600)
    dt = time.time() - t0
    m = meta(r)
    results[idx] = {"elapsed_s": dt, "cached": m["cached"], "details": m["details"],
                    "ids": m["ids"], "raw": None if m["ids"] is not None else r}


def scenario_pressure(srv, clients, rounds=4):
    """Concurrent rounds over a working set larger than the host pool."""
    out = {"clients": clients, "rounds": []}
    prefixes = [prompt(8192) for _ in range(4)]
    variants = [
        [p[i * 64 :] + [7] * (i + 3) for i, p in enumerate(prefixes)]
    ][0]
    for rnd in range(rounds):
        results = [None] * clients
        threads = []
        t0 = time.time()
        for i in range(clients):
            ids = variants[i % len(variants)]
            t = threading.Thread(target=_one, args=(srv, ids, results, i))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        dt = time.time() - t0
        ok = sum(1 for r in results if r and r["ids"])
        hits = [r["details"] for r in results if r and r["details"]]
        out["rounds"].append(
            {
                "round": rnd,
                "elapsed_s": dt,
                "completed": ok,
                "cached": [r["cached"] for r in results if r],
                "ttft_s": [round(r["elapsed_s"], 3) for r in results if r],
                "storage_hits": sum(
                    1 for d in hits if d and (d.get("storage") or 0) > 0
                ),
                "host_hits": sum(1 for d in hits if d and (d.get("host") or 0) > 0),
                "failures": [r["raw"] for r in results if r and not r["ids"]],
            }
        )
        print(
            f"  round {rnd}: {dt:.1f}s ok={ok}/{clients} "
            f"storage_hits={out['rounds'][-1]['storage_hits']} "
            f"host_hits={out['rounds'][-1]['host_hits']}",
            flush=True,
        )
    out["metrics"] = srv.metrics()
    return out


def scenario_cancel(srv):
    """Abort in-flight prefetches, then prove the pipeline still serves."""
    out = {"aborted": 0, "after": []}
    ids = prompt(8192) + [7] * 8
    srv.generate(ids)
    srv.flush()

    def abort_one():
        data = json.dumps(
            {
                "input_ids": list(ids),
                "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
            }
        ).encode()
        r = urllib.request.Request(
            srv.base + "/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(r, timeout=0.35).read()
        except Exception:
            out["aborted"] += 1

    ts = [threading.Thread(target=abort_one) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    time.sleep(2.0)
    before = srv.metrics()
    for i in range(3):
        r = srv.generate(ids)
        m = meta(r)
        out["after"].append(
            {"cached": m["cached"], "details": m["details"], "ids": m["ids"],
             "raw": None if m["ids"] is not None else r}
        )
    srv.flush()
    time.sleep(1.0)
    srv.generate(prompt(512) + [7] * 4, max_new=4, logprob=False)
    after = srv.metrics()
    out["host_used_before"] = before.get("sglang:hicache_host_used_tokens")
    out["host_used_after"] = after.get("sglang:hicache_host_used_tokens")
    out["metrics"] = after
    return out


def scenario_repeat(srv, length=1024):
    ids = prompt(length) + [7] * 8
    out = {"length": length, "rounds": []}
    ref = None
    for i in range(5):
        srv.flush()
        r = srv.generate(ids)
        m = meta(r)
        if ref is None:
            ref = m["ids"]
        out["rounds"].append(
            {
                "round": i,
                "cached": m["cached"],
                "details": m["details"],
                "ids_match": m["ids"] == ref,
                "raw": None if m["ids"] is not None else r,
            }
        )
        srv.generate(ids, max_new=4, logprob=False)
    out["metrics"] = srv.metrics()
    return out


def scenario_stress(srv, clients=32, rounds=4, length=8192):
    """Distinct-content rounds that overflow device+host: crop / capacity ladder."""
    out = {"clients": clients, "length": length, "rounds": []}
    for rnd in range(rounds):
        results = [None] * clients
        threads = []
        t0 = time.time()
        for i in range(clients):
            # Unique per client and per round: force real churn, no prefix reuse.
            ids = prompt(length)
            ids = ids[: length // 2] + [1000 + rnd * 37 + i] * 8 + ids[length // 2 + 8 :]
            t = threading.Thread(target=_one, args=(srv, ids, results, i))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        dt = time.time() - t0
        details = [r["details"] for r in results if r and r["details"]]
        out["rounds"].append(
            {
                "round": rnd,
                "elapsed_s": round(dt, 1),
                "completed": sum(1 for r in results if r and r["ids"]),
                "host_hits": sum(1 for d in details if (d.get("host") or 0) > 0),
                "storage_hits": sum(1 for d in details if (d.get("storage") or 0) > 0),
                "cached": sorted({r["cached"] for r in results if r}),
                "failures": [r["raw"] for r in results if r and not r["ids"]],
            }
        )
        print(
            f"  stress round {rnd}: {dt:.1f}s ok={out['rounds'][-1]['completed']}/{clients} "
            f"host_hits={out['rounds'][-1]['host_hits']} "
            f"storage_hits={out['rounds'][-1]['storage_hits']}",
            flush=True,
        )
    out["metrics"] = srv.metrics()
    return out



def scenario_capacity(srv, clients=8, length=4096, n_prefixes=12):
    """Host pool smaller than the working set: crop / defer / retry ladder."""
    out = {"clients": clients, "length": length, "prefixes": n_prefixes, "rounds": []}
    ids_list = []
    for i in range(n_prefixes):
        base = prompt(length)
        ids_list.append(
            base[: length // 2] + [2000 + i] * 8 + base[length // 2 + 8 :]
        )
    refs = {}
    for i, ids in enumerate(ids_list):
        refs[i] = meta(srv.generate(ids))["ids"]
    out["warmed"] = n_prefixes
    out["l3_files"] = len(os.listdir(L3))
    for rnd, width in enumerate((clients, clients * 4)):
        srv.flush()
        results = [None] * width
        threads = []
        t0 = time.time()
        for i in range(width):
            idx = i % n_prefixes
            t = threading.Thread(
                target=_one, args=(srv, ids_list[idx], results, i)
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        dt = time.time() - t0
        details = [r["details"] for r in results if r and r["details"]]
        matches = sum(
            1
            for i, r in enumerate(results)
            if r and r["ids"] == refs[i % n_prefixes]
        )
        out["rounds"].append(
            {
                "width": width,
                "elapsed_s": round(dt, 1),
                "completed": sum(1 for r in results if r and r["ids"]),
                "ids_match": matches,
                "host_hits": sum(1 for d in details if (d.get("host") or 0) > 0),
                "storage_hits": sum(1 for d in details if (d.get("storage") or 0) > 0),
                "cached": sorted({r["cached"] for r in results if r}),
                "failures": [r["raw"] for r in results if r and not r["ids"]],
            }
        )
        print(
            f"  capacity width={width}: {dt:.1f}s ok={out['rounds'][-1]['completed']}/{width} "
            f"ids_match={matches}/{width} host_hits={out['rounds'][-1]['host_hits']} "
            f"storage_hits={out['rounds'][-1]['storage_hits']}",
            flush=True,
        )
    out["metrics"] = srv.metrics()
    return out



def scenario_anchor(srv, churn=28, hits=4, length=8192):
    """Evict a prefetch's anchor while its L3 read is in flight."""
    out = {"churn": churn, "hits": hits, "length": length}
    ours = prompt(length) + [7] * 8
    ref = meta(srv.generate(ours))["ids"]
    for _ in range(2):
        srv.generate(ours, max_new=4, logprob=False)
    srv.flush()
    out["l3_files"] = len(os.listdir(L3))
    results = [None] * (churn + hits)
    threads = []
    t0 = time.time()
    for i in range(churn):
        ids = prompt(length)
        ids = ids[: length // 2] + [3000 + i] * 8 + ids[length // 2 + 8 :]
        t = threading.Thread(target=_one, args=(srv, ids, results, i, 8))
        t.start()
        threads.append(t)
    for j in range(hits):
        t = threading.Thread(
            target=_one, args=(srv, ours, results, churn + j, 8)
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    dt = time.time() - t0
    hit_results = results[churn:]
    out.update(
        {
            "elapsed_s": round(dt, 1),
            "completed": sum(1 for r in results if r and r["ids"]),
            "anchor_hits_completed": sum(1 for r in hit_results if r and r["ids"]),
            "anchor_hits_ids_match": sum(
                1 for r in hit_results if r and r["ids"] == ref
            ),
            "anchor_hits_cached": [r["cached"] for r in hit_results if r],
            "anchor_hits_details": [r["details"] for r in hit_results if r],
            "failures": [r["raw"] for r in results if r and not r["ids"]],
        }
    )
    out["metrics"] = srv.metrics()
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--port", type=int, default=30178)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--baseline", default=None, help="baseline JSON for matrix compare")
    args = ap.parse_args()
    srv = Server(args.port)
    t0 = time.time()
    if args.scenario == "matrix":
        res = {"cases": scenario_matrix(srv, None)}
    elif args.scenario == "baseline":
        res = {"cases": scenario_baseline(srv)}
    elif args.scenario == "missing":
        res = scenario_missing(srv)
    elif args.scenario == "pressure":
        res = scenario_pressure(srv, args.clients)
    elif args.scenario == "cancel":
        res = scenario_cancel(srv)
    elif args.scenario == "repeat":
        res = scenario_repeat(srv)
    elif args.scenario == "anchor":
        res = scenario_anchor(srv)
    elif args.scenario == "capacity":
        res = scenario_capacity(srv, args.clients)
    elif args.scenario == "stress":
        res = scenario_stress(srv, args.clients)
    else:
        raise SystemExit(f"unknown scenario {args.scenario}")
    res["wall_s"] = round(time.time() - t0, 1)
    text = json.dumps(res, indent=1)
    out = args.out or f"/var/tmp/e051_{args.scenario}.json"
    open(out, "w").write(text)
    print(f"wrote {out} ({res['wall_s']}s)")
    if args.scenario in ("matrix",) and args.baseline:
        base = json.load(open(args.baseline))["cases"]
        bmap = {(c["len"], c["append"]): c for c in base}
        bad = 0
        for c in res["cases"]:
            b = bmap[(c["len"], c["append"])]
            same = c["ids"] == b["ids"]
            dev = None
            if c["lp"] and b["lp"]:
                dev = max(abs(x - y) for x, y in zip(c["lp"], b["lp"]))
            if not same:
                bad += 1
            print(
                f"  {c['len']:5d}+{c['append']:2d} {c['path']:5s} cached={c['cached']} "
                f"ids_match={same} lp_dev={dev}"
            )
        print(f"  ids mismatches: {bad}/{len(res['cases'])}")


if __name__ == "__main__":
    main()
