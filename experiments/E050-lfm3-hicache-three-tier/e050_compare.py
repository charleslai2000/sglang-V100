"""E050: compare HiCache restore paths against a plain recompute.

For one prompt it records, in a single server session:
  cold   -> flush, request (full recompute, may write through to L2/L3)
  l1a/l1b-> immediate repeats (device hit)
  l3     -> flush (drops L1+L2, keeps disk), request (storage hit)

Each record keeps output ids and per-token prompt logprobs so the restored KV /
SWA / conv state can be compared with the recomputed one token by token.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

CASES = json.load(open("/var/tmp/lfm3_ref/input_ids.json"))


class Server:
    def __init__(self, port=30177, host="127.0.0.1"):
        self.base = f"http://{host}:{port}"

    def req(self, path, payload=None, timeout=1800):
        data = json.dumps(payload).encode() if payload is not None else b"{}"
        r = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(r, timeout=timeout) as f:
                body = f.read().decode()
        except urllib.error.HTTPError as exc:
            return {"http_error": exc.code, "body": exc.read().decode()[:400]}
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

    def generate(self, ids, max_new=8, top_k=20):
        payload = {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
            "return_meta_info": True,
            "return_cached_tokens_details": True,
        }
        if top_k:
            # logprob_start_len must stay at the last prompt token: asking for
            # earlier prompt logprobs forces a full recompute and disables the
            # prefix cache entirely (measured: cached_tokens == 0).
            payload["return_logprob"] = True
            payload["logprob_start_len"] = len(ids) - 1
            payload["top_logprobs_num"] = top_k
        start = time.perf_counter()
        out = self.req("/generate", payload)
        wall_ms = (time.perf_counter() - start) * 1000
        meta = out.get("meta_info", {})
        details = meta.get("cached_tokens_details") or {}
        return {
            "wall_ms": round(wall_ms, 1),
            "e2e_ms": round(meta.get("e2e_latency", 0.0) * 1000, 1),
            "ttft_ms": round(meta.get("latency", 0.0) * 1000, 1) if meta.get("latency") else None,
            "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "device": details.get("device", 0),
            "host": details.get("host", 0),
            "storage": details.get("storage") or 0,
            "storage_backend": details.get("storage_backend"),
            "output_ids": out.get("output_ids") or [],
            "prompt_logprobs": (meta.get("input_token_logprobs") or []),
            "output_logprobs": (meta.get("output_token_logprobs") or []),
        }


def compare(a, b, label_a, label_b):
    """Compare generated tokens and the first-position candidate distribution."""

    def first_entry(rec):
        for entry in rec["prompt_logprobs"]:
            if entry is not None and entry[0] is not None:
                return entry
        return None

    ea, eb = first_entry(a), first_entry(b)
    first_diff = None
    topk_diff = None
    argmax_equal = None
    if ea is not None and eb is not None:
        first_diff = abs(ea[0] - eb[0])
        tops_a = {t[1]: t[0] for t in (ea[2] or [])}
        tops_b = {t[1]: t[0] for t in (eb[2] or [])}
        shared = set(tops_a) & set(tops_b)
        common_diff = [abs(tops_a[t] - tops_b[t]) for t in shared]
        if common_diff:
            topk_diff = max(common_diff)
        if tops_a and tops_b:
            argmax_equal = max(tops_a, key=tops_a.get) == max(tops_b, key=tops_b.get)
        shared_tokens = len(shared)
    else:
        shared_tokens = 0
    ids_equal = a["output_ids"] == b["output_ids"]
    out_diffs = [
        abs(x[0] - y[0])
        for x, y in zip(a["output_logprobs"], b["output_logprobs"])
        if x and y
    ]
    return {
        "pair": f"{label_a} vs {label_b}",
        "output_ids_equal": ids_equal,
        "first_token_logprob_abs_diff": first_diff,
        "candidate_logprob_max_abs_diff": topk_diff,
        "candidate_argmax_equal": argmax_equal,
        "candidate_shared": shared_tokens,
        "output_logprob_max_abs_diff": max(out_diffs) if out_diffs else None,
        "cached_tokens": {label_a: a["cached_tokens"], label_b: b["cached_tokens"]},
        "levels": {
            label_a: {"device": a["device"], "host": a["host"], "storage": a["storage"]},
            label_b: {"device": b["device"], "host": b["host"], "storage": b["storage"]},
        },
        "wall_ms": {label_a: a["wall_ms"], label_b: b["wall_ms"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30177)
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--tag", default="p1024")
    ap.add_argument("--skip-flush", action="store_true")
    args = ap.parse_args()

    srv = Server(args.port)
    prompt = CASES["k4"][: args.prompt]

    rec = {}
    if not args.skip_flush:
        srv.flush()
    rec["cold"] = srv.generate(prompt, args.max_new)
    rec["l1a"] = srv.generate(prompt, args.max_new)
    rec["l1b"] = srv.generate(prompt, args.max_new)
    srv.flush()
    rec["l3"] = srv.generate(prompt, args.max_new)

    report = {
        "tag": args.tag,
        "prompt_tokens": args.prompt,
        "max_new": args.max_new,
        "records": {k: {kk: vv for kk, vv in v.items() if kk != "prompt_logprobs"} for k, v in rec.items()},
        "prompt_logprob_summary": {
            k: {"n": len(v["prompt_logprobs"])} for k, v in rec.items()
        },
        "comparisons": [
            compare(rec["cold"], rec["l1b"], "cold", "l1b"),
            compare(rec["cold"], rec["l3"], "cold", "l3"),
            compare(rec["l1b"], rec["l3"], "l1b", "l3"),
        ],
    }
    # keep the raw per-token logprobs for offline re-analysis
    json.dump(
        {"report": report, "raw": {k: v["prompt_logprobs"] for k, v in rec.items()}},
        open(f"/var/tmp/e050_cmp_{args.tag}.json", "w"),
    )
    print(json.dumps(report["comparisons"], indent=1))
    for k, v in rec.items():
        print(k, {kk: v[kk] for kk in ("wall_ms", "cached_tokens", "device", "host", "storage")})


if __name__ == "__main__":
    main()
