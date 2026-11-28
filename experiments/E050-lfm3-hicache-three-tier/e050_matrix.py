"""E050 matrix: 1K/4K/8K prefixes x {5,8,10} token appends, per cache level.

Modes
  hicache   : flush -> request (cold or L3) -> repeat x2 (L1) -> flood (L2) -> flush (L3)
  baseline  : one request per case on a server without HiCache (pure recompute)

Every record keeps the generated ids and the per-token output logprobs so the
restored paths can be compared with the recomputed baseline token by token.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

CASES = json.load(open("/var/tmp/lfm3_ref/input_ids.json"))
K4 = CASES["k4"]
SHIFTED = [(t + 1237) % 60000 for t in K4]
LONG = (K4 + SHIFTED)[:8192]


def prompt_for(length):
    return LONG[:length]


class Server:
    def __init__(self, port, label):
        self.base = f"http://127.0.0.1:{port}"
        self.label = label

    def req(self, path, payload=None, timeout=300):
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

    def generate(self, ids, max_new):
        payload = {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
            "return_meta_info": True,
            "return_cached_tokens_details": True,
            "return_logprob": True,
            "logprob_start_len": len(ids) - 1,
        }
        start = time.perf_counter()
        out = self.req("/generate", payload)
        wall_ms = (time.perf_counter() - start) * 1000
        meta = out.get("meta_info", {})
        details = meta.get("cached_tokens_details") or {}
        return {
            "wall_ms": round(wall_ms, 1),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "device": details.get("device", 0),
            "host": details.get("host", 0),
            "storage": details.get("storage") or 0,
            "ttft_ms": round((meta.get("latency") or 0.0) * 1000, 2),
            "output_ids": out.get("output_ids") or [],
            "output_logprobs": meta.get("output_token_logprobs") or [],
        }


def flood(srv, count=260, length=1024):
    """Occupy device cache with prompts that share no prefix with the targets."""
    for i in range(count):
        ids = [1000 + i] + [((t + 4096 * (i % 12) + 1) % 60000) for t in K4[: length - 1]]
        srv.generate(ids, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30177)
    ap.add_argument("--mode", choices=["hicache", "baseline"], default="hicache")
    ap.add_argument("--lengths", default="1024,4096,8192")
    ap.add_argument("--tails", default="5,8,10")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--l2-lengths", default="")
    ap.add_argument("--flood", type=int, default=260)
    args = ap.parse_args()

    srv = Server(args.port, args.mode)
    lengths = [int(x) for x in args.lengths.split(",")]
    l2_lengths = {int(x) for x in args.l2_lengths.split(",")} if args.l2_lengths else set()
    tails = [int(x) for x in args.tails.split(",")]

    out = {}
    for length in lengths:
        base = prompt_for(length)
        for tail in tails:
            case = f"L{length}_tail{tail}"
            ids = base + [7] * tail
            rec = {}
            if args.mode == "baseline":
                srv.flush()  # every baseline case must be a pure recompute
                rec["baseline"] = srv.generate(ids, args.max_new)
            else:
                srv.flush()
                rec["first"] = srv.generate(ids, args.max_new)
                rec["l1a"] = srv.generate(ids, args.max_new)
                rec["l1b"] = srv.generate(ids, args.max_new)
                if length in l2_lengths and tail == tails[0]:
                    flood(srv, args.flood)
                    rec["l2"] = srv.generate(ids, args.max_new)
                    rec["l2b"] = srv.generate(ids, args.max_new)
                srv.flush()
                rec["l3"] = srv.generate(ids, args.max_new)
                rec["l3b"] = srv.generate(ids, args.max_new)
            out[case] = rec
            summary = {
                k: {
                    "wall_ms": v["wall_ms"],
                    "ttft_ms": v["ttft_ms"],
                    "cached": v["cached_tokens"],
                    "levels": [v["device"], v["host"], v["storage"]],
                }
                for k, v in rec.items()
            }
            print(case, json.dumps(summary), flush=True)

    path = f"/var/tmp/e050_matrix_{args.mode}.json"
    json.dump(out, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    main()
