"""E050 probe: real L1/L2/L3 HiCache hit levels for LFM3 on V100.

Levels
  L1 = device radix cache (GPU KV)
  L2 = host pool (CPU memory, cache mode)
  L3 = file storage backend (disk)

Each request asks for `cached_tokens_details` so the level split is reported by
the server instead of being inferred from latency.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

CASES = json.load(open("/var/tmp/lfm3_ref/input_ids.json"))
BASE = CASES["k4"]


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

    def flush(self, attempts=40):
        """Flush touches only a fully idle scheduler, so retry until it sticks."""
        last = None
        for _ in range(attempts):
            last = self.req("/flush_cache", {})
            body = json.dumps(last)
            if "flushed" in body.lower() and "not flushed" not in body.lower():
                return last
            time.sleep(1.0)
        raise RuntimeError(f"/flush_cache never succeeded: {last}")

    def health(self, timeout=2):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=timeout):
                return True
        except Exception:
            return False

    def generate(self, ids, max_new=1):
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
        wall_ms = (time.perf_counter() - start) * 1000
        meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
        details = meta.get("cached_tokens_details") or {}
        return {
            "wall_ms": round(wall_ms, 1),
            "cached_tokens": meta.get("cached_tokens"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "device": details.get("device", 0),
            "host": details.get("host", 0),
            "storage": details.get("storage") or 0,
            "storage_backend": details.get("storage_backend"),
            "ids": out.get("output_ids") or [],
            "e2e_ms": round(meta.get("e2e_latency", 0.0) * 1000, 1),
            "raw_error": out.get("message") if "message" in out else None,
        }


def distinct_prefix(idx, length=1024):
    """A prompt that shares no long prefix with BASE or with other floods.

    Shifting every token id by a per-index offset keeps the sequences mutually
    disjoint (the vocabulary is 65k, so 4096-wide shifts at most wrap once).
    """
    offset = 4096 * (idx % 12) + 1
    # A unique first token keeps every flood prompt on its own radix path, so
    # the flood genuinely occupies device cache instead of sharing one prefix.
    return [1000 + idx] + [(tok + offset) % 60000 for tok in BASE[: length - 1]]


def flood(srv, count, length=1024):
    for i in range(count):
        srv.generate(distinct_prefix(i, length))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30177)
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--flood", type=int, default=200)
    ap.add_argument("--layer", default="all", choices=["all", "l1", "l2", "l3"])
    args = ap.parse_args()

    srv = Server(args.port)
    prompt = BASE[: args.prompt]
    results = {}

    if args.layer in ("all", "l1"):
        srv.flush()
        first = srv.generate(prompt)
        second = srv.generate(prompt)
        results["L1"] = {"cold": first, "repeat": second}
        print("L1 cold   ", {k: first[k] for k in ("cached_tokens", "device", "host", "storage", "wall_ms")})
        print("L1 repeat ", {k: second[k] for k in ("cached_tokens", "device", "host", "storage", "wall_ms")})

    if args.layer in ("all", "l2"):
        srv.flush()
        warm = srv.generate(prompt)
        flood(srv, args.flood)
        after = srv.generate(prompt)
        results["L2"] = {"warm": warm, "after_flood": after}
        print("L2 warm        ", {k: warm[k] for k in ("cached_tokens", "device", "host", "storage", "wall_ms")})
        print("L2 after flood ", {k: after[k] for k in ("cached_tokens", "device", "host", "storage", "wall_ms")})

    if args.layer in ("all", "l3"):
        srv.flush()  # clears L1 + L2, keeps L3 files
        verify = srv.generate(distinct_prefix(999, 128))  # a prompt with no L3 entry
        after_flush = srv.generate(prompt)
        results["L3"] = {"flush_sanity_cold_prompt": verify, "after_device_host_flush": after_flush}
        print("L3 flush ok, sanity", {k: verify[k] for k in ("cached_tokens", "device", "host", "storage")})
        print("L3 after flush ", {k: after_flush[k] for k in ("cached_tokens", "device", "host", "storage", "wall_ms")})

    out = f"/var/tmp/e050_probe_{args.layer}.json"
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
