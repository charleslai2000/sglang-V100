"""E050 timing: streaming TTFT/TPOT per cache level (cold / L1 device / L3 disk)."""

import json
import time
import urllib.error
import urllib.request

CASES = json.load(open("/var/tmp/lfm3_ref/input_ids.json"))
K4 = CASES["k4"]
SHIFTED = [(t + 1237) % 60000 for t in K4]
LONG = (K4 + SHIFTED)[:8192]


class Server:
    def __init__(self, port=30182):
        self.base = f"http://127.0.0.1:{port}"

    def req(self, path, payload=None, timeout=900):
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

    def stream_generate(self, ids, max_new=32):
        payload = {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
            "stream": True,
            "return_meta_info": True,
            "return_cached_tokens_details": True,
        }
        r = urllib.request.Request(
            self.base + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        first = None
        chunks = 0
        meta = {}
        with urllib.request.urlopen(r, timeout=900) as f:
            for raw in f:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body in ("", "[DONE]"):
                    continue
                event = json.loads(body)
                if "meta_info" in event and event["meta_info"]:
                    meta = event["meta_info"]
                text = event.get("text") or ""
                if text and first is None:
                    first = time.perf_counter()
                    chunks = 1
                elif text:
                    chunks += 1
        end = time.perf_counter()
        ttft = (first - start) * 1000 if first else None
        total = (end - start) * 1000
        details = meta.get("cached_tokens_details") or {}
        tpot = None
        if ttft is not None and chunks > 1:
            tpot = (total - ttft) / (chunks - 1)
        return {
            "ttft_ms": round(ttft, 1) if ttft else None,
            "total_ms": round(total, 1),
            "tpot_ms": round(tpot, 2) if tpot else None,
            "chunks": chunks,
            "cached": meta.get("cached_tokens"),
            "device": details.get("device", 0),
            "host": details.get("host", 0),
            "storage": details.get("storage") or 0,
        }


def main():
    srv = Server()
    out = {}
    for length in (1024, 4096, 8192):
        prompt = LONG[:length] + [7]
        rec = {}
        srv.flush()
        rec["cold_or_l3"] = srv.stream_generate(prompt)
        rec["l1a"] = srv.stream_generate(prompt)
        rec["l1b"] = srv.stream_generate(prompt)
        srv.flush()
        rec["l3a"] = srv.stream_generate(prompt)
        rec["l3b"] = srv.stream_generate(prompt)
        out[length] = rec
        print(length, json.dumps(rec), flush=True)
    json.dump(out, open("/var/tmp/e051_timing.json", "w"), indent=1)
    print("wrote /var/tmp/e051_timing.json")


if __name__ == "__main__":
    main()
