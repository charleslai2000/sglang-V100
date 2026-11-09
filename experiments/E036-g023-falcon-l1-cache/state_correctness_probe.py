#!/usr/bin/env python3
"""Serial force-miss/cache-hit comparison against the live Falcon service."""
import hashlib
import json
import os
import sys
import time

import requests
from transformers import AutoTokenizer

BASE = os.environ.get("G023_BASE", "http://127.0.0.1:30003")
MODEL = os.environ.get(
    "G023_MODEL", "/data/t011/hf/Falcon-H1-7B-Instruct-GPTQ-Int4"
)
MODE = os.environ.get("G023_MODE", "hit")
OUT = os.environ.get("G023_OUT", f"/data/t011/artifacts/g023-{MODE}.json")
EXPECTED_CACHED = {"miss": 0, "hit": 768}
if MODE not in EXPECTED_CACHED:
    raise SystemExit("G023_MODE must be miss or hit")

tok = AutoTokenizer.from_pretrained(
    MODEL, local_files_only=True, trust_remote_code=True
)
shared = tok.encode(
    "A quiet lighthouse stands above a cold northern sea. ",
    add_special_tokens=False,
)
prefix = (shared * 80)[:768]
suffixes = [
    tok.encode(text, add_special_tokens=False)
    for text in (
        "Write about branch alpha: The first ship turns west at dawn. Explain the scene in a short paragraph.",
        "Write about branch bravo: The second ship waits until midnight. Explain the scene in a short paragraph.",
    )
]
prompts = [prefix + suffix for suffix in suffixes]

def request(label, ids, max_new_tokens=256):
    started = time.perf_counter()
    response = requests.post(
        BASE + "/generate",
        json={
            "input_ids": ids,
            "rid": f"g023-{MODE}-{label}-{time.time_ns()}",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
                "sampling_seed": 908172,
            },
            "return_logprob": True,
        },
        timeout=240,
    )
    response.raise_for_status()
    result = response.json()
    meta = result.get("meta_info", {})
    row = {
        "label": label,
        "input_ids_sha256": hashlib.sha256(
            bytes(str(ids), "ascii")
        ).hexdigest(),
        "output_ids": result["output_ids"],
        "input_token_logprobs": meta.get("input_token_logprobs"),
        "output_token_logprobs": meta.get("output_token_logprobs"),
        "cached_tokens": meta.get("cached_tokens"),
        "prefill_tokens": meta.get("prompt_tokens"),
        "ttft": meta.get("e2e_latency"),
        "wall_seconds": time.perf_counter() - started,
        "meta": meta,
    }
    if len(row["output_ids"]) != max_new_tokens:
        raise AssertionError(
            f"{label}: expected {max_new_tokens} output tokens, got {len(row['output_ids'])}"
        )
    expected = 0 if MODE == "hit" and label == "cache-builder" else EXPECTED_CACHED[MODE]
    if row["cached_tokens"] != expected:
        raise AssertionError(
            f"{label}: expected cached_tokens={expected}, got {row['cached_tokens']}"
        )
    return row

# Build the checkpoint with exactly the shared prefix so its final recurrent
# state is the reference boundary, not an intermediate state from a longer prompt.
prime_ids = prefix
prime = request(
    "cache-builder",
    prime_ids,
    max_new_tokens=int(os.environ.get("G023_BUILDER_NEW_TOKENS", "1")),
)
first = request("branch-alpha-first", prompts[0])
repeat = request("branch-alpha-repeat", prompts[0])
branch = request("branch-bravo", prompts[1])
interleave = request("branch-alpha-after-bravo", prompts[0])
rows = [prime, first, repeat, branch, interleave]
report = {
    "mode": MODE,
    "base": BASE,
    "checkpoint": MODEL,
    "chunked_prefill_size": 2048,
    "dtype": "float16",
    "input_prefix_tokens": 768,
    "prompt_ids_sha256": [r["input_ids_sha256"] for r in rows],
    "expected_cached_tokens": EXPECTED_CACHED[MODE],
    "all_cached_tokens_validated": all(
        r["cached_tokens"] == (0 if MODE == "hit" and r["label"] == "cache-builder" else EXPECTED_CACHED[MODE])
        for r in rows
    ),
    "greedy_outputs_repeat_exact": first["output_ids"] == repeat["output_ids"],
    "greedy_outputs_after_branch_exact": first["output_ids"] == interleave["output_ids"],
    "first_logprob_repeat_max_abs_delta": max(
        (
            abs(a[0] - b[0])
            for a, b in zip(first["output_token_logprobs"], repeat["output_token_logprobs"])
            if a[0] is not None and b[0] is not None
        ),
        default=0.0,
    ),
    "first_logprob_interleave_max_abs_delta": max(
        (
            abs(a[0] - b[0])
            for a, b in zip(first["output_token_logprobs"], interleave["output_token_logprobs"])
            if a[0] is not None and b[0] is not None
        ),
        default=0.0,
    ),
    "first_repeat_divergence": next(
        (i for i, (a, b) in enumerate(zip(first["output_ids"], repeat["output_ids"])) if a != b), None
    ),
    "first_interleave_divergence": next(
        (i for i, (a, b) in enumerate(zip(first["output_ids"], interleave["output_ids"])) if a != b), None
    ),
    "same_prompt_outputs_repeat_equal": first["output_ids"] == repeat["output_ids"],
    "same_prompt_outputs_interleave_equal": first["output_ids"] == interleave["output_ids"],
    "records": rows,
}
Path = __import__("pathlib").Path
Path(OUT).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
