#!/usr/bin/env python3
"""Compare G002-M baseline and opt-in D512 launch candidate forward outputs."""
import argparse
import json
import os
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--candidate", required=True)
args = parser.parse_args()

base = torch.load(args.base, map_location="cpu", weights_only=False)
candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
for key in ("input_ids", "next_token_ids", "logits"):
    if key not in base or key not in candidate:
        raise SystemExit(f"missing {key} in capture")
assert base["input_ids"] == candidate["input_ids"], "input token IDs differ"
assert torch.equal(base["next_token_ids"], candidate["next_token_ids"]), "sampled token outputs differ"
a = base["logits"].float()
b = candidate["logits"].float()
d = (a - b).abs()
result = {
    "input_ids_equal": True,
    "shape": list(a.shape),
    "next_token_ids_equal": True,
    "mae": d.mean().item(),
    "p99": torch.quantile(d.flatten(), 0.99).item(),
    "max": d.max().item(),
    "max_abs_base": a.abs().max().item(),
}
print(json.dumps(result, indent=2))
