#!/usr/bin/env python3
"""Compare sequential per-K-group FP32 dot against c57/reference at failing output."""
import json,torch
from pathlib import Path
P=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E039/diagnose-failure-m4.json');d=json.load(open(P));print(json.dumps(d,indent=2))
