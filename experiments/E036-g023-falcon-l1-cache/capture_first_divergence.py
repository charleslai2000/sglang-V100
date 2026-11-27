#!/usr/bin/env python3
"""Helpers for correlatable G023 traces.

Standard module hooks cannot observe CUDA graph replay or access ForwardBatch
kwargs. Falcon's model-boundary recorder is therefore the authoritative path;
this module only provides reusable JSONL identity helpers for eager callbacks.
"""
import hashlib
import json
import os
import threading

import torch

_LOCK = threading.Lock()
_OUT = os.environ.get("G023_CAPTURE_PATH", "/data/t011/artifacts/g023-capture.jsonl")


def _hash(x):
    if not torch.is_tensor(x) or torch.cuda.is_current_stream_capturing():
        return None
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write(row):
    """Append one complete, self-identifying JSONL record."""
    required = ("request_id", "forward_mode", "decode_step", "execution")
    if any(key not in row for key in required):
        raise ValueError(f"trace row missing identity fields: {row}")
    with _LOCK, open(_OUT, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def _identity(batch, phase):
    mode = batch.forward_mode.name
    is_graph = bool(batch.forward_mode.is_cuda_graph())
    decode_mode = batch.forward_mode.is_decode()
    return {
        "request_id": list(batch.rids or []) or ["<synthetic-forward>"],
        "forward_mode": mode,
        "decode_step": int(batch.forward_iter) if decode_mode and batch.forward_iter is not None else None,
        "execution": "cuda_graph_capture" if is_graph and phase == "capture" else ("cuda_graph_replay" if is_graph else "eager"),
        "phase": phase,
    }


def mamba_factory(config):
    def hook(module, args, output):
        """No-op placeholder: kwargs are unavailable to standard forward hooks."""
        return
    return hook


def mamba_post_factory(config):
    return mamba_factory(config)


def model_factory(config):
    def hook(module, args, output):
        tensor = output
        if isinstance(output, (tuple, list)):
            tensor = next((x for x in output if torch.is_tensor(x)), None)
        if tensor is not None and torch.cuda.is_current_stream_capturing():
            return
        # Standard forward hooks do not receive ForwardBatch; don't emit
        # unjoinable records. Falcon's model-boundary trace is authoritative.
        return
    return hook
