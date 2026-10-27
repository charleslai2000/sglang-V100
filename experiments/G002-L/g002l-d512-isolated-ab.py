#!/usr/bin/env python3
"""Isolated D512 Gemma4 Full-attention reference and TileLang comparison.

Builds deterministic FP16 Q/K/V with page-size-16 paged KV and a non-identity
page table. Compares against SGLang's current SM70 Triton extend kernel. This is
an operator contract test, not a replacement for captured production activations.
"""
import argparse
import json
import math
import statistics
import sys
import time

import torch

sys.path.insert(0, "python")
from sglang.srt.layers.attention.tilelang_fa_v100 import paged_forward
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd

D, H, HKV, PAGE, SCALE = 512, 8, 2, 16, 1.0

def make_case(tokens, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(tokens, H, D, device=dev, dtype=torch.float16) * 0.03
    k = torch.randn(tokens, HKV, D, device=dev, dtype=torch.float16) * 0.03
    v = torch.randn(tokens, HKV, D, device=dev, dtype=torch.float16) * 0.03
    pages = math.ceil(tokens / PAGE)
    # Permute logical pages to exercise block-table addressing.
    perm = torch.arange(pages, device=dev, dtype=torch.int32).flip(0)
    phys_k = torch.empty_like(k)
    phys_v = torch.empty_like(v)
    for logical in range(pages):
        n = min(PAGE, tokens - logical * PAGE)
        physical = int(perm[logical])
        phys_k[physical * PAGE : physical * PAGE + n] = k[logical * PAGE : logical * PAGE + n]
        phys_v[physical * PAGE : physical * PAGE + n] = v[logical * PAGE : logical * PAGE + n]
    # Padded pages are never read, but initialize for sanitizer-friendly inputs.
    if tokens % PAGE:
        phys_k[pages * PAGE - (PAGE - tokens % PAGE) :] = 0
        phys_v[pages * PAGE - (PAGE - tokens % PAGE) :] = 0
    kp = phys_k.view(pages, PAGE, HKV, D).contiguous()
    vp = phys_v.view(pages, PAGE, HKV, D).contiguous()
    bt = perm.view(1, pages).contiguous()
    seq = torch.tensor([tokens], device=dev, dtype=torch.int32)
    qo = torch.tensor([0, tokens], device=dev, dtype=torch.int32)
    prefix = torch.tensor([0], device=dev, dtype=torch.int32)
    return q.contiguous(), k.contiguous(), v.contiguous(), kp, vp, bt, seq, qo, prefix

def run_triton(q, k, v, tokens):
    dev = q.device
    out = torch.empty_like(q)
    slots = torch.arange(tokens, device=dev, dtype=torch.int64)
    indptr = torch.tensor([0, tokens], device=dev, dtype=torch.int32)
    extend_attention_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), out,
        k.contiguous(), v.contiguous(), indptr, indptr, slots,
        None, True, None, tokens, 1.0, 1.0, SCALE,
        logit_cap=0.0, sliding_window_size=-1,
    )
    return out

def run_tile(case):
    q, _, _, kp, vp, bt, seq, qo, prefix = case
    return paged_forward(
        q, kp, vp, bt, seq, qo, prefix,
        block_size=PAGE, num_kv_heads=HKV, softmax_scale=SCALE,
        causal=True, sliding_window_size=-1,
    )[0]

def event_ms(fn, rounds=7):
    vals=[]
    for _ in range(rounds):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b))
    return vals

parser=argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, nargs="+", default=[1024,4096])
parser.add_argument("--rounds", type=int, default=7)
args=parser.parse_args()
assert torch.cuda.get_device_capability() == (7,0)
results=[]
for tokens in args.tokens:
    case=make_case(tokens, 20260301+tokens)
    q,k,v,*_=case
    triton=run_triton(q,k,v,tokens)
    tile=run_tile(case)
    delta=(tile.float()-triton.float()).abs()
    metrics={"mae":delta.mean().item(),"p99":torch.quantile(delta.flatten(),.99).item(),"max":delta.max().item(),"max_abs_ref":triton.float().abs().max().item()}
    # Warm and compile outside timing, then interleave order deterministically.
    for _ in range(2): run_tile(case); run_triton(q,k,v,tokens)
    torch.cuda.synchronize()
    tvals=[]; lvals=[]
    for i in range(args.rounds):
        fns=(run_triton,run_tile) if i%2==0 else (run_tile,run_triton)
        for fn in fns:
            if fn is run_triton: tvals += event_ms(lambda: run_triton(q,k,v,tokens),1)
            else: lvals += event_ms(lambda: run_tile(case),1)
    results.append({"tokens":tokens,"shape_q":list(q.shape),"shape_kv":list(k.shape),"dtype":str(q.dtype),"scale":SCALE,"page_size":PAGE,"gqa":H//HKV,"numerics":metrics,"triton_event_ms":tvals,"tilelang_event_ms":lvals,"triton_median_ms":statistics.median(tvals),"tilelang_median_ms":statistics.median(lvals)})
print(json.dumps(results,indent=2))
