# Plan: S4-M2 V100 decode Attention

## Established inputs

- Frozen S1 c96 baseline remains 391.20±0.98 aggregate tok/s (n=3); fixed node-level ledger describes two batch96 graph forwards and 3,006 kernels. See `experiments/S004-v100-decode-saturation/DECODE-NODE-LEDGER-2026-10-05.md`.
- Owner identity is now confirmed: the node ledger's title “Mamba state/layout” is wrong. In pinned source, `_fwd_grouped_kernel_stage1` is defined and launched by `sglang/srt/layers/attention/triton_ops/decode_attention.py::_decode_grouped_att_m_fwd`; `_fwd_kernel_stage2` immediately follows in graph order as its split-KV output reduction. The graph contains 44 stage1+stage2 pairs per forward, matching Falcon's full-attention layers. The grouped stage1 cost is 0.7096 ms/call and 22.46%; stage2 adds 0.0151 ms/call (0.48%), total Attention pair 0.7248 ms/call ×44 = 31.89 ms/forward (22.94% of graph kernel time), excluding adjacent QKV/rotary/cache-store and O projection which are separately charged owners.
- Exact geometry from node and source: graph grid stage1 `(96,2,64)`, block `(128,1,1)`, 0 static shared memory, 128 registers/thread; stage2 grid `(96,12,1)`, block `(128,1,1)`, 0 shared, 26 registers/thread. Falcon model geometry QH=12/KVH=2/D=128 ⇒ grouped heads=6, two head-tiles of six query heads, history per request dynamic through indptr and split=64 maximum; BLOCK_H=16, BLOCK_N=16, MIN_BLOCK_KV=32, `num_warps=4`, `num_stages=2`. Real runtime KV is FP16 with page size 16 and packed paged pool; exact tensor strides/page-index geometry still to be read from actual call fixture.
- Next measured non-Attention residual is GPTQ q4 row reconstruction, 88/forward, 0.210 ms/call, grid 192×384×1, 13.30%; exact semantic projection role still needs source/order mapping. GPTQ remains read-only during M2.
- Donor source reconciliation E018 establishes pinned 1Cat D128 generic decode and known adapter gaps; E016 historical work is exploratory and not production-exact. Compare its pinned scalar decode contract only after the live tensor/layout boundary is understood.

## Decisive frontier

First reconcile the asserted Attention residual with the authoritative ledger and decode model call path. Then identify full production operation, geometry, companions, operand/layout semantics; only after owner identity is proven, qualify mature donors against real c96 operands and benchmark fully accounted operations. If the ledger owner is in fact Mamba, do not relabel it as Attention; examine the actual Attention family instead and proceed only within M2.
