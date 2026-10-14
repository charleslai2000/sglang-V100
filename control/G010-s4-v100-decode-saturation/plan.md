# Plan: S4 V100 decode saturation and owner ledger

## Frozen baseline and decode coverage

- Frozen S1 Falcon service identity `07343bd165855e9cd35ae63acea6cee5819c8c4e`; no replacement of baseline records.
- c96 1K/256 n=3 aggregate 391.20±0.98 tok/s; c104 400.74±0.80; c112 411.20±1.20. These operating points are not a hardware ceiling. Fixed node-level run c96 was 391.806 tok/s, 256 batch96 decode graph forwards, correct complete outputs.
- Existing Nsight node report has two graph forwards, 3006 graph-kernel activities, 278.024843ms total intervals/forward pair denominator equivalent 139.012422ms per forward. Nsight Systems CUDA kernel-interval percentages, not counter-based active-cycle shares.

## Corrected c96 decode owner ledger

The October 5 node report has a semantic owner misclassification: `_fwd_grouped_kernel_stage1` was listed as Mamba state/layout. Pinned source and graph adjacency show it is grouped GQA **decode Attention**, launched by `decode_attention.py::_decode_grouped_att_m_fwd`, immediately followed by its `_fwd_kernel_stage2` softmax/output reduction. Corrected existing owner rows (times/counts unchanged):

| Owner | Calls/forward | Geometry | ms/call | ms/forward | Share |
|---|---:|---|---:|---:|---:|
| Mamba selective scan | 44 | 32×96×24 | 1.00634 | 44.279 | 31.85% |
| Grouped split-KV decode Attention stage1 | 44 | 96×2×64; block 128; 128 regs/thread | 0.709606 | 31.223 | 22.46% |
| Attention split-KV final reduction stage2 | 44 | 96×12×1; block 128; 26 regs/thread | 0.01519 | 0.668 | 0.48% |
| GPTQ q4 row reconstruction | 88 | 192×384×1 | 0.21004 | 18.484 | 13.30% |

Full measured Attention kernel pair = 44×(0.709606+0.01519)=31.891ms/forward, 22.94% of graph-kernel intervals; QKV, rotary/cache-write and output projection remain separate owners. Falcon config geometry QH=12/KVH=2/D=128, 6 query heads per KV head, 44 full-attention layers. Triton source has block-H=16, block-N=16, min KV block=32, 4 warps, 2 stages. Server uses page size 16 and FP16 Q/K/V; page lookup uses per-forward kv_indptr/kv_indices over dynamic history; max split grid=64 with active split counts derived from seq lengths.

Other ledger rows (h884 GEMM 12.34%, WMMA 8.80%, cublas SGEMM 4.53%, Volta GEMV 1.65%, other) remain as originally measured. GPTQ row-reconstruction semantics are still unresolved/read-only. Corrected owner classification does not imply new timing or service result.

## Next action

G013/S4-M2 owns the grouped GQA Attention donor comparison. Frozen baseline, node trace and graph coverage remain sufficient; do not recapture unless a materially faster donor passes real-op correctness and paired service qualification. After M2 closes, proceed to exact GPTQ role mapping under G011/G010, without altering GPTQ in M2. Mamba SSU was evaluated under G011/G012; no re-open absent materially changed evidence. No prefill, scheduler retuning or V100 ceiling claim.
