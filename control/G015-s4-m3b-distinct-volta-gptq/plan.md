# Plan — G015 S4-M3B

## Established inputs

- M3A closed, shuffled path rejected; frozen `use_shuffle=false` remains safe path.
- Authoritative c4 GPTQ owner = 51.655% graph GPU time; 88 calls/forward. Gate/up 44×(M/N/K 4/24576/3072); down 44×(4/3072/12288); Q4/group128/GPTQ-v1/desc_act/nonidentity g_idx.
- E024 has six real c4 fixtures, packed-reference comparisons, candidate S1 route, timing, and controlled high-amplitude shuffled failure probe.

## Conclusion — M3B REJECTED

AutoGPTQ CUDA-old was integrated in the disposable S1 GPTQ seam, retaining S1 fallback and without Attention/Mamba changes. Graph1/2/4 capture/replay and c1/c2/c4 service waves completed. However, repeated deterministic greedy IDs diverged from frozen S1 at both 1K and 4K prompts, failing the explicit correctness gate. **M3B REJECTED.** No commit/push. Candidate graph ledger total was 36.900 ms/forward and old GPTQ node family absent; rejected output invalidates adoption. Frozen c4 baseline GPTQ residual remains 18.667655 ms/forward, 51.655%, 88 calls. Named mature compatible donors in this stage are exhausted; a new SM70 small-M GPTQ kernel/design requires separate authorization. Candidate patch was removed from local production source and disposable S1 extraction restored; V100 released at 1 MiB.
