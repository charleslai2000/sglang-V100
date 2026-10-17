# E043 results — c57 Q4 vector-load study

## OBSERVED

### Provenance and instruction attribution
- Historical G017 source from commit `5fac74206b55464dca5993d943120746a24b959a`: `sgl-kernel/csrc/gemm/gptq/gptq_candidate_v8.cu`, SHA256 `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`.
- E026 build log `g017-close-final-rebuild.log` binds source to frozen final binary `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g017-close-final-sm70.so`, SHA256 `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`. E026 also records torch 2.9.1+cu128, CUDA 12.8. No rebuild was performed.
- Source: each CTA warp has lane `threadIdx.x`; `n0=blockIdx.x*128+threadIdx.x*4`; four qweights use `qw[row*N+n0+j]`, j=0..3. N and n0 are multiples of 128 and 4 respectively, so row*N+n0 is divisible by 4 (16-byte qweight alignment assuming base allocation is >=16B aligned; fixture runtime pointer alignment still to be measured). These are four adjacent uint32 words (16 bytes).
- Exact cubin SASS for `packed_fp32_partial` shows four qweight loads at 0x04a0/0x04b0/0x04c0/0x04d0, each 32-bit scalar, using `R46+{4,12,0,8}`; R46 traces to parameter slot `c[0][0x168]`. Source and host launch ABI map that slot to `qw`. Consumer sequence uses R48/R49/R51/R43 in `SHF/LOP3` nibble extraction. **The four U16 loads at 0x0560/0590/05a0/06f0 are scales**, based on `c[0][0x178]` and consumed by HADD2 at 0x06c0/0780/07c0/08b0. E040's earlier attribution was correct; E042's earlier Q4 attribution was wrong. Errata appended to E040/E042; raw and sampled reports unchanged.

### Isolated uint4 candidate and exact operator diagnostic
- `q4_vector.cu` uses the historical c57 partial/reducer arithmetic and launch geometry, replacing only four scalar `qw[...]` loads with `const uint4 packed=*reinterpret_cast<const uint4*>(qw+row*N+n0)`. Runtime base and row/thread address alignment checks require 16B alignment. Source SHA256 `ff51a85712f4e6d8b82550256fa1b28244804c0bf0ae733e123fa280183d8e30`; SM70 extension SHA256 `94f3a5c57d7297a17af508a2eb2055fa8abea888239c880270587ac40e1abe17`.
- Exact-c57 native-FP16 outputs matched bit-for-bit in all six L43 projection × M1/M2/M4 cells (`mismatch_count=0`); actual qweight pointer `%16==0` in all. This is a semantic identity diagnostic, not the independent strict pre-cast oracle or full six-fixture numerical qualification.
- Candidate SASS emits one `LDG.E.128.CONSTANT.SYS` at relative PC 0x04b0 for 16 Q4 bytes, versus exact-c57 four scalar 32-bit loads at 0x04a0–0x04d0. Scalar scales loads remain. `cuobjdump --dump-resource-usage`: candidate partial REG48, STACK0, SHARED0, LOCAL0; reducer REG12, STACK0, SHARED0, LOCAL0. Exact c57 partial baseline NCU launch register count 56; matched candidate NCU reports 48. Candidate shows no local/spill instructions/bytes. DRAM and L2 do not improve in matched one-cell NCU (details below); scalar instruction count reduction did not reduce global traffic.

### Alternating complete-Linear A/B
- Three fresh-process pairs per cell alternate order c57→candidate, candidate→c57, c57→candidate. Each timing closure includes partial, ascending split reducer, final native FP16 output; separate Eager and CUDA Graph replay. Raw JSON under `/mnt/data/pc01/sglang-v100-migration/experiments/E043/export/{gate_up,down}-m{1,2,4}-paired.json`, source hash and per-file SHA are in `artifacts.sha256` and immediately below.

| Projection/M | Eager candidate/c57 median ratio | Graph candidate/c57 median ratio | Graph interpretation |
|---|---:|---:|---|
| gate_up M1 | 1.0307 | 1.0452 | 4.5% slower |
| gate_up M2 | 1.0422 | 1.0570 | 5.7% slower |
| gate_up M4 | 1.0510 | 1.0567 | 5.7% slower |
| down M1 | 1.0000 | 1.0147 | 1.5% slower |
| down M2 | 1.0000 | 1.0286 | 2.9% slower |
| down M4 | 1.0124 | 1.0327 | 3.3% slower |

The aggregate M4 opportunity is decisively below the required 12% reduction; all Graph cells regress, M1/M2 provide no gain. No full N1/M2 stress matrix, numerical repair, Forward, model, or service tests were warranted.

### Matched NCU E040-style ApplicationReplay: gate_up M4
- Exact-c57 report `/mnt/data/pc01/sglang-v100-migration/experiments/E043/reports/c57-gate_up-m4-20261018T011110.ncu-rep`, SHA256 `c249f6ed17f4fc287551bc031e7164d94bd3d7b1d1d1d666f5ea6bf17a1bc584`.
- Candidate report `/mnt/data/pc01/sglang-v100-migration/experiments/E043/reports/candidate-gate_up-m4-20261018T011123.ncu-rep`, SHA256 `85121934e8ea991e4a92c5c0d7f9b66252991d571b1733ff93180ea02e4ce8e3`.
- Both selected only the corresponding partial kernel on the target V100, identical fixture, 4GiB MemoryMax/0 swap, external 9GiB abort. No cgroup events; peak cgroup was below 3.36GiB; minimum sampled MemAvailable 11,116,252kB; V100 idle at end. NCU warning noted: cache/clock not controlled; durations are instrumented kernel scope only.

| Metric | c57 | uint4 | Change |
|---|---:|---:|---:|
| Kernel duration (instrumented µs) | 229.504 | 248.160 | +8.1% |
| DRAM read MB | 38.321536 | 38.422016 | +0.3% |
| DRAM write MB | 16.301568 | 16.149152 | -0.9% |
| L2 read sectors | 4,311,288 | 4,355,907 | +1.0% |
| L2 write sectors | 1,401,076 | 1,390,727 | -0.7% |
| Global load instructions | 8,257,536 | 6,488,064 | -21.4% |
| Registers/thread | 56 | 48 | -14.3% |
| Long scoreboard % | 31.89 | 38.34 | +6.45 pp |
| Short scoreboard % | 0.034 | 0.084 | +0.050 pp |
| Local bytes/instructions | 0 / 0 | 0 / 0 | unchanged |

This shows one targeted implementation does remove scalar global-load instructions and lowers register use, but DRAM/L2 traffic is unchanged/slightly higher and kernel time is slower. NCU is diagnostic, not complete-operator timing; it is consistent with the paired A/B regressions.

#### Pair JSON SHA256
`gate_up-m1-paired.json` eb32963c399c958236dee9b1849be8bcbcf38239b71d793725a0b0352509d113; `gate_up-m2-paired.json` 0029d41c8dc3a39d01040938b2b0485eb940c5dad0da577236a6e64f88c4f11b; `gate_up-m4-paired.json` 4d02684c9bfcd0342603a93d861ebd61b090797cb56109df4e3a3f4ee1a71951; `down-m1-paired.json` 716e281b3853fcd3822645e9bdd8f5517c69323cc8290ed463e9911e4811ba91; `down-m2-paired.json` f68727e9007d53c77e5a1319813c98b8eec38c30618d3cf0f86ef1c0fab1823d; `down-m4-paired.json` 466980899e46d5e489febb815b9a6956a23df09bd406475e20565ce533a629ac.


## INFERRED
- Source layout permits a naturally aligned per-thread 16-byte Q4 vector read; actual fixture pointers confirmed `%16==0`. SASS proves one 128-bit load replaces four scalar load instructions, but matched NCU showed no traffic reduction and greater long-scoreboard share. Coalescing/sector traffic was already efficient enough that reducing instruction count alone did not help this implementation.
- The candidate’s additional pointer alignment assertion/fallback control path and changed instruction scheduling may explain its delay, but no individual cause is isolated. The observation is decisive for this exact candidate: it does not satisfy the required performance gate, regardless of why.

## UNRESOLVED
- Full strict pre-cast six-fixture×M1–4 and L43/down M4×4 numerical qualification, FP32 Graph scratch/cast/independent scratch/repeats; not warranted after operator gate failure.
- Matched NCU was collected only on gate_up M4; no need to expand counters after decisive complete-Linear regression.
- No whole-Forward, model, or service test; no numerical repair or production integration. No production performance claim.

## Bounded answer
**PERFORMANCE-NO-GO for the tested Q4 uint4 load implementation.** Exact provenance and SASS attribution are corrected: four 32-bit Q4 scalar loads became one `LDG.E.128`; four U16 instructions previously mistaken for Q4 are actually scales. Candidate matched exact c57 FP16 outputs on six L43 cells, but three genuinely alternating complete-Linear pairs show 1.033–1.057× Graph latency across all M1/M2/M4 rows, with M4 weighted opportunity far below the required 12% reduction. One matched gate_up M4 NCU report confirms 21.4% fewer global load instructions and 14.3% fewer registers, but DRAM reads +0.3%, L2 reads +1.0%, long-scoreboard +6.45 pp, and instrumented kernel duration +8.1%. No numerical repair, full correctness qualification, model/Forward/service tests or production integration were warranted after the decisive operator gate failure. This closes the tested implementation only; no Owner ACCEPTED.