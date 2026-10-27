# G002-M D512 warp candidate results

## Decision

**NO-GO: retain the production default of 8 warps.** The sole candidate, opt-in D512 `num_warps=4`, had exact captured-logit equality but regressed both the seven Full-layer CUDA Event sum and total 1K/4K Forward Event. No further warp/tile search is justified by this evidence. TileLang D512 remains NO-GO (compiler error `Get different layout for V_ref`).

## Scope and reproducibility

- GPU: Tesla V100-SXM2-16GB, UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`.
- Model: `/data/t011/models/gemma4-w4a16-ct`, FP16, compressed-tensors W4A16; static memory fraction 0.76; context 8192; batch 1; output 1; CUDA Graph and radix cache disabled; SWA/full ratio 0.8.
- Baseline and candidate used identical per-length/repetition seed inputs and same binary/worktree. The candidate was enabled only through `SGLANG_G002M_D512_NUM_WARPS=4`; baseline had that variable unset. Default remains 8.
- Six A/B pairs: 1K and 4K, three repetitions each. Each pair was compared for input IDs, next-token IDs and full vocabulary logits.
- Re-run from repo root: `bash experiments/G002-L/g002m-run-forward-ab.sh`. It writes raw captures/logs under `/data/t011/artifacts/g002m/`; it requires the pinned model, V100 and local toolchain paths in the script.
- Targeted NCU layer attribution: `bash experiments/G002-L/g002l-ncu-layer-capture.sh`. It explicitly unsets the candidate variable and selects the exact layer 5 and 41 NVTX range. NCU replay time is not a production timing.

## A/B results

Medians of three real measured ModelRunner Forward CUDA Events and sums of the seven layer CUDA Events (ms):

| Input | Config | Forward median | Seven D512 Full layers median |
|---:|---|---:|---:|
| 1K | baseline, 8 warps | 313.710 | 50.904 |
| 1K | candidate, 4 warps | 413.045 | 149.757 |
| 4K | baseline, 8 warps | 1813.300 | 671.429 |
| 4K | candidate, 4 warps | 3341.978 | 2206.742 |

Each input/candidate comparison had identical input IDs and next-token IDs; logits shape `[1,262144]`; MAE, p99 absolute error and max absolute error were all exactly 0. Timings by repetition and raw comparison output can be reproduced from the artifact index below.

Candidate relative regressions: 1K Forward +31.7%, seven-layer sum +194.2%; 4K Forward +84.3%, seven-layer sum +228.6%. This is a clear NO-GO, not a marginal result. Service c1/c4 testing was not performed.

## Hardware evidence

The exact layer 5 and 41 D512 Triton `_fwd_kernel` launch was attributed by layer NVTX range plus CUDA Runtime/kernel correlation, not launch order or matching grid alone. The candidate NCU reports both show grid `(1,8,256)`, block `(128,1,1)`, 48 registers/thread, 64 KiB shared memory, 6.25% achieved occupancy, DRAM/L2 throughput 59.75%/27.82% (layer 5) and 59.64%/27.79% (layer 41), with long-scoreboard metric 17.58. These are the 4-warp candidate reports, not baseline reports. NCU is used for counters only; replay timings are discarded. The archived raw reports and CSVs are indexed below. These samples do not justify further parameter search.

## Artifact index

Raw artifacts are preserved outside the repository at `/data/t011/artifacts/g002m/`:

- `base-{1024,4096}-r{1,2,3}.{log,jsonl,pt}` and `w4-{1024,4096}-r{1,2,3}.{log,jsonl,pt}`: full execution logs, benchmark JSONL and captured input IDs/next token/full logits.
- `layer5-full.ncu-rep`, `layer41-full.ncu-rep`: NCU source reports; `layer5-raw.csv`, `layer41-raw.csv`: exported raw metrics; corresponding `layer*-ncu.log` and `layer*-prefill.jsonl`: capture logs and measured-run records.
- Compare any pair with `python experiments/G002-L/g002m-compare-forwards.py --base <base.pt> --candidate <w4.pt>`.
- Per-run event values are the `Prefill diagnostic Event` and `Prefill diagnostic layer ... CUDA Event` lines in the A/B `.log` files.

The raw NCU reports are not checked into Git because they are multi-megabyte generated binaries; the repository contains the acquisition command, result interpretation and exact external artifact paths.
