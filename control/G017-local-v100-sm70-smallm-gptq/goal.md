# Goal: S4-M3C local V100 SM70 small-M GPTQ kernel

## Authorized outcome

Develop and qualify at least one new correct, CUDA-Graph-compatible SM70 GPTQ kernel for the frozen Falcon-H1 GPTQ-v1 Q4/group128/desc_act checkpoint on the local Tesla V100-SXM2-16GB, then demonstrate repeatable uninstrumented c4 service improvement with c1/c2 regressions bounded and an evidence-backed NCU explanation. The user explicitly authorizes new SM70 kernel development and production-path A/B under this Goal.

## Frozen authority and constraints

- G016 READY authority; local device UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`, Tesla V100-SXM2-16GB, CC7.0. Never use host RTX 2060 (`GPU-04ad2dd9-245d-0e7b-c96a-12da83a92b99`). No reboot/module reload; no disruption to other work.
- Model `tiiuae/Falcon-H1-7B-Instruct-GPTQ-Int4`, revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`; preserve Q4, group128, GPTQ-v1, `desc_act`, non-identity `g_idx` semantics, model math and outputs.
- Preserve production Attention, Mamba, CUDA Graph capture/replay sizes 1/2/4 and S1 runtime unless a separate explicit authorization is obtained. Nsys Falcon server trace is deferred/not qualified and is not a blocker; use targeted root NCU on the local V100.
- Existing production `use_shuffle=false` GPTQ route is correctness authority. The shuffled FP16 candidate is rejected; do not retry or integrate it.
- Do not rely on remote V100-32GB kernel timings or owner percentages as local evidence. Reconstruct the local baseline and profile both c4 geometries locally.
- New implementation must preserve semantics and pass real-weight/output, high-amplitude finite, graph, deterministic token, state isolation, and service acceptance before adoption. Do not relax numerical/token tolerance to make a candidate pass.
- Root is authorized only for tightly targeted NCU, explicitly pinned to the V100. Keep reports and working artifacts under `/mnt/data/pc01/sglang-v100-migration`; `/mnt/data` has limited free space. No unbounded captures/builds.

## Baseline facts available

- Production c4 mapping from prior E023: per forward 44 layers, each with gate_up M/N/K `4/24576/3072`, down `4/3072/12288`, Q4/group128/desc_act=true/use_shuffle=false; 88 GPTQ alt GEMM calls and historically 18.667655 ms/forward (51.655% of remote V100-32GB graph kernel intervals). This is a geometry hypothesis/starting map only; establish local identities/times.
- Existing local targeted NCU report: `/mnt/data/pc01/sglang-v100-migration/profiles/root-ncu-gptq-m1-targeted.ncu-rep`; identifies `gemm_half_q_half_alt_4bit_kernel`, V100 Device 0 CC7.0, with measured SM/DRAM/L2/resources/warp stalls. It is generic M1 geometry and cannot substitute for both requested c4 geometries.
- Source baseline and preceding control checkpoint: `66abb118847ebf74c698820da003350a352d0ca2`; production installer fix commit `36b35cb68fe493c202f7d6c64100de29738b2684`. Pinned runtime/model identities and local smoke evidence are in G016.

## Completion condition

At least one new production candidate has: (1) source and donor provenance; (2) local c4 NCU before/after for gate_up and down plus complete production-equivalent GPTQ operation timings; (3) real-weight real-activation correctness for L0/L22/L43 gate_up/down and high-amplitude finite stress with exact GPTQ zero/group/g_idx semantics; (4) graph capture/replay and deterministic greedy token equality with frozen S1 across 1K/4K c1/c2/c4; (5) paired repeated uninstrumented c1/c2/c4 service A/B on the same local V100, reporting TPS/ITL/VRAM and graph coverage; (6) profiler-backed residual ledger, reproducible configuration and a dedicated checkpoint commit. Accept only a repeatable material c4 gain without unacceptable c1/c2 regression. Otherwise provide measured hardware/software limitation evidence, restore frozen production source, and report the bounded result without claiming a ceiling.

## Exclusions

No Attention/Mamba changes, no shuffled-FP16 route, no GPU/driver changes, no Nsys worker injection, no broad tuning sweep without evidence, no claims based on synthetic-only kernels or remote timing, no adoption on compilation success alone.

## Numerical authority revision — 2026-10-09

By explicit user decision for G017, the mathematical correctness object is the candidate's actual pre-cast FP32 accumulator. The unchanged elementwise limit `abs(error) <= 1e-4 + 5e-5 * abs(reference)` is evaluated against GPTQ-v1 Q4/g128 dequantization and GEMM accumulated in FP64, then cast once to FP32. Production output remains FP16. Historical FP32 GEMM comparator results are preserved as diagnostics of its reduction path, not the sole mathematical authority. This is a versioned authority change; previous evidence files are immutable and not retroactively relabeled.

## Result

In progress; no ACCEPTED milestone. Local final split-K candidate SHA `02f923ea7329912a4413f179f292acb444a5e001caebf110b24f531cabf70d8b` passes the unchanged strict FP64-derived FP32 accumulator oracle on all six fixtures (artifact `candidate-final-fp64-authority.jsonl`, SHA `957311364c7b2986aabafb1e3ed318c2c24a6de9e9293e17be678a9d85c2c92b`). The exact final SHA also passes 18-cell greedy token parity against frozen S1 (artifact `candidate-token-identity-finalsha-v2.jsonl`, SHA `ff26dbe48c0f99090213ced0726bfd599df11df65da01891074c934bed6d6ca5`), and M4 eager/Graph capture/replay smoke; service graph1/2/4 startup and runtime dispatch are logged. Fresh paired c4 service gain is 6.3–6.7%, c1/c2 also improve; matched NCU and native FPM forward timing explain the service gain at aggregate level. Detailed GPU attribution still lacks Graph/eager tags and per-forward request IDs; a complete registered production checkpoint/audit is not yet done. S1 remains default, candidate stays isolated/opt-in.
