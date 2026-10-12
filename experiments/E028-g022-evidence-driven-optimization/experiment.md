# E028 — G022 owner attribution and accepted GPTQ optimization reconciliation

## Question
Can archived G020/G021/G017 evidence identify a leading low-batch decode optimization owner and provide a valid starting S1→exact-SHA Candidate performance loop, without a new production source edit? A bounded follow-up added matched S1/Candidate NCU profiles at actual GPTQ M1/M2/M4 shapes.

## Prediction
The 1.90 ms c4 FP32 GEMV is lm_head but too small to dominate c4. The accepted G017 GPTQ Candidate should improve S1 at the full-op, CUDA Event forward, and service boundaries. It may be the completed first G022 optimization loop; a further source edit is not justified until its remaining owner budget is tied to a same-step measurement.

## Object and identity
Falcon-H1-7B-Instruct-GPTQ-Int4, revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`, Tesla V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`. Frozen S1 extension SHA-256 `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`; accepted G017 Candidate extension SHA-256 `02f923ea7329912a4413f179f292acb444a5e001caebf110b24f531cabf70d8b`. Candidate is exact-path/SHA opt-in; S1 stays default.

## Inputs and procedure
Inspect archived G019 raw Chrome traces and actual `step[DECODE bs=N]` annotation bounds; source-audit Falcon H1/LogitsProcessor; compare G021 NCU signature; verify final route module hashes and read E026 exact final service, token identity, operator, NCU, and forward Event artifacts. A bounded exact-SHA NCU follow-up profiled the frozen L43 M4 gate-up/down fixture at actual selected M=1,2,4 on the authorized V100, eight passes/report, with a MemAvailable >=8GiB floor and V100-only guard. The c1/c2 profiles selected the first 1/2 rows of the immutable M4 activation operand; no new service or whole-forward run. Distinguish same-step trace sums (can overlap), selected-kernel NCU, CUDA Event GPU-forward, and client service metrics. Record unsupported/missing contexts explicitly.

## Measurements needed
1. Exact large GEMV symbol/dtype/grid/count and lm_head call path; contrast NCU 62.08us grid835 signature.
2. Same actual c4 step owner category sums and ranking.
3. Exact-SHA S1/Candidate service TPS, TTFT, ITL/E2E; exact-SHA CUDA Event actual batch 1/2/4 distributions; matched M1/M2/M4 GPTQ NCU and correctness/token/state results.
4. Remaining unmeasured contexts and limits: 4K Event A/B, prefill/cache A/B, exact request joins, full Graph hardware counters, Other GEMM source attribution.

## Run constraints
No production source modification, candidate dispatch/default change, or model/service run. The bounded selected-kernel NCU follow-up used only the designated V100 and retained >=8 GiB MemAvailable. Preserve unrelated dirty source state, G017 S1/Candidate, G021/R17 scope, and source-runtime route defaults. No correctness tolerance change. Large raw reports stay in external artifact storage.
