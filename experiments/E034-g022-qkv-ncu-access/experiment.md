# E034 — G022 QKV M2 matched-NCU feasibility check

## Question
Can an exact layer-0 Falcon QKV M2 input/weight pair run the previously correctness-screened SM70 small-GEMM route under a single-kernel, host-floor-guarded NCU capture to establish the required matched-counter evidence?

## Hypothesis
One exact-weight M2 target capture on the authorized V100 may be feasible under the already-used host-floor guard. This is a feasibility check only; no production route or candidate adoption is authorized.

## Object and identity
Falcon-H1-7B-Instruct-GPTQ-Int4 layer 0 Q/K/V weights from `/mnt/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4/model-00001-of-00002.safetensors`, concatenated and cast to runtime FP16 `[2048,3072]`; synthetic M2 activation `[2,3072]`. Target V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189`; source branch `main`, base `21835967efa1da62e77657997e084083f7886e6d`.

## Procedure
Run the already-present mature `sm70_small_gemm` function through an isolated probe, patching only its private Python shape table in-process for the exact QKV shape; compare to `torch.nn.functional.linear` at unchanged `rtol=0.002, atol=0.001`. Then attempt exactly one filtered NCU launch with a bounded metric set and only the authorized V100. Maintain `MemAvailable >= 8 GiB`; do not alter system perf-counter permissions or run a privileged workaround. Stop immediately if counters are not accessible.

## Measurements needed
Exact input/output shape, correctness, selected kernel identity, NCU availability, host MemAvailable and target GPU UUID. No broad profiling or service run.
