# E034 Results — QKV M2 matched-NCU feasibility

## OBSERVED
- Isolated probe loaded exact layer-0 Falcon Q/K/V BF16 checkpoint tensors from the stated safetensors shard, concatenated/cast them to FP16 `[2048,3072]`, and generated contiguous synthetic FP16 M2 activation `[2,3072]` on the designated Tesla V100. The mature small-M kernel was exposed in-process for this geometry only (no repository production edit).
- Output was finite `[2,2048]`; comparison with `F.linear` had max absolute error 0.00390625 and zero elements outside the unchanged `rtol=0.002, atol=0.001` bound.
- NCU initial name filter reported no kernels. Retried on exact function name `kernel`, `--launch-count 1`, bounded metrics and the authorized V100. NCU reported `ERR_NVGPUCTRPERM` and exited nonzero before a report. No counters were collected. No privilege/system permission change attempted.
- NCU attempts did not profile an application replay. Target V100 returned to 7 MiB/0%; host MemAvailable remained >8GiB (~15 GiB). No service or production source changes.

## INFERRED
- The exact-weight M2 isolated path is runnable and correctness-screened; it does not resolve performance or full production-path validity.
- Ordinary user-level NCU counter collection is currently blocked by device performance-counter permissions. Further NCU cannot be made safe/valid without an authorized change to that external access condition.

## UNRESOLVED
- No matched S1-vs-small-GEMM kernel NCU counters, production Graph Forward A/B, repeated service A/B, Prefill/cache outcomes, or capacity result. Exact production `QKVParallelLinear` dispatch, state/layout and service behavior remain unqualified. Do not add a production route.
- NCU access path cannot be fixed within this experiment without privileged/system intervention, which is not authorized.

## Bounded answer
The isolated exact-weight QKV M2 check passed the unchanged output bound, but NCU stopped at `ERR_NVGPUCTRPERM` with no report. No counter evidence or optimization qualification exists. Preserve the current production route and resume only if the authorized environment provides counter access.