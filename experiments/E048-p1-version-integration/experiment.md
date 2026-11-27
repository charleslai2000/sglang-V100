# E048 — P1 version integration: official v0.5.23 + fork SM70 increments

## Question

`main` stopped at E047 (`d4d15c7f8`). Two upstream sources carry work that is not
in this tree: official `sgl-project/sglang` tag `v0.5.23` and the SM70 fork
`haohervchb/sglang-V100` (`d85b3e27`). Which of their commits are genuinely
missing here, which have equivalents already, and does taking the missing SM70
increments (plus the `/v1/decisions` endpoint that v0.5.23 introduces) leave the
V100 harness correct and fast?

## Setup

| Item | Value |
|---|---|
| Base commit | `d4d15c7f8` (E047), branch `v100-harness` |
| Worktree | `/data/workspace/3rdparty/sglang-V100` (single worktree, no sibling) |
| GPU | Tesla V100-SXM2-16GB, cc 7.0 |
| Python | `/data/t011/repos/marlin_v100-g002/source/.venv/bin/python` (torch 2.9.1+cu128) |
| Oracle | `/var/tmp/oracle-venv/bin/python` (HF transformers, same checkpoint) |
| Model | `/var/tmp/lfm3` (LiquidAI/LFM3-1.7B, fp16) |
| Server flags | `--dtype float16 --attention-backend triton --sampling-backend pytorch --context-length 8448 --mem-fraction-static 0.60 --max-running-requests 32 --cuda-graph-max-bs 32 --mamba-scheduler-strategy extra_buffer` (graph ON, radix ON) |

## Ancestry (measured, not assumed)

| Ref | Commit | Relation to `d4d15c7f8` |
|---|---|---|
| `v0.5.23` (official tag) | `297e54fd10` | not an ancestor; `ours..tag` = 6 commits, `tag..ours` = 5021 |
| `haohervchb/main` | `d85b3e2705` | shared root `4c46ecde80`; merge-base `2d06a902a` |
| fork tip vs base | — | `base..fork` = 67, `fork..base` = 182 |

The clone is **shallow** (`.git/shallow` still holds 11 boundary commits after
deepening; the official lineage is still cut), so `git merge-base` against the
official tag returns empty and `git rev-list` cannot prove containment. The
per-commit evidence used instead: `git cherry` patch-ids plus content probes.

## Port scope

Ported (SM70/V100 generic, verified by content probe to be absent here):

| Fork commit | Content |
|---|---|
| `7707ea8a79` | warm prefill kernels to the resolved pool size (`max_warmup_bs = max_running_requests`) |
| `75204b6255`, `44691574b8` | opt-in custom all-reduce over PCIe P2P without NVLink (`SGLANG_CUSTOM_AR_ALLOW_PCIE`, byte cap) |
| `5a2a0336eb`, `d4540d2a1a` | preserve topology, push-only PCIe dispatch, unit tests |
| `37c257efe8`, `87a298480a` | check every peer before enabling custom reduction |
| `0e6b6a5542` (AR hunk only) | `SGLANG_SM70_AR_PUSH16` push-block cap for SM70 x TP4 |

Skipped, with reason:

| Area | Commits | Why |
|---|---|---|
| Qwen3.8 Flash-Next kernels, NF4 paths | `52996dfe53`, `28749dcc1a`, `75ff5f3cb5` | model-specific; no weights or target for it here |
| QSA indexer / ring / cache masking | `53bb5e249c`, `2ed917e42b`, `1be191c94a`, `95733d4f12` … | same |
| MTP / speculative tuning | `ac6c0a4ee1`, `9f6d5a7338`, `5771ab6cd5`, `23c879b38c`, `2a1c410b97` | speculative decoding is outside P1 scope; its unit tests stay red as before |
| Multimodal preprocessing | `0f7544834c`, `aecaae972b` | out of scope |
| Hyperconnection HC kernels | `0e6b6a5542` (rest) | needs the Qwen4-Exp model path |
| Official v0.5.23 tail (`grpc_bridge`, `rust_server` sidecar, `deepep_v2`, AMD CI) | `297e54fd10`, `0ff86de8a0`, `a09c1a096c`, `618136d4d3` | files absent here, features absent; merging them would pull unrelated subsystems |
| Official PD HiCache load-back | `2a7987f900` | HiCache is explicitly out of scope this phase |

Ported from v0.5.23 because P1 acceptance requires it:

1. `entrypoints/openai/serving_decisions.py` (new) + the decision protocol block
   in `protocol.py` + `POST /v1/decisions` in `http_server.py`.
2. Adapters so the endpoint works on this tree's runtime interfaces:
   prompt round-trip probe, `score_prompts` per-prompt label ids, probabilities
   → logprobs, `ReasoningParser` without `tokenizer=`, no `server_args` on the
   serving base.

## Verification plan

`/v1/score` and `/v1/decisions` against an independent HF oracle; the E046
correctness matrix, the E047 kernel references and the 8K oracle on the
integrated tree; `unit/mem_cache` + `unit/configs` + the mamba conv tests;
weight-free Gemma/Gemma2/Falcon-H1 construction; the E047 HTTP matrix as a
performance spot check; and a new hot-prefix 5/8/10-token incremental extend
benchmark with real cache-hit, GPU-busy and idle accounting.
