# E026 — G017-CLOSE final clean-build qualification

Status: **PASS on final loader-hardened binary; independent checkpoint/push remains pending.** S1 is default; candidate is explicit opt-in only.

## Source/build identity

- Clean worktree `/home/charles/Workspaces/3rdparty/sglang-V100-G017-close`, branch `g017-close-prod-checkpoint`, base `66abb118847ebf74c698820da003350a352d0ca2`. Original dirty worktree unchanged.
- Full V100-only sgl-kernel wheel built from clean checkout: CUDA/nvcc 12.8.93, torch 2.9.1+cu128, Python 3.12, g++-12 12.4, SM70, one build job. Final wheel SHA `53f873251ccbe5a56e7608810a0b994a5a9559f27ce9d4f4a1f2c41dfd02a9ec`; final binary SHA **`c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`**, 24,050,016 bytes at `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g017-close-final-sm70.so`. This rebuild followed loader audit hardening; the earlier `570f22…` build is superseded.
- Final changed-source SHA-256 manifest:
  - `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py` `8da780d92924e522ecb6bc11fabfb182d43c829ef8c70887c775bca6b02b374c`
  - `sgl-kernel/CMakeLists.txt` `b21dd1504b6a5090d052d5c4a1560757422bcd4f29612b177e0413d70c301eae`
  - `sgl-kernel/csrc/common_extension.cc` `ef2356294953a314fb3d8c15840ae0b34a93efc5d61b97a620f64cfe70ed05bf`
  - `sgl-kernel/csrc/gemm/gptq/gptq_candidate_v8.cu` `fe4d9d3710a3900a8983d32567e5c9eff9c83acea7b3e04fccb50df3437949b1`
  - `sgl-kernel/python/sgl_kernel/__init__.py` `85a56bc141485e21fb3849b96e8a80a087f6676e33ddd02f49a5ed1a0a406258`
  - `sgl-kernel/python/sgl_kernel/load_utils.py` `bf4041587c32a870d122691d803101145668774efc1f39d2a29f1c32bf074ab7`
- Build log `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g017-close-final-rebuild.log`; base SHA and source manifest/toolchain are emitted by the checked-in build script. No prototype or diagnostic kernel branch is included.

## Final binary qualification and preserved evidence

- **Strict numerical authority: PASS 6/6**, zero bad elements using FP64 GPTQ dequant/GEMM cast once to FP32 and unchanged `1e-4 + 5e-5*abs(reference)` against actual pre-cast candidate FP32 accumulator. Final c57 rows `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g017-close-final-c57-fp64.jsonl`.
- Historical FP32 comparator remains a diagnostic, not authority: the preserved reference reports one FP32-vs-FP32 L43/down mismatch at `[0,1034]`; the strict FP64-derived comparator has zero. Original reference artifact `/mnt/data/pc01/sglang-v100-migration/experiments/E026/candidate-final-fp64-authority.jsonl` retains its provenance.
- FP16/high-amplitude stress on six fixtures: factors 1/2.5/4/8 finite, outputs equal FP32 accumulator cast to FP16; expected high-amplitude L43/down overflow at factor 16. Final c57 artifact `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g017-close-final-c57-fp16.jsonl`.
- Final binary op passed CUDA Graph capture/replay at M=1/2/4. Full clean service startup loaded exact c57 SHA, emitted actual candidate dispatch for M=4, captured Graph1/2/4 and reached ready. Log `/mnt/data/pc01/sglang-v100-migration/logs/g017-final-c57-candidate-server.log`. Service process was stopped and no server workers remain.
- Frozen qualification matrix retained because the final binary source difference after prior qualification is confined to loader/C++ include hardening; final exact c57 strict numerical, FP16 stress, M1/2/4 graph-op and full server dispatch/Graph startup were rerun. Prior exact frozen results: greedy token parity 18/18, seeded bounded candidate/S1 isolation both 3/3, and three-repeat service A/B over six cells with gains 6.2%–8.4% and improved ITL in all cells. Artifacts: `g017-close-clean-token-identity.jsonl`, `g017-close-clean-candidate-state-isolation-v5.json`, `g017-close-clean-candidate-control-isolation-seeded.json`, and `g017-close-clean-{s1,candidate}-service.json`. No broad profiler/tuning work was added.
- Only target V100 UUID `GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189` was used. Final idle reading: 7 MiB, 0%; no G017 server remains. RTX 2060 was never targeted.

## Integration audit

- CMake includes candidate CUDA TU and custom op registration only under `SGL_KERNEL_V100_ONLY`; default build source set and registered `gptq_gemm` were not altered. S1 dispatcher branch remains the existing `gptq_gemm` call; candidate requires explicit enable, no-shuffle Q4, FP16, SM70, M=1..4. It checks resolved loaded path and mandatory expected SHA before dispatch. No candidate use otherwise.
- Candidate loader verifies file exists and mandatory SHA matches **before importing**. Missing path, missing expected SHA, and mismatched SHA were exercised and raise ImportError. With candidate flag unset, normal architecture-local extension discovery remains the selector; optional `SGLANG_KERNEL_G017_S1_SHA256` pins and verifies the selected architecture-local S1 module without changing selection. Candidate custom op is registered only in V100-only binary. S1 default is unchanged.
- Normal production install contract is the built `sgl-kernel` wheel plus the package's standard import mechanism; opt in with `SGLANG_G017_Q4_FP32_CANDIDATE=1`, explicit `SGLANG_KERNEL_G017_CANDIDATE_PATH`, and mandatory `SGLANG_G017_CANDIDATE_SHA256=c57e…`. The tested host has a pre-existing editable finder that redirects imports to original dirty tree. A copied `/tmp/g017-ep-site` finder plus temporary Triton resolver accommodation was used only for clean-checkout qualification; no installed finder, original worktree, or system package was modified. Workaround details are outside the commit at `/mnt/data/pc01/sglang-v100-migration/experiments/E026/g017-close-runtime-package-override.txt`. **Production has no external runtime workaround dependency.**

## Residual and close gate

Service gain is confirmed; unmatched forward-time residual remains **UNKNOWN**. S1 remains default.

The following G017 files are staged for checkpoint: control Goal/Plan/T002/frontier; E026 report and qualification scripts; GPTQ dispatcher; V100-only CMake and op registration; candidate CUDA TU; sgl_kernel package init/loader. No binary or host workaround file is staged. Commit and push to a new dedicated ref, verify exact local/remote SHA and clean worktree, then update G017 to ACCEPTED. Until remote verification succeeds, keep ACTIVE.
