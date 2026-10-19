# Pi LSP — final post-restart acceptance

- Checked: 2026-10-18, after user restarted Pi.
- Workspace root: `/home/charles/Workspaces/3rdparty/sglang-V100` (Pi PID 2984194, cwd confirmed through `/proc`).
- Pi package: `@earendil-works/pi-coding-agent` 0.85.1.
- LSP extension: `/home/charles/Workspaces/3rdparty/pi-lsp-extension`, version 1.0.5.
- No production source edits; no server installation; no worktree/branch.

## Runtime/config activation

At inspection (04:11 local), current Pi PID 2984194 started 04:05:16. Active LSP child processes, all started after the restarted Pi:

| Server | PID | Start | Runtime evidence |
|---|---:|---|---|
| Pyright LSP | 2987170, 2989102 | 04:06:15, 04:06:56 | Both expose configured Conda PATH, `VIRTUAL_ENV=/home/charles/miniconda3/envs/sglang-v100`, and configured `PYTHONPATH`. Two process instances exist because the manager starts one per root.
| rust-analyzer | 2991813 | 04:07:44 | Executable is pinned toolchain binary `~/.rustup/toolchains/1.96.1-x86_64-unknown-linux-gnu/bin/rust-analyzer`; configured RUSTUP_TOOLCHAIN and external CARGO_TARGET_DIR visible in its environment.
| clangd-18 | 2997285 | 04:09:12 | Command line names `/mnt/data/sglang-V100/pi-lsp-cuda-validation/compile-db/cuda-source`.

All JSON configs parse. `pyrightconfig.json` selects Python 3.12, `venvPath=/home/charles/miniconda3/envs`, `venv=sglang-v100`, and project import roots. `.pi/settings.json` has Pyright environment, Rust binary/toolchain and clangd clean-CDB overrides. LSP workspace diagnostics for the configuration files returned 0 diagnostics. These are active-process/config evidence, not inferred from doctor alone.

## Python — PASS for environment/import resolution; PARTIAL for broad semantic diagnostics

Environment confirmed with `/home/charles/miniconda3/envs/sglang-v100/bin/python`:

- Python 3.12; torch 2.9.1+cu128 imports from the environment.
- `sglang` resolves to the editable project package.
- `gpu_uuid_lock` is not a separately installed distribution; local module is `experiments/E035-g022-actorder-q4-wmma/gpu_uuid_lock.py`, included in project/Pyright extra paths.

Actual post-restart Pi LSP requests:

- `diagnostics(test/srt/cpu/test_binding.py)` → **No diagnostics**. Before repair, Torch and SGLang imports were unresolved.
- `diagnostics(test/srt/cpu/test_activation.py)` → **No diagnostics**. Before repair, Torch and SGLang imports were unresolved.
- `documentSymbol(test/srt/cpu/test_binding.py)` → `TestGemm`, `test_binding`, locals.
- `goToDefinition(test/srt/cpu/test_binding.py, test_binding)` → same file, line 12.
- `findReferences(test/srt/cpu/test_binding.py, test_binding)` → one reference.
- `hover(...)` → `(method) def test_binding(self: Self@TestGemm) -> None`.
- `diagnostics(experiments/E044.../qualify_six.py)` → 2 remaining errors: NumPy uint-array right-shift and advanced-index typing. Its `documentSymbol` request returns the harness symbols. These are isolated harness typing issues, not hidden or disabled.
- `diagnostics(python/sglang/srt/layers/quantization/fp8_utils.py)` → no unresolved `torch`/`sglang`/`sgl_kernel`; remaining unresolved optional backends include `aiter`, `flashinfer`, `deep_gemm`. The environment probe confirms `aiter` and `deep_gemm` are not installed, while `flashinfer` is available to Python but loaded from an isolated migration dependency path not included in Pyright roots. Remaining tuple/Tensor/private Torch API diagnostics are Pyright type-check results, not import failure.
- `documentSymbol(fp8_utils.py)` → function/class symbols; `findReferences(use_aiter_triton_gemm_w8a8_tuned_gfx950)` → four references across `fp8.py` and `fp8_utils.py`; `goToDefinition` resolves to `fp8_utils.py:69`; hover returns `(n: int, k: int) -> bool`.

**Before/after:** test-file diagnostics changed from unresolved Torch/SGLang imports to clean 0-diagnostic results after restart. Broad production-module diagnostics are reduced to actual missing optional packages and type-check complaints; not suppressed.

## Rust — PARTIAL (navigation works; rust-analyzer diagnostics disagree with Cargo)

Crate facts: `rust/sglang-grpc/Cargo.toml` is a standalone Cargo root; edition 2024; `_core` cdylib; default feature `pyo3/extension-module`. Pinned analyzer is 1.96.1, target output at `/mnt/data/sglang-V100/rust-target`.

Fresh LSP requests:

- `documentSymbol(lib.rs)` → modules, `GrpcServerHandle`, `TokenizerInfo`, functions including `extract_tokenizer_info` and `start_server`.
- `findReferences(extract_tokenizer_info)` → declaration at line 94 and caller line 203.
- `goToDefinition` from caller line 203 → `lib.rs:94`.
- `hover` at definition → `fn extract_tokenizer_info(runtime_handle: &PyObject) -> PyResult<TokenizerInfo>`.
- `diagnostics(lib.rs)` still emits tracing callsite/value and PyO3 slot/method table type errors; `diagnostics(server.rs)` similarly flags tracing macros.

Independent compiler comparison (already run with locked dependencies):

```text
RUSTUP_TOOLCHAIN=1.96.1-x86_64-unknown-linux-gnu \
CARGO_TARGET_DIR=/mnt/data/sglang-V100/rust-target \
cargo check --manifest-path rust/sglang-grpc/Cargo.toml --locked
Finished `dev` profile; exit 0
```

Thus the reported tracing/PyO3 diagnostics are not compiler errors under the configured manifest/default feature set. The active rust-analyzer process has the requested environment and Cargo target path. Its child proc-macro server executable is under `/usr/local/rustup/toolchains/stable-.../libexec/rust-analyzer-proc-macro-srv`, not the user toolchain directory; likely analyzer/proc-macro toolchain skew is a concrete lead, but root cause is not proven. No Rust source changed. Further RA proc-macro/toolchain configuration diagnosis remains necessary; classify PARTIAL, not PASS.

## CUDA — PARTIAL; strict database scope respected

The configured authoritative DB is `/mnt/data/sglang-V100/pi-lsp-cuda-validation/compile-db/cuda-source/compile_commands.json`. It contains **one** real production-probe entry, retained from the prior production compile database after removing the stale E044 generated `.cu` path. The retained source is `python/sglang/jit_kernel/csrc/fast-hadamard-transform/hadamard_jit.cuh`; all source/include paths exist. Its recorded command specifies clang++-18, CUDA mode, CUDA include root, C++20, `--cuda-gpu-arch=sm_70`, and `-DSGL_CUDA_ARCH=700`. `clangd --check` emitted its exact command from the DB; direct compile passed host + sm_70 with only version/pragma-once warnings. This is a database-backed production TU, not a guessed NVCC command.

Post-restart Pi LSP evidence for that covered production `.cuh`:

- `documentSymbol` → `hadamard_mult_thread_chunk_12`, `fast_hadamard_transform_kernel`, launch helpers, kernel classes, etc.
- `goToDefinition` on `hadamard_mult_thread_chunk_12` → same file line 89.
- `findReferences` → two refs in that file.
- `hover` → exact function signature `template<int kNChunks> inline void hadamard_mult_thread_chunk_12(float x[kNChunks][12])`.
- `diagnostics` → only two include hygiene warnings (`static_switch.h`, `<algorithm>` unused directly).
- A **real cross-file** reference test on `SGL_DEVICE` in this same covered CUDA header: `goToDefinition` → `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh:99`; `findReferences` → 12 locations across the production `.cuh` and `utils.cuh`; hover returns the macro definition. This satisfies cross-file navigation on the actual covered production command.

Other existing production-looking files are not in the authoritative DB and are not declared verified:

- `python/sglang/jit_kernel/csrc/sm70_nvfp4_moe_decode.cu` and `experiments/E043.../q4_vector.cu` receive fallback/incomplete diagnostics (`Python.h` absent). Temporary syntax experiments showed this comes from PyTorch extension headers needing the Conda `include/python3.12` path. A trial adding that path made both compile with the retained SM70 flags, but **that include argument was not found in their actual JIT/extension build record**, so it was not promoted into the real CDB. Those two files remain **UNVERIFIED** as configured build TUs; their fallback diagnostics must not be treated as source failures or successful semantic validation.
- LSP can still parse symbols for `sm70_nvfp4_moe_decode.cu`, but that alone is not counted as database-backed.

The invalid E044 path was removed from original `production-probes/compile_commands.json`; clangd is configured to the clean copy. No guessed parameter was added to the authoritative database. CUDA rating PARTIAL because the required real DB covers the production probe and cross-file navigation, but not all requested production `.cu` files.

## Regression

- Markdown `documentSymbol(control/G022.../plan.md)` → `G022 plan`, `T010 result`, `Established constraints and evidence`; `diagnostics(experiment.md)` → no diagnostics. **PASS**.
- Ordinary C++ `sgl-kernel/csrc/cpu/gemm.cpp`: symbols include `pack_vnni`, `tinygemm_kernel`, `weight_packed_linear`; definition request returns a location, references for `pack_vnni` returns declaration + call; hover returns full function signature; diagnostics → no diagnostics. **PASS**.
- ARM-only `sgl-kernel/csrc/cpu/aarch64/gemm_int8.cpp` remains excluded from host x86 failure judgment; needs ARM target DB to assess.

## Change summary / diffs

- `.pi/settings.json`: Pyright PATH/VIRTUAL_ENV/PYTHONPATH and Pyright config root marker; direct pinned rust-analyzer path, toolchain and external Cargo target; clangd compile DB switched to clean external database.
- New `pyrightconfig.json` and E044-specific Pyright config.
- External compile database: stale E044 generated source removed; clean one-entry copy used by clangd. SM70 macros/architecture are from existing production probe record.
- `.pi/lsp-doctor.md` documents status and evidence.
- No production source edits. `git diff --check` clean at last check; unrelated/protected dirty workspace files were left untouched.

## Final ratings

| Area | Rating | Remaining |
|---|---|---|
| Python | **PASS** for environment/import resolution + representative test navigation; **PARTIAL** overall | E044 NumPy typing; optional absent backends and broad type diagnostics.
| Rust | **PARTIAL** | rust-analyzer emits errors contradicted by successful `cargo check`; proc-macro child appears from stable toolchain path.
| CUDA | **PARTIAL** | Real SM70 production probe command/navigation verified; other requested `.cu` TUs not in actual DB, so UNVERIFIED.
| C++ | **PASS** | Ordinary file tested clean; architecture-specific ARM file excluded.
| Markdown | **PASS** | Section symbols/diagnostics tested.

Do not report all PASS. No source changes were made and no commit was created for local LSP configuration/evidence.
