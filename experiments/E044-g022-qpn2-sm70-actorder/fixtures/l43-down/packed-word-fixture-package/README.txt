Package: 43-mlp-down_proj-call0000 + strict-oracle tensors

Original fixture files (copied byte-for-byte; original manifest.json is unchanged):
- manifest.json
- x.bin
- qweight.bin
- qzeros.bin
- scales.bin
- g_idx.bin
- s1_output.bin (captured baseline output in fixture)

Derived stress inputs and saved FP32 tensors:
- c57_stress_x_fp16.bin: first four x rows multiplied by 4, rounded to FP16. Same stress input for both arms.
- stress_fp32_reference.bin: FP64 unpack/dequant/GEMM result cast to FP32, computed from the exact stress input.
- c57_stress_fp32_output.bin: exact c57 pre-cast FP32 output.
- packed_candidate_stress_fp32_output.bin: packed-word candidate pre-cast FP32 output.
- per-arm *_stress_fp32_reference.bin and *_stress_x_fp16.bin are redundant copies retained to make each arm self-contained.
- derived_manifest.json describes derived tensor provenance, binary SHAs, and oracle SHA.

FP32 tensors are raw contiguous little-endian float32 bytes with shape [4,3072], row-major. Stress input is raw contiguous little-endian float16 bytes with shape [4,12288]. Original x.bin remains the unscaled fixture [4,12288].

Binary SHA-256:
- exact c57: c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9
- packed-word candidate: 2238344028546fb9712ebcddba60f3d9659846a2dfbaad8483db77598e434b64

Both saved FP32 outputs are byte-identical in this stress case; both violate the strict elementwise tolerance at eight points. See packed-word-reuse-results.md in the repository for coordinates and error details.
