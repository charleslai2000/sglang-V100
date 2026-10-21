# T011 frozen binary worker execution (diagnostic, not qualified)

- Binary path: `/data/t011/build/qpn128-final/frozen/qpn2_multiwarp-6f3ee5732ca571149c9209e403c1d03330a2a151a56541ca40b38c2f849af90f.so`
- Binary SHA256: `6f3ee5732ca571149c9209e403c1d03330a2a151a56541ca40b38c2f849af90f`
- Fresh worker harness `/data/t011/artifacts/t011-worker-identity.py`, SHA256 `8751567651d47b7f6e251b1f339b65aeb4510d80a74e1f65a1d4437cbf317001`.
- Worker log `/data/t011/artifacts/commonops128/t011-worker-identity.log`, SHA256 `a26d94b108ff88a6437c0483d7aab89049e988e28a699129de478c2ada2669a4`.
- Target UUID: `eaeb0cc6-58aa-8413-bef3-7fb075992189`; worker loaded and packed 88/88 fused runtime GPTQ Linear modules.
- Actual 512-token Prefill returned logits `(1,130049)` and hidden `(512,3072)`, then actual model DECODE calls at M1/M2/M4 returned respective logits `(1,130049)`, `(2,130049)`, `(4,130049)`. Dispatch log covered all 88 modules for M1 and uses exact binary SHA above.
- **Important failure:** Prefill hidden contained non-finite values (`PREFILL_OK ... finite False`). The diagnostic did not locate layer/token/state cause. M1/M2/M4 Decode reports finite hidden values, but were independent dummy decode forwards and not valid sequential request/state continuation. Thus no numerical correctness claim and no decoded state qualification.
- Allocator worker-load snapshot: allocated 12,247,307,776; reserved 13,371,441,152 bytes. After M1/M2/M4: allocated up to 12,256,083,456; reserved unchanged. These are not NVML/allocator peaks; capacity not certified.
- CUDA Graph was disabled in this worker. No state isolation/cancel/release/reuse, FP32 oracle, multi-step output, or service A/B was performed.
