# temporary live verification hook for M1B; installed only in candidate runtime
from pathlib import Path
p=Path('/hy-tmp/sglang-V100/python/sglang/srt/layers/attention/mamba/mamba.py')
s=p.read_text();marker='''            # Cast the per-head base before expansion so FP32 SSU preserves the
''';assert s.count(marker)==1
old='''            dt_d = dt_d.to(ssm_dtype)[:, :, None].expand(-1, -1, self.head_dim)
'''
new='''            dt_d = dt_d.to(ssm_dtype)[:, :, None].expand(-1, -1, self.head_dim)
            if os.getenv("S4_M1B_AUDIT") == "1" and dt_d.shape[0] == 96:
                state = selective_state_update
                if not hasattr(state, "_s4_m1b_orig"):
                    setattr(state, "_s4_m1b_orig", state)
                    def audited(*args, **kwargs):
                        if args[2].shape[0] == 96:
                            fp=open("/hy-tmp/sglang-logs/m1b-tie-audit.jsonl","a")
                            fp.write(__import__("json").dumps({"dt_stride": args[2].stride(), "A_stride": args[3].stride(), "dt_bias_stride": kwargs["dt_bias"].stride(), "D_stride": args[6].stride(), "mode": str(args[2].dtype)})+"\\n");fp.close()
                        return state._s4_m1b_orig(*args, **kwargs)
                    globals()["selective_state_update"] = audited
                    state._s4_m1b_orig=state
'''
assert s.count(old)==1;s=s.replace(old,new,1);p.write_text(s)
