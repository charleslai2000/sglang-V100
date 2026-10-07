#!/usr/bin/env python3
"""Apply a reversible env-gated NVTX label around exactly one c96 decode forward."""
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text();assert 'S4_M1B_NVTX' not in s
anchor='import torch\n';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'import torch.cuda.nvtx as s4_m1b_nvtx\n',1)
old='''                        # FIXME: pp is not compatible with overlap
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
'''
new='''                        # FIXME: pp is not compatible with overlap
                        s4_m1b_capture = (
                            os.getenv("S4_M1B_NVTX") == "1"
                            and batch.forward_mode.is_decode()
                            and batch.batch_size() == 96
                        )
                        if s4_m1b_capture:
                            s4_m1b_range = s4_m1b_nvtx.range_start("S4_M1B_DECODE96")
                        try:
                            batch_result = self.model_worker.forward_batch_generation(
                                batch, **fwd_kwargs
                            )
                        finally:
                            if s4_m1b_capture:
                                s4_m1b_nvtx.range_end(s4_m1b_range)
'''
assert s.count(old)==1;s=s.replace(old,new,1)
old='''                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
'''
new='''                s4_m1b_capture = (
                    os.getenv("S4_M1B_NVTX") == "1"
                    and batch.forward_mode.is_decode()
                    and batch.batch_size() == 96
                )
                if s4_m1b_capture:
                    s4_m1b_range = s4_m1b_nvtx.range_start("S4_M1B_DECODE96")
                try:
                    batch_result = self.model_worker.forward_batch_generation(
                        batch, **kwargs
                    )
                finally:
                    if s4_m1b_capture:
                        s4_m1b_nvtx.range_end(s4_m1b_range)
'''
assert s.count(old)==1;s=s.replace(old,new,1);p.write_text(s)
print('patched',p)
