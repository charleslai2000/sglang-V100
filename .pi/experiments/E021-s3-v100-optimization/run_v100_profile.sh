#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export PYTHONPATH=/hy-tmp/sglang-V100-s2-07343bd165/python
export S2_BASE=http://127.0.0.1:30003
export S3_OUT="${S3_OUT:?must set dedicated S3_OUT}"
exec python /hy-tmp/sglang-tmp/s3-v100-trace-workload.py
