#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration/experiments/E044
UUID=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
PROJ=${1:?gateup|down}; M=${2:?1|4}; SPLIT=${3:?8|16|32}
case "$PROJ:$M:$SPLIT" in gateup:1:8|gateup:4:8|down:1:8|down:1:16|down:1:32|down:4:8|down:4:16|down:4:32) ;; *) echo invalid shape; exit 2;; esac
mkdir -p "$ART/logs" "$ART/reports" "$ART/tmp"
RUN="t011-${PROJ}-m${M}-s${SPLIT}-$(date +%Y%m%dT%H%M%S)"; UNIT="${RUN//[^a-zA-Z0-9]/-}"; LOG="$ART/logs/$RUN.log"; REP="$ART/reports/$RUN.ncu-rep"
mem_kb(){ awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
gpu(){ nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader | grep -F "$UUID"; }
[[ $(mem_kb) -ge 8388608 ]]; [[ $(gpu | cut -d, -f1 | tr -d ' ') == "$UUID" ]]
apps=$(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader); [[ -z "$apps" ]]
METRICS=${T011_METRICS:-gpu__time_duration.avg,smsp__cycles_active.sum,smsp__cycles_elapsed.sum,smsp__warps_eligible.sum,smsp__warps_active.sum,smsp__issue_active.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_aperture_device_op_read.sum,lts__t_sectors_aperture_device_op_write.sum}
TARGET="$ROOT/experiments/E044-g022-qpn2-sm70-actorder/t011_ncu_target.py"; PY=/home/charles/miniconda3/envs/sglang-v100/bin/python
if [[ "$PROJ" == gateup ]]; then export T011_FIXTURE=43-mlp-gate_up_proj-call0000; else export T011_FIXTURE=43-mlp-down_proj-call0000; fi
export T011_M="$M" T011_SPLIT="$SPLIT"
UNIT_LOG="$LOG"
(
  sudo -n systemd-run --scope --unit="$UNIT" -p MemoryMax=4G -p MemorySwapMax=0 -- /usr/bin/sudo -n /usr/bin/env T011_FIXTURE="$T011_FIXTURE" T011_M="$M" T011_SPLIT="$SPLIT" CUDA_VISIBLE_DEVICES="$UUID" PYTHONPATH="$ROOT/experiments/E035-g022-actorder-q4-wmma:$ROOT/experiments/E044-g022-qpn2-sm70-actorder:$ROOT/sgl-kernel/python:$ROOT/python" CUDA_HOME=/usr/local/cuda-12.8 /usr/local/cuda/bin/ncu --devices 0 --replay-mode application --app-replay-mode strict --app-replay-match all --app-replay-buffer file --profile-from-start on --kernel-name 'regex:.*kern.*' --kernel-name-base demangled --launch-count 1 --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite --export "$REP" "$PY" "$TARGET"
) > "$LOG" 2>&1
sha256sum "$REP" >> "$LOG"
