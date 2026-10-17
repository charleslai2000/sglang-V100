#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration/experiments/E043
UUID=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
PROJ=${1:?projection gate_up|down}; M=${2:?M}; ARM=${3:?c57|candidate}
[[ "$PROJ" == gate_up || "$PROJ" == down ]]; [[ "$M" == 1 || "$M" == 2 || "$M" == 4 ]]; [[ "$ARM" == c57 || "$ARM" == candidate ]]
mkdir -p "$ART/logs" "$ART/reports" "$ART/tmp"
RUN="${ARM}-${PROJ}-m${M}-$(date +%Y%m%dT%H%M%S)"; UNIT="e043-${RUN//[^a-zA-Z0-9]/-}"
LOG="$ART/logs/$RUN.log";REP="$ART/reports/$RUN.ncu-rep";exec > >(tee -a "$LOG") 2>&1
mem(){ awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
gpu(){ nvidia-smi -i 1 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader; }
[[ $(mem) -ge 8388608 ]] || { echo ABORT_MEM;exit 70; };[[ $(gpu|cut -d, -f1|tr -d ' ') == "$UUID" ]] || { echo ABORT_UUID;exit 71; }
apps=$(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader);[[ -z "$apps" ]] || { echo "ABORT_BUSY $apps";exit 72; }
PY=/home/charles/miniconda3/envs/sglang-v100/bin/python; TARGET="$ROOT/experiments/E043-g022-c57-q4-vector-loads/ncu_target.py"; export PYTHONPATH="$ROOT/sgl-kernel/python:$ROOT/python"
ARGS=(--devices 0 --replay-mode application --app-replay-mode strict --app-replay-match all --app-replay-buffer file --profile-from-start on --kernel-name 'regex:.*packed_fp32_partial.*' --kernel-name-base demangled --launch-count 1 --cache-control none --clock-control none --force-overwrite --export "$REP" --metrics gpu__time_duration.avg,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_aperture_device_op_read.sum,lts__t_sectors_aperture_device_op_write.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__sass_inst_executed_op_global_ld.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__stack_size,smsp__sass_data_bytes_mem_local.sum,smsp__sass_inst_executed_op_local.sum)
sudo -n systemd-run --scope --unit="$UNIT" -p MemoryMax=4G -p MemorySwapMax=0 --uid=charles --gid=charles --setenv=CUDA_VISIBLE_DEVICES="$UUID" --setenv=PYTHONPATH="$PYTHONPATH" --setenv=CUDA_HOME=/usr/local/cuda-12.8 --setenv=CUDACXX=/usr/local/cuda-12.8/bin/nvcc --setenv=PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:/usr/bin:/bin --setenv=TMPDIR="$ART/tmp" -- /usr/bin/sudo -n /usr/bin/env CUDA_VISIBLE_DEVICES="$UUID" /usr/local/cuda/bin/ncu "${ARGS[@]}" "$PY" "$TARGET" "$PROJ" "$M" "$ARM" &
PID=$!;CG='';for _ in $(seq 1 100);do CG=$(sudo -n systemctl show "$UNIT.scope" -p ControlGroup --value 2>/dev/null||true);[[ -n "$CG" && -r "/sys/fs/cgroup$CG/memory.max" ]]&&break;sleep .1;done
[[ -n "$CG" ]]||{ sudo -n systemctl stop "$UNIT.scope"||true;wait "$PID"||true;exit 73;};D=/sys/fs/cgroup$CG;[[ $(cat "$D/memory.max") == 4294967296 && $(cat "$D/memory.swap.max") == 0 ]]||{ sudo -n systemctl stop "$UNIT.scope"||true;wait "$PID"||true;exit 74;}
MIN=999999999;RC=0;while kill -0 "$PID" 2>/dev/null;do a=$(mem);((a<MIN))&&MIN=$a;echo "SAMPLE MemAvailable_kB=$a cgroup=$(cat "$D/memory.current") peak=$(cat "$D/memory.peak") events=$(tr '\n' ' '<"$D/memory.events") GPU=$(gpu)";if ((a<9437184));then sudo -n systemctl stop "$UNIT.scope"||true;RC=75;break;fi;sleep 1;done
wait "$PID"||RC=$?;echo "FINAL min_mem=$MIN GPU=$(gpu)";[[ -f "$REP" ]]&&sha256sum "$REP"||true;exit "$RC"
