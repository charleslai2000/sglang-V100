#!/usr/bin/env bash
# One bounded NCU ApplicationReplay fixture run. Invoke from the repo root.
set -Eeuo pipefail
ROOT=/home/charles/Workspaces/3rdparty/sglang-V100
ART=/mnt/data/pc01/sglang-v100-migration/experiments/E042
UUID=GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189
GPU_INDEX=1
PROJ=${1:?projection gate_up|down}; M=${2:?M=1|4}
[[ "$PROJ" == gate_up || "$PROJ" == down ]]; [[ "$M" == 1 || "$M" == 4 ]]
mkdir -p "$ART/logs" "$ART/reports" "$ART/tmp"
RUN="c57-${PROJ}-m${M}-$(date +%Y%m%dT%H%M%S)"
UNIT="e042-${RUN//[^a-zA-Z0-9]/-}"
LOG="$ART/logs/$RUN.log"; REP="$ART/reports/$RUN.ncu-rep"
exec > >(tee -a "$LOG") 2>&1
mem_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
gpu_snapshot() { nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader; }
printf 'RUN=%s ts=%s\n' "$RUN" "$(date --iso-8601=ns)"
printf 'PRECHECK MemAvailable_kB=%s\n' "$(mem_kb)"; gpu_snapshot
[[ $(mem_kb) -ge 8388608 ]] || { echo ABORT_MEM_FLOOR; exit 70; }
[[ $(gpu_snapshot | cut -d, -f1 | tr -d ' ') == "$UUID" ]] || { echo ABORT_UUID; exit 71; }
apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name --format=csv,noheader)
[[ -z "$apps" ]] || { echo "ABORT_GPU_BUSY $apps"; exit 72; }

SECTIONS=${E042_SECTIONS:-SourceCounters,WarpStateStats}
METRICS=${E042_METRICS:-}
ARGS=(--devices 0 --replay-mode application --app-replay-mode strict --app-replay-match all
      --app-replay-buffer file --profile-from-start on
      --kernel-name 'regex:.*packed_fp32_partial.*' --kernel-name-base demangled
      --launch-count 1 --cache-control none --clock-control none
      --force-overwrite --export "$REP")
if [[ -n "$METRICS" ]]; then ARGS+=(--metrics "$METRICS"); else IFS=, read -r -a SECARR <<< "$SECTIONS"; for sec in "${SECARR[@]}"; do ARGS+=(--section "$sec"); done; fi
TARGET="$ROOT/experiments/E042-g022-c57-source-sampling/ncu_fixture_target.py"
PY=/home/charles/miniconda3/envs/sglang-v100/bin/python
PYTHONPATH="$ROOT/sgl-kernel/python:$ROOT/python"
CUDA_HOME=/usr/local/cuda-12.8
printf 'COMMAND sudo systemd-run --scope MemoryMax=4G MemorySwapMax=0; root NCU mode=application buffer=file selected=packed_fp32_partial launch-count=1 sections=%s metrics=%s target=%s %s\n' "$SECTIONS" "$METRICS" "$PROJ" "$M"

sudo -n systemd-run --scope --unit="$UNIT" -p MemoryMax=4G -p MemorySwapMax=0 \
  --uid=charles --gid=charles \
  --setenv=CUDA_VISIBLE_DEVICES="$UUID" --setenv=PYTHONPATH="$PYTHONPATH" \
  --setenv=HF_HOME=/mnt/data/pc01/sglang-v100-migration/cache/hf \
  --setenv=HF_HUB_CACHE=/mnt/data/pc01/sglang-v100-migration/cache/hf/hub \
  --setenv=TRITON_CACHE_DIR=/mnt/data/pc01/sglang-v100-migration/cache/triton \
  --setenv=TORCHINDUCTOR_CACHE_DIR=/mnt/data/pc01/sglang-v100-migration/cache/inductor \
  --setenv=TMPDIR="$ART/tmp" --setenv=TMP="$ART/tmp" --setenv=TEMP="$ART/tmp" \
  --setenv=CUDA_HOME="$CUDA_HOME" --setenv=CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
  --setenv=PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  -- /usr/bin/sudo -n /usr/bin/env CUDA_VISIBLE_DEVICES="$UUID" /usr/local/cuda/bin/ncu "${ARGS[@]}" "$PY" "$TARGET" "$PROJ" "$M" measure &
CLIENT_PID=$!
CG=""
for _ in $(seq 1 100); do
  CG=$(sudo -n systemctl show "$UNIT.scope" -p ControlGroup --value 2>/dev/null || true)
  [[ -n "$CG" ]] && [[ -r "/sys/fs/cgroup$CG/memory.max" ]] && break
  sleep 0.1
done
if [[ -z "$CG" ]] || [[ ! -r "/sys/fs/cgroup$CG/memory.max" ]]; then
  echo 'ABORT cgroup path unavailable'; sudo -n systemctl stop "$UNIT.scope" || true; wait "$CLIENT_PID" || true; exit 73
fi
CGDIR="/sys/fs/cgroup$CG"
if [[ $(cat "$CGDIR/memory.max") != 4294967296 ]] || [[ $(cat "$CGDIR/memory.swap.max") != 0 ]]; then
  echo "ABORT cgroup limit unverified path=$CG memory.max=$(cat "$CGDIR/memory.max") swap.max=$(cat "$CGDIR/memory.swap.max")"
  sudo -n systemctl stop "$UNIT.scope" || true; wait "$CLIENT_PID" || true; exit 74
fi
echo "CGROUP path=$CG memory.max=$(cat "$CGDIR/memory.max") swap.max=$(cat "$CGDIR/memory.swap.max")"
RC=0; MAX_PEAK=0; MIN_AVAIL=999999999
while kill -0 "$CLIENT_PID" 2>/dev/null; do
  avail=$(mem_kb); usage=$(cat "$CGDIR/memory.current" 2>/dev/null || echo 0); peak=$(cat "$CGDIR/memory.peak" 2>/dev/null || echo 0); events=$(tr '\n' ' ' < "$CGDIR/memory.events" 2>/dev/null || true)
  (( usage > MAX_PEAK )) && MAX_PEAK=$usage
  (( avail < MIN_AVAIL )) && MIN_AVAIL=$avail
  printf 'SAMPLE ts=%s MemAvailable_kB=%s cgroup_current=%s cgroup_peak=%s events=%s gpu=' "$(date --iso-8601=ns)" "$avail" "$usage" "$peak" "$events"; gpu_snapshot
  if (( avail < 9437184 )); then
    echo "ABORT reason=MemAvailable_below_9GiB unit=$UNIT"
    sudo -n systemctl stop "$UNIT.scope" || true
    RC=75; break
  fi
  sleep 1
done
# Read final counters before systemd removes the transient scope cgroup.
if [[ -r "$CGDIR/memory.peak" ]]; then MAX_PEAK=$(cat "$CGDIR/memory.peak"); fi
if [[ -r "$CGDIR/memory.events" ]]; then FINAL_EVENTS=$(tr '\n' ' ' < "$CGDIR/memory.events"); else FINAL_EVENTS='scope-removed'; fi
wait "$CLIENT_PID" || RC=$?
printf 'FINAL MemAvailable_kB=%s cgroup_peak=%s min_sampled_MemAvailable_kB=%s events=%s gpu=' "$(mem_kb)" "$MAX_PEAK" "$MIN_AVAIL" "$FINAL_EVENTS"; gpu_snapshot
if [[ -f "$REP" ]]; then sha256sum "$REP"; stat -c 'REPORT size=%s bytes' "$REP"; else echo REPORT_ABSENT; fi
exit "$RC"
