#!/usr/bin/env python3
"""Normalize selected NCU kernel rows from raw `ncu --import --page raw --csv` exports.

Usage: python parse_ncu_hotspots.py prefill.csv decode.csv out.csv
Only groups identical kernel-name/grid/block/device rows within each phase. Values
are means of NCU replay records and remain KERNEL_SCOPE; they are not whole-Forward
sums, complete Forward metrics, or GPU-active interval unions.
"""
import csv
import sys

METRICS = [
    'gpu__time_duration.sum',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__inst_executed_pipe_tensor.avg',
    'smsp__inst_executed_pipe_fp16.avg',
    'smsp__sass_thread_inst_executed_op_fadd_pred_on.avg',
    'smsp__sass_thread_inst_executed_op_fmul_pred_on.avg',
    'smsp__sass_thread_inst_executed_op_ffma_pred_on.avg',
    'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'lts__t_bytes.sum', 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'lts__t_sector_hit_rate.pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
]


def rows_for(path, phase):
    with open(path, newline='') as f:
        raw = list(csv.reader(f))
    header, units = raw[0], raw[1]
    data = [r for r in raw[2:] if r and r[0].strip()]
    idx = {name: header.index(name) for name in METRICS if name in header}
    grouped = {}
    for r in data:
        key = (r[6], r[10], r[9], r[11])  # exact kernel, grid, block, NCU device
        grouped.setdefault(key, []).append(r)
    for (kernel, grid, block, device), group in grouped.items():
        out = {'phase': phase, 'kernel_name': kernel, 'grid': grid,
               'block': block, 'ncu_device_index': device,
               'ncu_replay_records': len(group)}
        for metric in METRICS:
            if metric in idx:
                i = idx[metric]
                values = [float(r[i].replace(',', '')) for r in group]
                out[metric] = sum(values) / len(values)
                out[metric + '.unit'] = units[i]
            else:
                out[metric] = ''
                out[metric + '.unit'] = 'NOT_REPORTED'
        yield out


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    output = list(rows_for(sys.argv[1], 'PREFILL_EXTEND'))
    output.extend(rows_for(sys.argv[2], 'DECODE_GRAPH_REPLAY'))
    fields = ['phase', 'kernel_name', 'grid', 'block', 'ncu_device_index', 'ncu_replay_records']
    for metric in METRICS:
        fields += [metric, metric + '.unit']
    with open(sys.argv[3], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f'wrote {len(output)} kernel/grid rows to {sys.argv[3]}')


if __name__ == '__main__':
    main()
