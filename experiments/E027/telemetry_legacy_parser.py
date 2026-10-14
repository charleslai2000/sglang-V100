#!/usr/bin/env python3
"""Parse G018 legacy telemetry CSV (schema v1, 14 data fields / 13 header).

Old rows are: script_unix_timestamp, nvidia_smi_timestamp, UUID, then the
remaining 11 queried values. This parser preserves that positional layout.
New schema-v2 captures use the aligned parser/schema and must not use this.
"""
import csv


def read_legacy(path):
    with open(path, newline="") as f:
        rows = csv.reader(f)
        header = next(rows)
        if len(header) != 13:
            raise ValueError(f"legacy header expected 13 fields, got {len(header)}")
        for line, row in enumerate(rows, 2):
            if len(row) != 14:
                raise ValueError(f"legacy row {line} expected 14 fields, got {len(row)}")
            yield {
                "script_timestamp": float(row[0]),
                "nvidia_smi_timestamp": row[1],
                "uuid": row[2],
                "gpu_util": float(row[3]),
                "memory_util": float(row[4]),
                "sm_clock_mhz": float(row[5]),
                "memory_clock_mhz": float(row[6]),
                "pstate": row[7],
                "power_w": float(row[8]),
                "temperature_c": float(row[9]),
                "memory_total_mib": float(row[10]),
                "memory_used_mib": float(row[11]),
                "throttle_supported": row[12],
                "throttle_active": row[13],
            }
