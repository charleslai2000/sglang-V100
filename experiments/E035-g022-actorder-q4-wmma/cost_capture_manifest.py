#!/usr/bin/env python3
"""Durable JSONL manifest writes shared by the cost-capture launcher and CPU test."""
import json
import os

def write_manifest_record(stream, record):
    stream.write(json.dumps(record) + "\n")
    stream.flush()
    os.fsync(stream.fileno())

def record_arm_start(stream, run_id, arm, gpu_uuid, cuda_uuid, binary,
                     binary_sha256, report_prefix, started_ns):
    record = {
        "event": "arm_start",
        "run_id": run_id,
        "arm": arm,
        "uuid": gpu_uuid,
        "cuda_ordinal0_uuid": cuda_uuid,
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "report_prefix": str(report_prefix),
        "started_ns": started_ns,
    }
    write_manifest_record(stream, record)
    return record
