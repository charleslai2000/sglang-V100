#!/usr/bin/env python3
"""CPU fake-arm test for the exact durable arm_start manifest writer."""
import json
import tempfile
from pathlib import Path
from cost_capture_manifest import record_arm_start

def main():
    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "fake.manifest.jsonl"
        with manifest.open("w") as stream:
            record_arm_start(stream, "cpu-run-1", "parallel",
                             "GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "/fake/parallel.so", "sha-fake", "/fake/report", 123)
            record_arm_start(stream, "cpu-run-1", "c57",
                             "GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "/fake/c57.so", "sha-c57", "/fake/report-c57", 124)
            stream.write(json.dumps({"event": "arm_end", "arm": "c57", "stdout": json.dumps({"event": "ready", "arm": "c57"})}) + "\\n")
            stream.flush()
        text = manifest.read_text()
        assert "\\n" in text, "CPU fixture must reproduce the historical literal-backslash-n defect"
        fixed_manifest = Path(d) / "fixed.manifest.jsonl"
        with fixed_manifest.open("w") as stream:
            record_arm_start(stream, "cpu-run-1", "parallel",
                             "GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "/fake/parallel.so", "sha-fake", "/fake/report", 123)
            record_arm_start(stream, "cpu-run-1", "c57",
                             "GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "eaeb0cc6-58aa-8413-bef3-7fb075992189",
                             "/fake/c57.so", "sha-c57", "/fake/report-c57", 124)
        lines = fixed_manifest.read_text().splitlines()
        assert len(lines) == 2, lines
        records = [json.loads(line) for line in lines]
        assert [r["event"] for r in records] == ["arm_start", "arm_start"]
        assert [r["arm"] for r in records] == ["parallel", "c57"]
        assert all(r["run_id"] == "cpu-run-1" for r in records)
        assert all(r["cuda_ordinal0_uuid"] == r["uuid"].removeprefix("GPU-") for r in records)
    print("PASS: CPU fake arm_start writes one parseable JSON object per manifest line")

if __name__ == "__main__":
    main()
