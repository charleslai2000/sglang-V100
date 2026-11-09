#!/usr/bin/env python3
"""Apply the pinned TileLang 0.1.8 SM70 BF16 compatibility patch."""
from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from pathlib import Path

EXPECTED_VERSION = "0.1.8"
PATCH = Path(__file__).resolve().parents[1] / "patches" / "tilelang-sm70-bf16-compat.patch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tilelang-root",
        type=Path,
        help="TileLang source root (defaults to the installed tilelang package root)",
    )
    args = parser.parse_args()

    version = importlib.metadata.version("tilelang")
    if version != EXPECTED_VERSION:
        raise SystemExit(
            f"TileLang {EXPECTED_VERSION} is required for this patch; found {version}"
        )
    if args.tilelang_root:
        root = args.tilelang_root.resolve()
    else:
        import tilelang

        root = Path(tilelang.__file__).resolve().parent
    if not (root / "src/tl_templates/cuda/common.h").is_file():
        raise SystemExit(f"TileLang CUDA template source is missing under {root}")

    check = subprocess.run(
        ["patch", "--batch", "--forward", "--dry-run", "-p1", "-i", str(PATCH)],
        cwd=root,
        check=False,
    )
    if check.returncode == 0:
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(PATCH)],
            cwd=root,
            check=True,
        )
        return

    # Treat an already-applied patch as success, but reject all other drift.
    reverse = subprocess.run(
        ["patch", "--batch", "--reverse", "--dry-run", "-p1", "-i", str(PATCH)],
        cwd=root,
        check=False,
    )
    if reverse.returncode != 0:
        raise SystemExit("TileLang source differs from both pristine and patched 0.1.8")
    print("TileLang 0.1.8 SM70 patch already present")


if __name__ == "__main__":
    main()
