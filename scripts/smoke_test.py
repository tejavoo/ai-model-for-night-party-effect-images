#!/usr/bin/env python3
"""Run a small end-to-end training/inference smoke test without coremltools."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="nightflash-smoke-"))
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "source"
    target = workdir / "target"
    output = workdir / "run"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_dummy_data.py"), "--output", str(workdir / "data"), "--count", "4", "--size", "64"],
        check=True,
    )
    source = workdir / "data" / "source"
    target = workdir / "data" / "target"
    checkpoint = output / "checkpoints" / "latest.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nightflash.train",
            "--source-dir",
            str(source),
            "--target-dir",
            str(target),
            "--output-dir",
            str(output),
            "--smoke-test",
        ],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )
    inferred = output / "inference" / "source-000.png"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nightflash.infer",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(source / "source-000.png"),
            "--output",
            str(inferred),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )
    print(f"Smoke test passed: {checkpoint}")
    print(f"Inference output: {inferred}")


if __name__ == "__main__":
    main()
