#!/usr/bin/env python3
"""Create tiny synthetic domains for smoke tests.

This is only a pipeline test. It is not a meaningful disposable-camera dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_image(path: Path, size: int, style: str, index: int) -> None:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    if style == "source":
        rgb = np.stack([0.25 + 0.55 * xx, 0.35 + 0.35 * yy, 0.65 - 0.25 * xx], axis=-1)
    else:
        rgb = np.stack([0.45 + 0.35 * xx, 0.22 + 0.25 * yy, 0.12 + 0.18 * (1 - xx)], axis=-1)
    wave = (0.04 * np.sin(index + xx * 9.0))[..., None]
    rgb = np.clip(rgb + wave, 0.0, 1.0)
    image = Image.fromarray(np.uint8(rgb * 255), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((size * 0.25, size * 0.2, size * 0.75, size * 0.8), outline=(240, 220, 180), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dummy")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    root = Path(args.output)
    for index in range(args.count):
        make_image(root / "source" / f"source-{index:03d}.png", args.size, "source", index)
        make_image(root / "target" / f"target-{index:03d}.png", args.size, "target", index + 3)
    print(f"Created {args.count} source and {args.count} target images in {root}")


if __name__ == "__main__":
    main()
