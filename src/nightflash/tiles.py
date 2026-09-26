"""Tiled inference for full-resolution output.

Why
---
Training runs on square crops (usually 256px), so the whole-image path
letterboxes a 12-megapixel photo down to 256px — a scale of about 0.06. Geometry
survives that, but so does almost no detail, which is exactly what the halation
and high-frequency heads are supposed to provide.

Tiling fixes the *scale* problem rather than the geometry problem: each tile is
processed at close to native resolution and stitched back together.

How
---
1. Cut overlapping tiles from the **original** image.
2. Run each tile through the normal letterbox pipeline, so every tile is seen at
   a scale near 1.0.
3. Accumulate into a float accumulator weighted by a raised-cosine (Hann) window,
   which falls to exactly zero at tile borders so neighbours crossfade.
4. Normalise by the accumulated weight.

The seams are therefore zero by construction rather than blended away afterwards.

Caveats
-------
* Halation near a tile border is computed from a truncated highlight field, so
  glow that should cross a boundary is weakened. The overlap hides most of it.
* ``region_exposure`` becomes **tile-local**: its 4x4 grid describes the tile,
  not the photograph. Disable it when tiling if you want global regions.
* The subject gate is likewise computed per tile, which is usually *better* for
  a large subject.
* Cost grows with the number of tiles. A 4032x3024 image at 512px tiles with 25%
  overlap needs about 50 tiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from PIL import Image

TileFn = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class TileBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def tile_boxes(
    width: int, height: int, tile_size: int, overlap: int
) -> list[TileBox]:
    """Cover the frame with exactly ``tile_size`` boxes, overlapping.

    Edge boxes deliberately extend *past* the image. Every box is therefore a
    full ``tile_size`` square, which keeps the blend window well defined at the
    frame border. Partial edge tiles would have a window that approaches zero
    there, and dividing by a near-zero weight turns image corners black.
    """

    if tile_size < 8:
        raise ValueError("tile_size must be at least 8")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    step = max(1, tile_size - overlap)

    def axis(total: int) -> list[int]:
        if total <= tile_size:
            return [0]
        starts = list(range(0, total - tile_size + 1, step))
        if starts[-1] != total - tile_size:
            starts.append(total - tile_size)
        return starts

    return [
        TileBox(x, y, x + tile_size, y + tile_size)
        for y in axis(height)
        for x in axis(width)
    ]


def _hann_1d(length: int, floor: float = 1e-2) -> np.ndarray:
    """Raised cosine with a small positive floor at both ends.

    The floor matters: ``np.hanning`` starts at ~2e-8, and a near-zero weight
    makes the blend numerically fragile. Because the accumulated value is
    ``sum(w * pixel) / sum(w)``, the *same* weight must be used in numerator and
    denominator — clamping only one of them scales the result and turns corners
    black. A 1% floor is far too small to create a visible seam.
    """

    if length <= 1:
        return np.ones(max(length, 1), dtype=np.float64)
    ramp = np.hanning(length + 2)[1:-1]
    return np.maximum(ramp, floor).astype(np.float64)


def blend_window(box: TileBox, floor: float = 1e-3) -> np.ndarray:
    """Separable Hann window with a small positive floor.

    The separable form squares the per-axis floor, so the corner weight would
    otherwise be 1e-2 * 1e-2 = 1e-4. That is still numerically safe because the
    same weight is used in numerator and denominator, but clamping the product
    makes the guarantee explicit rather than emergent.
    """

    window = np.outer(_hann_1d(box.height), _hann_1d(box.width))
    return np.maximum(window, floor)


def extract_tile(
    image: Image.Image, box: TileBox, pad_mode: str = "edge"
) -> tuple[Image.Image, TileBox]:
    """Crop ``box`` from ``image``, padding where it runs off-frame.

    ``pad_mode`` controls what the model sees outside the frame:

    ``edge``
        Replicate the border row/column. The safe default; never invents a
        gradient and never introduces a hard black bar.
    ``reflect``
        Mirror the border. Often better for images with strong vertical
        structure, but it can create a visible mirrored continuation.
    ``black``
        Zero fill. Only correct if the model was trained on padded black borders.
    """

    valid_left = max(0, box.left)
    valid_top = max(0, box.top)
    valid_right = min(image.width, box.right)
    valid_bottom = min(image.height, box.bottom)

    inner = image.crop((valid_left, valid_top, valid_right, valid_bottom))
    pad_left = valid_left - box.left
    pad_top = valid_top - box.top
    pad_right = box.right - valid_right
    pad_bottom = box.bottom - valid_bottom

    if pad_left or pad_top or pad_right or pad_bottom:
        array = np.asarray(inner, dtype=np.uint8)
        if pad_mode == "black":
            mode = "constant"
        elif pad_mode == "reflect":
            height, width = array.shape[:2]
            if pad_top >= height or pad_bottom >= height or pad_left >= width or pad_right >= width:
                mode = "edge"  # reflection cannot exceed the axis length
            else:
                mode = "reflect"
        else:
            mode = "edge"
        array = np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=mode,
        )
        inner = Image.fromarray(array, mode="RGB")

    visible = TileBox(
        pad_left, pad_top,
        pad_left + (valid_right - valid_left),
        pad_top + (valid_bottom - valid_top),
    )
    return inner, visible


def iter_tiles(
    image: Image.Image, tile_size: int, overlap: int, pad_mode: str = "edge"
) -> Iterator[tuple[TileBox, Image.Image]]:
    for box in tile_boxes(image.width, image.height, tile_size, overlap):
        tile, _ = extract_tile(image, box, pad_mode)
        yield box, tile


def tile_count(width: int, height: int, tile_size: int, overlap: int) -> int:
    return len(tile_boxes(width, height, tile_size, overlap))


def apply_tiled(
    image: Image.Image,
    process: TileFn,
    *,
    tile_size: int = 512,
    overlap: int = 128,
    pad_mode: str = "edge",
    progress: Callable[[int, int], None] | None = None,
) -> Image.Image:
    """Run ``process`` over overlapping tiles and crossfade the results.

    ``process`` receives a full ``tile_size`` tile (padded per ``pad_mode`` where
    the tile runs past the frame) and must return a same-sized image.
    """

    boxes = tile_boxes(image.width, image.height, tile_size, overlap)
    accumulator = np.zeros((image.height, image.width, 3), dtype=np.float64)
    weights = np.zeros((image.height, image.width), dtype=np.float64)

    for index, box in enumerate(boxes, start=1):
        tile, visible = extract_tile(image, box, pad_mode)
        result = process(tile)
        if result.size != tile.size:
            raise ValueError(
                f"tile processor changed size: {tile.size} -> {result.size}"
            )
        patch = np.asarray(result.convert("RGB"), dtype=np.float64)

        # Window over the full tile, then keep only the part that is real.
        window = blend_window(TileBox(0, 0, tile_size, tile_size))
        crop = patch[visible.top : visible.bottom, visible.left : visible.right]
        sub_window = window[
            visible.top : visible.bottom, visible.left : visible.right
        ]

        y0, y1 = max(0, box.top), min(image.height, box.bottom)
        x0, x1 = max(0, box.left), min(image.width, box.right)
        accumulator[y0:y1, x0:x1] += crop * sub_window[..., None]
        weights[y0:y1, x0:x1] += sub_window

        if progress is not None:
            progress(index, len(boxes))

    # The blend window has a positive floor and every pixel is covered by at
    # least one full tile, so the denominator is never near zero. Clamping it here
    # would desynchronise numerator and denominator, which scales the result.
    if not np.all(weights > 0):
        raise RuntimeError("tiled blend produced an uncovered pixel")
    blended = accumulator / weights[..., None]
    return Image.fromarray(np.uint8(np.clip(blended, 0.0, 255.0)), mode="RGB")


def estimate_tiles(
    width: int, height: int, tile_size: int, overlap: int
) -> dict[str, float | int]:
    count = tile_count(width, height, tile_size, overlap)
    return {
        "image": f"{width}x{height}",
        "tiles": count,
        "tile_size": tile_size,
        "overlap": overlap,
        "megapixels_processed": round(count * tile_size * tile_size / 1e6, 1),
    }


def suggested_tile_size(image_size: int, target_pixels: int = 512 * 512) -> int:
    """A tile size giving roughly ``target_pixels`` of work per tile."""

    return max(128, int(math.sqrt(target_pixels) // 8 * 8))
