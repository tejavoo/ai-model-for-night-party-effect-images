"""Tests for tiled inference and composite heads."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from nightflash.arch import HEADS, SIMPLE_HEADS, build_generator, new_arch_config, run_generator
from nightflash.heads import ALL_HEADS, CompositeHead, ResidualHead
from nightflash.tiles import (
    TileBox,
    apply_tiled,
    blend_window,
    estimate_tiles,
    extract_tile,
    tile_boxes,
    tile_count,
)

SHAPES = [(600, 800), (800, 600), (1600, 1200), (4032, 3024), (500, 400)]


# ------------------------------------------------------------------- tiles


@pytest.mark.parametrize("width,height", SHAPES)
def test_tiles_are_full_size(width: int, height: int) -> None:
    """Every box is exactly tile_size, even where it runs off the frame.

    Partial edge tiles would have a blend window approaching zero at the frame
    border, which is what turned corners black in the first implementation.
    """

    for size, overlap in ((256, 64), (512, 128)):
        for box in tile_boxes(width, height, size, overlap):
            assert box.width == size
            assert box.height == size


@pytest.mark.parametrize("width,height", SHAPES)
def test_tiles_cover_the_frame_with_no_gaps(width: int, height: int) -> None:
    for size, overlap in ((256, 64), (512, 128)):
        cover = np.zeros((height, width), np.int32)
        for box in tile_boxes(width, height, size, overlap):
            y0, y1 = max(0, box.top), min(height, box.bottom)
            x0, x1 = max(0, box.left), min(width, box.right)
            cover[y0:y1, x0:x1] += 1
        assert cover.min() > 0, "some pixel is not covered by any tile"
        assert cover.max() >= 1


def test_overlap_must_be_valid() -> None:
    with pytest.raises(ValueError, match="overlap"):
        tile_boxes(600, 800, 256, 256)
    with pytest.raises(ValueError, match="overlap"):
        tile_boxes(600, 800, 256, -1)
    with pytest.raises(ValueError, match="tile_size"):
        tile_boxes(600, 800, 4, 1)


def test_blend_window_has_a_floor() -> None:
    """A near-zero window makes sum(w*x)/sum(w) numerically fragile."""

    window = blend_window(TileBox(0, 0, 256, 256))
    assert window.min() >= 1e-3
    assert window.max() <= 1.0
    assert window[128, 128] > 0.99


def test_tiled_identity_preserves_corners_and_edges() -> None:
    """The regression: corners used to blend to near-black."""

    image = Image.new("RGB", (600, 800), (200, 120, 90))
    out = np.asarray(apply_tiled(image, lambda t: t, tile_size=256, overlap=64), float)
    for y, x in [(0, 0), (0, -1), (-1, 0), (-1, -1), (400, 300)]:
        assert tuple(int(v) for v in out[y, x]) == (200, 120, 90), f"pixel {y},{x} wrong"


def test_tiled_identity_has_no_seam_ridge() -> None:
    """A linear gradient must stay linear: a seam would introduce a step."""

    width, height = 900, 700
    ramp = np.zeros((height, width, 3), np.uint8)
    ramp[:, :, 0] = np.linspace(0, 255, width).astype(np.uint8)[None, :]
    image = Image.fromarray(ramp)
    for size, overlap in ((256, 64), (512, 128)):
        out = np.asarray(apply_tiled(image, lambda t: t, tile_size=size, overlap=overlap), float)
        error = np.abs(out - np.asarray(image, float)).mean()
        assert error < 0.5, f"seam detected at tile={size}: mean error {error:.3f}"


def test_tiled_identity_preserves_size() -> None:
    image = Image.new("RGB", (1600, 1200), (10, 20, 30))
    out = apply_tiled(image, lambda t: t, tile_size=512, overlap=128)
    assert out.size == image.size


def test_extract_tile_pads_and_reports_visible_region() -> None:
    image = Image.new("RGB", (600, 800), (5, 5, 5))
    box = tile_boxes(600, 800, 256, 64)[0]
    tile, visible = extract_tile(image, box)
    assert tile.size == (256, 256)
    assert visible.width <= 256 and visible.height <= 256


def test_extract_tile_covers_a_corner_box() -> None:
    image = Image.new("RGB", (300, 300), (1, 2, 3))
    box = tile_boxes(300, 300, 512, 128)[0]  # single box larger than the image
    tile, visible = extract_tile(image, box)
    assert tile.size == (512, 512)
    assert visible.width == 300 and visible.height == 300


def test_tile_processor_must_preserve_size() -> None:
    image = Image.new("RGB", (600, 800))
    with pytest.raises(ValueError, match="changed size"):
        apply_tiled(image, lambda t: t.resize((10, 10)), tile_size=256, overlap=64)


def test_estimate_tiles_reports_useful_numbers() -> None:
    info = estimate_tiles(4032, 3024, 512, 128)
    assert info["tiles"] == tile_count(4032, 3024, 512, 128)
    assert info["megapixels_processed"] > 0


def test_small_image_needs_one_tile() -> None:
    assert tile_count(300, 300, 512, 128) == 1


# ---------------------------------------------------------------- composite


def test_public_head_list_includes_composites() -> None:
    assert set(SIMPLE_HEADS) <= set(HEADS)
    assert "lut3d+freqsplit" in HEADS
    assert set(HEADS) == set(ALL_HEADS)


@pytest.mark.parametrize("head", HEADS)
def test_every_head_builds_and_is_bounded(head: str) -> None:
    config = new_arch_config("edgehat", head=head)
    out = ResidualHead(48, config)
    result = out(torch.rand(1, 48, 32, 32), torch.rand(1, 3, 32, 32))
    assert result.shape == (1, 3, 32, 32)
    assert float(result.abs().max()) <= config.residual_limit + 1e-6


def test_composite_head_is_a_composite() -> None:
    config = new_arch_config("edgehat", head="lut3d+freqsplit")
    head = ResidualHead(48, config)
    assert isinstance(head.impl, CompositeHead)
    assert len(head.impl.parts) == 2


def test_composite_head_needs_two_members() -> None:
    with pytest.raises(ValueError, match="at least two"):
        CompositeHead(("freqsplit",), 48, new_arch_config("edgehat"))


def test_unknown_head_is_rejected() -> None:
    from nightflash.config import ModelConfig

    with pytest.raises(ValueError, match="unknown head"):
        ResidualHead(48, ModelConfig(head="nonsense"))
    with pytest.raises(ValueError, match="unknown head"):
        ResidualHead(48, ModelConfig(head="freqsplit+banana"))


def test_lut3d_composite_requires_the_image() -> None:
    head = ResidualHead(48, new_arch_config("edgehat", head="lut3d+freqsplit"))
    assert head.needs_image is True
    with pytest.raises(ValueError, match="needs the source image"):
        head(torch.rand(1, 48, 16, 16))


def test_non_lut_head_does_not_need_the_image() -> None:
    head = ResidualHead(48, new_arch_config("edgehat", head="tanh+freqsplit"))
    assert head.needs_image is False


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("arch", ["edgehat", "swinir"])
def test_generator_trains_with_every_head(head: str, arch: str) -> None:
    config = new_arch_config(arch, head=head)
    model = build_generator(config)
    residual, effects, _ = run_generator(model, torch.rand(1, 3, 64, 64))
    (residual.mean() + effects.mean()).backward()
    assert any(p.grad is not None for p in model.parameters())
