"""Tests for aspect-ratio correctness at inference time.

Regression targets:

* ``ImageOps.fit`` centre-cropping then resizing back, which both loses content
  and stretches the result on non-square frames
* an elliptical vignette / skewed light leak caused by normalising each axis to
  [-1, 1] independently
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from nightflash.effects import _leak_basis, _vignette
from nightflash.geometry import fit_transform, prepare, restore

SIZES = [(600, 800), (800, 600), (512, 512), (1000, 300), (300, 1000)]


def _grid_image(width: int, height: int) -> Image.Image:
    """Image with a regular grid, so any geometric distortion is measurable."""

    image = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    step_x = max(1, width // 12)
    step_y = max(1, height // 16)
    for x in range(0, width, step_x):
        draw.line([(x, 0), (x, height)], fill=(90, 90, 90))
    for y in range(0, height, step_y):
        draw.line([(0, y), (width, y)], fill=(90, 90, 90))
    return image


@pytest.mark.parametrize("width,height", SIZES)
@pytest.mark.parametrize("fit", ["letterbox", "native"])
def test_lossless_fit_modes_preserve_geometry(width: int, height: int, fit: str) -> None:
    """Letterbox/native must round-trip to the original size with no distortion."""

    source = _grid_image(width, height)
    array, transform = prepare(source, 256, fit=fit)
    result = restore(array, transform)

    assert result.size == (width, height)
    a = np.asarray(source.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
    b = np.asarray(result.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
    mae = float(np.abs(a - b).mean())
    # Lossless modes differ only by resampling error; crop was ~3.2/255.
    assert mae < 1.0, f"{fit} distorted geometry on {width}x{height}: MAE {mae:.2f}"


def test_crop_mode_is_lossy_and_documented_as_such() -> None:
    """The legacy mode loses edges; this test pins the behaviour we replaced."""

    source = _grid_image(600, 800)
    array, transform = prepare(source, 256, fit="crop")
    result = restore(array, transform)
    assert result.size == (600, 800)  # size is restored ...
    a = np.asarray(source.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
    b = np.asarray(result.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
    assert float(np.abs(a - b).mean()) > 1.5  # ... but content is not


@pytest.mark.parametrize("width,height", [(600, 800), (800, 600)])
def test_no_content_is_lost_outside_the_frame(width: int, height: int) -> None:
    """Letterbox padding must not bleed into the result."""

    source = _grid_image(width, height)
    array, transform = prepare(source, 256, fit="letterbox")
    left, top, right, bottom = transform.content_box
    assert left >= 0 and top >= 0
    assert right <= transform.target_size[0] and bottom <= transform.target_size[1]
    result = restore(array, transform)
    assert result.size == (width, height)


def test_padding_uses_the_requested_mode() -> None:
    source = Image.new("RGB", (600, 800), (10, 20, 30))
    array, transform = prepare(source, 512, fit="letterbox", pad_mode="black")
    assert (array[:, 0, 0] == 0.0).all()  # top-left corner is padding


def test_fit_transform_matches_prepare() -> None:
    for width, height in SIZES:
        source = _grid_image(width, height)
        _, transform = prepare(source, 256, fit="letterbox")
        described = fit_transform((width, height), 256, "letterbox")
        assert described.target_size == transform.target_size
        assert described.content_box == transform.content_box
        assert described.scale == pytest.approx(transform.scale)


def test_model_input_sizes_are_valid() -> None:
    """Padded dimensions must stay positive and a multiple of 8 in native mode."""

    for width, height in SIZES:
        described = fit_transform((width, height), 256, "native")
        w, h = described.target_size
        assert w % 8 == 0 and h % 8 == 0
        assert w > 0 and h > 0


# --------------------------------------------------------------------------
# effects: vignette and leak must be isotropic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("height,width", [(257, 257), (193, 257), (257, 193)])
def test_vignette_is_circular(height: int, width: int) -> None:
    """Equal pixel distance from centre must give equal falloff."""

    vign = _vignette(height, width, torch.device("cpu"), torch.float32)[0, 0]
    cy, cx = height // 2, width // 2
    d = 30
    samples = [
        vign[cy - d, cx].item(),
        vign[cy + d, cx].item(),
        vign[cy, cx - d].item(),
        vign[cy, cx + d].item(),
    ]
    assert max(samples) - min(samples) < 1e-5, f"elliptical vignette on {height}x{width}: {samples}"


def test_vignette_reaches_one_at_the_corner() -> None:
    for height, width in [(257, 257), (193, 257), (257, 193)]:
        vign = _vignette(height, width, torch.device("cpu"), torch.float32)[0, 0]
        assert vign[0, 0].item() == pytest.approx(1.0, abs=0.01)
        assert vign[height // 2, width // 2].item() == pytest.approx(0.0, abs=0.01)


def _probe(basis: torch.Tensor, u: float, v: float) -> float:
    """Evaluate a basis at a given isotropic offset from the frame centre."""

    h, w = basis.shape
    xs = torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0
    ys = torch.arange(h, dtype=torch.float32) - (h - 1) / 2.0
    denominator = max(1.0, min(h, w) / 2.0)
    col = int((xs / denominator - u).abs().argmin())
    row = int((ys / denominator - v).abs().argmin())
    return float(basis[row, col])


PROBES = [(0.5, 0.0), (0.5, 0.3), (-0.5, 0.2), (0.8, -0.4), (0.0, 0.7), (-0.9, -0.6)]


@pytest.mark.parametrize("height,width", [(193, 257), (257, 193), (193, 385), (385, 193)])
def test_leak_basis_is_a_function_of_isotropic_coordinates(height: int, width: int) -> None:
    """The leak must depend on *physical* position, not on normalised coordinates.

    The naive implementation normalises each axis independently to [-1, 1], so the
    pattern is "consistent" in normalised space but changes shape in pixels. The
    correct implementation shares one scale across both axes, so the value at a
    given physical offset is identical regardless of frame shape.
    """

    reference = _leak_basis(257, 257, torch.device("cpu"), torch.float32)[0, 0]
    candidate = _leak_basis(height, width, torch.device("cpu"), torch.float32)[0, 0]

    for u, v in PROBES:
        expected = _probe(reference, u, v)
        actual = _probe(candidate, u, v)
        assert actual == pytest.approx(expected, abs=0.02), (
            f"leak value at isotropic ({u}, {v}) differs on {height}x{width}: "
            f"{actual:.4f} vs {expected:.4f} on the square reference"
        )


def test_naive_formula_would_fail_this_test() -> None:
    """Pins *why* the fix exists: per-axis [-1,1] is not physically consistent."""

    def naive(height: int, width: int) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height)
        xs = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        diagonal = torch.sigmoid((xx + 0.65 * yy + 0.35) * 5.0)
        edge = torch.sigmoid((xx - 0.25) * 6.0) * torch.sigmoid((0.85 - yy) * 4.0)
        return (0.65 * diagonal + 0.35 * edge).clamp(0.0, 1.0)

    reference = naive(257, 257)
    candidate = naive(193, 257)
    worst = 0.0
    for u, v in PROBES:
        worst = max(worst, abs(_probe(candidate, u, v) - _probe(reference, u, v)))
    # The naive formula is off by a large margin at the same physical points.
    assert worst > 0.05, f"naive formula unexpectedly agreed (max delta {worst:.4f})"
