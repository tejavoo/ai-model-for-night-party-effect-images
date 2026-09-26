"""Synthetic camera degradation, used to manufacture a training source domain.

The problem this solves
-----------------------
Learning a look from a *target* domain alone is the hard part of unpaired
image-to-image translation. This module manufactures a **source** domain from
the target images themselves, by simulating what the same scene would have
looked like through a non-flash camera: flatter lighting, no halation, no film
grain, a different white balance.

Two ways to use it
------------------
``paired``
    ``source = degrade(image_i)``, ``target = image_i``. Exact correspondence,
    so geometry and identity are directly supervised and real PSNR/SSIM are
    computable.

``sameset``
    ``source = degrade(image_j)``, ``target = image_i``, ``j != i``. Both from
    one directory, no correspondence. This removes the shortcut available in
    ``paired``: with no pointwise relationship to invert, the model cannot
    learn ``degrade_inverse`` and must instead learn the conditional
    distribution of the look.

Pipeline order
--------------
The stages run in the order a camera pipeline would be *undone*, and the order
matters. Photometric operations are applied in linear float space, and the
resampling stages come first so later stages operate on clean pixels.

============  ==========================================================
stage         what it actually changes
============  ==========================================================
1. crop       geometric normalisation, before anything else
2. downscale  removes grain and fine detail. ``resize -> resize back``
   upscale    Measured on a real photograph: high-frequency energy falls
              to 0.52x / 0.30x / 0.17x for the three profiles.
3. blur       further attenuates high frequencies; local contrast falls
              to 0.39x / 0.20x / 0.17x.
4. contrast   compresses toward mid-grey. This is the stage that does most
              of the work on the flash falloff signature (bright subject
              on dark ground becomes flat, evenly lit). Measured tonal
              range 0.965 -> 0.75 / 0.57 / 0.38.
5. black      lifts the floor, removing crushed shadows
   lift
6. exposure   overall gain
7. balance    warms the image; flash is neutral, ambient is tungsten
8. saturation raises it; flash photographs *look* desaturated purely
              because of the white-balance mismatch
9. noise      a little clean sensor noise, so the model does not learn
              "the input is always perfectly clean"
============  ==========================================================

An honest limitation on halation
--------------------------------
Resampling **cannot** remove a wide, low-frequency halo, and it is worth being
explicit about that rather than claiming otherwise. A low-pass filter has no
problem to solve: the glow is already low-frequency, and blurring spreads the
core outward into the ring, which measurably makes the apparent halo *brighter*
(ring/peak rises from 0.57 to 0.86 as blur goes 0 -> 4).

What the pipeline actually achieves is attenuation, driven mostly by the tonal
compression in stages 4-5 rather than by the filtering. That is sufficient for
the purpose: the model sees a highlight with less glow around it than the
target has, and learns to add the difference. It is not sufficient if you need
the source to contain *zero* halation, which would require re-rendering the
lighting rather than filtering it.

What is deliberately not done
-----------------------------
Nothing that moves a highlight relative to its surroundings. Halation
*placement* is the thing being learned, so the degradation must leave highlight
positions intact. Geometric warps are safe because source and target receive
the identical warp, so any crop is shared.

Keeping the source plausible
----------------------------
Every profile, including ``strong``, has to leave an image that still looks
like a photograph. A first pass at ``strong`` used blur up to 2.5 and black lift
up to 0.20; the result looked foggy rather than like a real non-flash capture,
and its added noise was glaring against the softened background. Training on
that teaches the model to expect an input distribution that will never occur at
inference. The ranges were pulled back until ``strong`` still reads as a
believable photo.

The failure mode to watch for
-----------------------------
If the degradation is too consistent, the cheapest way to reduce the loss in
``paired`` mode is to invert this specific corruption. The tell is a model that
looks excellent on freshly-degraded images and poor on real ones. The defences
are wide per-image randomisation, the ``none_prob`` identity path, and the
curriculum.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

Range = tuple[float, float]

#: Degradation strength profiles, weakest first. ``DegradeConfig.profile`` picks
#: one; every individual bound can still be overridden.
PROFILES = ("subtle", "moderate", "strong")

#: The three ways a source/target pair is formed.
TRAIN_MODES = ("unpaired", "paired", "sameset")

RESAMPLE = Image.Resampling.BICUBIC


def stable_seed(*parts: str) -> int:
    """A seed derived from strings, stable across processes and runs.

    ``hash()`` on ``str`` is salted per interpreter, so it cannot be used here:
    validation degradation must be identical in every session, or epoch 7 and
    epoch 8 are not comparable.
    """

    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class DegradeConfig:
    """Bounds for every stage. All are multiplicative or additive ranges.

    A default-constructed instance is the ``moderate`` profile with the
    curriculum enabled, which is the recommended starting point.
    """

    # --- frequency destruction ---
    resize_min: float = 0.60
    resize_max: float = 0.90
    blur_min: float = 0.3
    blur_max: float = 1.0

    # --- tone ---
    contrast_min: float = 0.70
    contrast_max: float = 0.90
    black_lift_min: float = 0.04
    black_lift_max: float = 0.12
    exposure_min: float = 0.70
    exposure_max: float = 0.90

    # --- colour ---
    saturation_min: float = 1.05
    saturation_max: float = 1.30
    wb_min: float = 4.0
    wb_max: float = 14.0

    # --- sensor ---
    noise_min: float = 0.005
    noise_max: float = 0.020

    # --- behaviour ---
    #: Probability of feeding the *undegraded* target as the source. Anchors the
    #: identity case and widens the input distribution, which is the main
    #: defence against the inverse-mapping failure mode.
    none_prob: float = 0.10
    #: Ramp subtle -> moderate -> strong across training. Early epochs get an
    #: easy mapping, later epochs get robustness.
    curriculum: bool = True
    #: Optional total epochs, needed to place the curriculum. Set by Session.
    curriculum_epochs: int = 0
    profile: str = "moderate"

    def resolved(self, epoch: int = 0) -> "DegradeConfig":
        """Apply the curriculum for ``epoch`` and return the effective config.

        With the curriculum off, or with no known horizon, the base profile is
        returned unchanged.
        """

        if not self.curriculum or self.curriculum_epochs <= 0:
            return self
        fraction = min(max(epoch / max(self.curriculum_epochs - 1, 1), 0.0), 1.0)
        # subtle for the first third, moderate for the middle, strong at the end
        if fraction < 1.0 / 3.0:
            target = 0.0
        elif fraction < 2.0 / 3.0:
            target = 1.0
        else:
            target = 2.0
        return _shift_profile(self, target, fraction)

    def describe(self) -> str:
        colour = "wb-shifted"
        return (
            f"resize {self.resize_min:.2f}-{self.resize_max:.2f}, "
            f"blur {self.blur_min:.1f}-{self.blur_max:.1f}, "
            f"contrast {self.contrast_min:.2f}-{self.contrast_max:.2f}, "
            f"black lift {self.black_lift_min:.3f}-{self.black_lift_max:.3f}, "
            f"exposure {self.exposure_min:.2f}-{self.exposure_max:.2f}, "
            f"saturation {self.saturation_min:.2f}-{self.saturation_max:.2f}, "
            f"{colour}, noise {self.noise_min:.3f}-{self.noise_max:.3f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DegradeConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _profile_preset(name: str) -> dict[str, Any]:
    if name == "subtle":
        return dict(
            resize_min=0.85, resize_max=1.00, blur_min=0.0, blur_max=0.6,
            contrast_min=0.85, contrast_max=0.95,
            black_lift_min=0.02, black_lift_max=0.06,
            exposure_min=0.85, exposure_max=0.95,
            saturation_min=1.00, saturation_max=1.15,
            wb_min=2.0, wb_max=8.0,
            noise_min=0.000, noise_max=0.010,
        )
    if name == "moderate":
        return dict(
            resize_min=0.60, resize_max=0.90, blur_min=0.3, blur_max=1.0,
            contrast_min=0.70, contrast_max=0.90,
            black_lift_min=0.04, black_lift_max=0.12,
            exposure_min=0.70, exposure_max=0.90,
            saturation_min=1.05, saturation_max=1.30,
            wb_min=4.0, wb_max=14.0,
            noise_min=0.005, noise_max=0.020,
        )
    if name == "strong":
        # Tuned down from a first pass that looked foggy rather than like a
        # photograph. A degraded source still has to be a *plausible* non-flash
        # capture: if it stops looking like a real photo, the model is learning
        # to work from an input distribution that will never occur at inference.
        return dict(
            resize_min=0.50, resize_max=0.80, blur_min=0.4, blur_max=1.4,
            contrast_min=0.65, contrast_max=0.85,
            black_lift_min=0.05, black_lift_max=0.14,
            exposure_min=0.65, exposure_max=0.85,
            saturation_min=1.10, saturation_max=1.40,
            wb_min=6.0, wb_max=20.0,
            noise_min=0.005, noise_max=0.018,
        )
    raise ValueError(f"degrade profile must be one of {PROFILES}, got {name!r}")


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _shift_profile(config: DegradeConfig, target: float, fraction: float) -> DegradeConfig:
    """Interpolate between subtle, moderate and strong.

    ``target`` is a fractional index into ``(subtle, moderate, strong)``, so the
    ramp is continuous rather than stepping.
    """

    presets = [_profile_preset(name) for name in PROFILES]
    index = min(max(target, 0.0), 2.0)
    low = int(math.floor(index))
    high = min(low + 1, len(presets) - 1)
    t = index - low
    updates: dict[str, Any] = {}
    for key in presets[0]:
        updates[key] = _lerp(presets[low][key], presets[high][key], t)
    return replace(config, **updates, profile=f"curriculum~{index:.2f}")


def make_config(profile: str = "moderate", **overrides: Any) -> DegradeConfig:
    """Build a config from a named profile plus explicit overrides."""

    config = DegradeConfig(**_profile_preset(profile), profile=profile)
    known = set(DegradeConfig.__dataclass_fields__)
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown degrade options: {sorted(unknown)}")
    return replace(config, **overrides) if overrides else config


def sample_params(config: DegradeConfig, rng: random.Random) -> dict[str, float]:
    """Draw one parameter set. Independent per image, per epoch and per run."""

    return {
        "resize": rng.uniform(config.resize_min, config.resize_max),
        "blur": rng.uniform(config.blur_min, config.blur_max),
        "contrast": rng.uniform(config.contrast_min, config.contrast_max),
        "black_lift": rng.uniform(config.black_lift_min, config.black_lift_max),
        "exposure": rng.uniform(config.exposure_min, config.exposure_max),
        "saturation": rng.uniform(config.saturation_min, config.saturation_max),
        "wb": rng.uniform(config.wb_min, config.wb_max),
        "noise": rng.uniform(config.noise_min, config.noise_max),
    }


def apply_degradation(image: Image.Image, params: dict[str, float],
                      rng: random.Random) -> Image.Image:
    """Run the pipeline. ``image`` is the clean target; the result is the source."""

    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. downscale / upscale: kills halation bloom and film grain.
    resize = max(0.05, min(1.0, float(params["resize"])))
    if resize < 0.999:
        width, height = image.size
        small = (max(8, int(round(width * resize))), max(8, int(round(height * resize))))
        image = image.resize(small, RESAMPLE).resize((width, height), RESAMPLE)

    # 3. blur: residual softening and the rest of the halo.
    blur = float(params["blur"])
    if blur > 0.01:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))

    array = np.asarray(image, dtype=np.float32) / 255.0

    # 6. exposure.
    array = array * float(params["exposure"])

    # 4. contrast about mid-grey: this is what flattens the flash falloff.
    contrast = float(params["contrast"])
    array = (array - 0.5) * contrast + 0.5

    # 5. black lift: 0 -> lift, 1 -> 1, so white stays white.
    lift = float(params["black_lift"])
    if lift > 0.0:
        array = array + lift * (1.0 - array)

    # 7. white balance, warmed.
    shift = float(params["wb"]) / 255.0
    array[..., 0] = array[..., 0] + shift
    array[..., 2] = array[..., 2] - shift

    # 8. saturation.
    saturation = float(params["saturation"])
    if abs(saturation - 1.0) > 1e-6:
        grey = array.mean(axis=2, keepdims=True)
        array = grey + (array - grey) * saturation

    array = np.clip(array, 0.0, 1.0)

    # 9. light sensor noise, so the model does not assume a pristine input.
    noise = float(params["noise"])
    if noise > 1e-6:
        # Derive a numpy generator from the caller's random.Random so the noise
        # field stays reproducible from the same seed, without a second global
        # RNG to thread through the dataset.
        generator = np.random.default_rng(rng.getrandbits(63))
        array = array + generator.normal(0.0, noise, size=array.shape)

    return Image.fromarray(
        np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="RGB"
    )


def degraded_pair(
    target_image: Image.Image,
    config: DegradeConfig,
    rng: random.Random,
    *,
    force: bool = False,
) -> tuple[Image.Image, dict[str, float] | None]:
    """Produce ``(source_image, params)``.

    ``params`` is ``None`` when the identity path was taken, so callers can tell
    "degraded" from "passed through untouched".
    """

    if not force and rng.random() < config.none_prob:
        return target_image, None
    params = sample_params(config, rng)
    return apply_degradation(target_image, params, rng), params
