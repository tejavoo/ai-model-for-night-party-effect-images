"""Training-time augmentation, tuned for a photographic style transfer.

The current loader only did: scale to the short side, random crop, horizontal
flip. That is enough to stop the model memorising exact crops, but it leaves the
training distribution much narrower than the scenes a camera app actually sees.

What this adds, and why each one earns its place here
-------------------------------------------------------
``scale jitter`` (0.8-1.25)
    Varies how much of the scene fits in the crop. A 256px crop of a 12MP night
    photo is a close-up; the same crop of a 3MP wide shot is a whole room. Without
    jitter the model silently learns only the first case.

``rotation`` (±12 degrees, fill mirrored)
    A camera is rarely level. Small rotations stop the model from learning that a
    horizon is always horizontal. Kept small: beyond about 15 degrees you are
    teaching the model to accept tilted *content*, which is not the goal.

``horizontal and vertical flip``
    Doubles the effective dataset. Vertical flip is safe here because neither
    domain has an inherent up direction. Diagonal flip is deliberately excluded
    because it would mirror real text and faces.

``colour jitter`` — OFF by default
    Brightness / contrast / saturation / temperature noise.
    **This is a trap for style transfer.** Your target domain *is* a colour
    look. Perturbing target colour teaches the discriminator that the style is
    fuzzy, and it can wash out exactly the grade you are trying to learn. It is
    available because it is sometimes useful, but it should be off unless you
    have already memorised the target set.

``holes`` (cutout)
    Small random rectangles set to the dataset mean. Cutout is standard for
    classification because it forces context use. For *appearance* transfer it is
    less useful and can teach the model to hallucinate texture over occluders,
    which is a bad trait for a camera filter. Small and rare by default.

Consistency rules that matter more than the individual knobs
-----------------------------------------------------------
* Geometry is applied **identically to source and target** in every batch, so the
  pair still describes the same crop position. With unpaired sampling the two
  images come from different files, but both must be warped the same way or the
  discriminator sees two different scenes and the content losses pull in
  contradictory directions.
* **Colour jitter is never applied to the target.** The target carries the style.
* The same geometric parameters are used for a source image and its mask, so the
  person mask stays aligned.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

RESAMPLE = Image.Resampling.BICUBIC
FILL = (114, 114, 114)


@dataclass
class AugmentConfig:
    """Bounds for every knob. All are opt-in except crop jitter and h-flip."""

    # --- geometry ---
    scale_min: float = 0.8
    scale_max: float = 1.25
    rotate_degrees: float = 12.0
    hflip: float = 0.5  # probability
    vflip: float = 0.5  # probability

    # --- photometric ---
    brightness: float = 0.0  # +/- fraction, 0 disables
    contrast: float = 0.0
    saturation: float = 0.0
    temperature: float = 0.0
    apply_colour_to_target: bool = False  # strongly discouraged

    # --- occlusion ---
    holes: int = 0  # number of rectangles, 0 disables
    hole_size: float = 0.12  # as a fraction of the crop side
    hole_count_prob: float = 0.0  # probability of applying holes at all

    def describe(self) -> str:
        parts = [
            f"scale {self.scale_min}-{self.scale_max}",
            f"rot +/-{self.rotate_degrees}deg",
            f"hflip {self.hflip}",
            f"vflip {self.vflip}",
        ]
        colour = max(self.brightness, self.contrast, self.saturation, self.temperature)
        parts.append(f"colour {colour}" if colour else "colour off")
        parts.append(f"holes {self.holes}" if self.holes else "holes off")
        return ", ".join(parts)

    @classmethod
    def conservative(cls) -> "AugmentConfig":
        return cls(scale_min=0.9, scale_max=1.15, rotate_degrees=6.0)

    @classmethod
    def strong(cls) -> "AugmentConfig":
        return cls(
            scale_min=0.7, scale_max=1.4, rotate_degrees=15.0,
            brightness=0.10, contrast=0.10, saturation=0.10, temperature=0.05,
            holes=2, hole_size=0.10, hole_count_prob=0.3,
        )


def _sample_geometry(config: AugmentConfig, rng: random.Random) -> dict:
    return {
        "scale": rng.uniform(config.scale_min, config.scale_max),
        "degrees": rng.uniform(-config.rotate_degrees, config.rotate_degrees),
        "hflip": rng.random() < config.hflip,
        "vflip": rng.random() < config.vflip,
    }


def resize_with_scale(
    image: Image.Image, size: int, scale: float = 1.0, resample: int = RESAMPLE
) -> tuple[Image.Image, int]:
    """Scale the short side to ``size * scale``, keeping aspect ratio."""

    width, height = image.size
    target = size * scale
    factor = target / min(width, height)
    new_w = max(size, int(round(width * factor)))
    new_h = max(size, int(round(height * factor)))
    return image.resize((new_w, new_h), resample), int(round(target))


def apply_geometry(
    image: Image.Image,
    size: int,
    params: dict,
    rng: random.Random,
    *,
    allow_partial_crop: bool = True,
    resample: int = RESAMPLE,
    fill: int | tuple[int, ...] = FILL,
) -> Image.Image:
    """Scale, crop, rotate and flip. Returns exactly ``size x size``.

    ``allow_partial_crop`` lets a scale jitter that undershoots still produce a
    full crop by resizing up at the end, instead of padding. That keeps every
    sample at native crop resolution, which matters when the same size is used
    for tiled inference.

    ``resample`` and ``fill`` must be passed as ``NEAREST``/``0`` for masks, so a
    rotation does not blur a binary person mask into a grey ramp.
    """

    scaled, target = resize_with_scale(image, size, params.get("scale", 1.0), resample)
    width, height = scaled.size

    if width < size or height < size:
        if allow_partial_crop:
            # Upscale just enough so the crop is always full resolution.
            factor = size / min(width, height)
            scaled = scaled.resize(
                (max(size, int(round(width * factor))), max(size, int(round(height * factor)))),
                resample,
            )
        else:
            # Pad with a neutral fill, matched across source and target.
            padded = Image.new(image.mode, (target, target), fill)
            padded.paste(scaled, ((target - width) // 2, (target - height) // 2))
            scaled = padded
        width, height = scaled.size

    left = rng.randint(0, width - size) if width > size else (width - size) // 2
    top = rng.randint(0, height - size) if height > size else (height - size) // 2
    out = scaled.crop((left, top, left + size, top + size))

    degrees = params.get("degrees", 0.0)
    if abs(degrees) > 1e-3:
        out = out.rotate(degrees, resample=resample, expand=False, fillcolor=fill)
    if params.get("hflip"):
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if params.get("vflip"):
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return out


def apply_colour(
    image: Image.Image, config: AugmentConfig, rng: random.Random
) -> Image.Image:
    """Brightness / contrast / saturation / temperature jitter.

    Off by default. See the module docstring for why this is a trap for style
    transfer.
    """

    if max(config.brightness, config.contrast, config.saturation, config.temperature) <= 0:
        return image

    array = np.asarray(image, dtype=np.float32) / 255.0
    if config.brightness:
        array = array * (1.0 + rng.uniform(-config.brightness, config.brightness))
    if config.contrast:
        factor = 1.0 + rng.uniform(-config.contrast, config.contrast)
        array = (array - 0.5) * factor + 0.5
    if config.temperature:
        shift = rng.uniform(-config.temperature, config.temperature)
        array = array.copy()
        array[..., 0] += shift
        array[..., 2] -= shift
    if config.saturation:
        factor = 1.0 + rng.uniform(-config.saturation, config.saturation)
        grey = array.mean(axis=2, keepdims=True)
        array = grey + (array - grey) * factor
    return Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="RGB")


def apply_holes(
    image: Image.Image, config: AugmentConfig, rng: random.Random
) -> Image.Image:
    """Cutout. Off by default; see the module docstring."""

    if config.holes <= 0 or rng.random() >= config.hole_count_prob:
        return image
    array = np.array(image)
    size = image.size[0]
    for _ in range(config.holes):
        span = max(1, int(size * config.hole_size))
        left = rng.randint(0, max(0, size - span))
        top = rng.randint(0, max(0, size - span))
        # Fill with the image's own mean, not black, so the model cannot learn
        # "black rectangle means damaged sensor".
        array[top : top + span, left : left + span] = array.reshape(-1, 3).mean(axis=0)
    return Image.fromarray(array, mode="RGB")


def make_pair(
    source: Image.Image,
    target: Image.Image,
    size: int,
    config: AugmentConfig,
    rng: random.Random,
    mask: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    """Apply one shared geometric warp to source, target and optional mask.

    Colour jitter is applied to the source only unless explicitly overridden,
    because the target carries the style signal.
    """

    params = _sample_geometry(config, rng)

    warped_source = apply_geometry(source, size, params, rng)
    warped_target = apply_geometry(target, size, params, rng)
    warped_mask = None
    if mask is not None:
        # Nearest-neighbour plus a 0 fill keeps a binary person mask binary and
        # stops a rotated corner from inventing a person.
        warped_mask = apply_geometry(
            mask.convert("L"), size, params, rng,
            resample=Image.Resampling.NEAREST, fill=0,
        )

    warped_source = apply_holes(warped_source, config, rng)
    warped_source = apply_colour(warped_source, config, rng)
    if config.apply_colour_to_target:
        warped_target = apply_colour(warped_target, config, rng)
    else:
        warped_target = apply_holes(warped_target, config, rng)

    return warped_source, warped_target, warped_mask
