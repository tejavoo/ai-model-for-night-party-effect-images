"""Aspect-ratio-correct image geometry for inference.

The training pipeline sees square ``N x N`` crops. Naively running that model on a
600x800 photo does one of two things, both wrong:

``ImageOps.fit(img, (512, 512))``
    Centre-crops. A 600x800 frame loses 25% of its height, then the square result
    is resized back to 600x800, which **stretches** it. Content is lost *and* the
    geometry is distorted.

Running the model on the raw rectangle
    Preserves content and aspect, but shifts the global context: the network was
    trained where a 128px receptive field covers 25% of the frame, and at 800px
    wide it covers only 16%.

The fix is to keep the frame square for the network while guaranteeing that every
input pixel maps back to the same place in the output:

* ``letterbox`` (default) scales the image to fit inside ``size x size`` and pads
  the remainder. No crop, no stretch, and the network still sees the square
  distribution it was trained on.
* ``native`` scales the short edge to ``size``, pads to a multiple of 8, and runs
  at the true aspect ratio. Preserves context better but shifts scale.
* ``crop`` is the old lossy behaviour, kept only for reproducing old results.

Every mode returns the parameters needed to invert the transform exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

FitMode = Literal["letterbox", "native", "crop"]
PadMode = Literal["edge", "reflect", "black"]

PAD_MULTIPLE = 8  # three stride-2 stages; keeps things tidy although the model
#                 would tolerate any size thanks to dynamic interpolation


@dataclass(frozen=True)
class LetterboxTransform:
    """Describes a scale + pad, and how to undo it."""

    original_size: tuple[int, int]  # (width, height) of the source image
    target_size: tuple[int, int]  # (width, height) fed to the model
    scale: float  # source pixels -> model pixels
    pad_left: int
    pad_top: int
    content_box: tuple[int, int, int, int]  # (left, top, right, bottom) in model space

    def model_to_source_box(self) -> tuple[float, float, float, float]:
        """The region of the *source* image that is visible, in source pixels."""

        left, top, right, bottom = self.content_box
        return (
            (left - self.pad_left) / self.scale,
            (top - self.pad_top) / self.scale,
            (right - self.pad_left) / self.scale,
            (bottom - self.pad_top) / self.scale,
        )


def _pad_to(
    array: np.ndarray,
    target_width: int,
    target_height: int,
    pad_left: int,
    pad_top: int,
    mode: PadMode,
) -> np.ndarray:
    pad_right = target_width - array.shape[1] - pad_left
    pad_bottom = target_height - array.shape[0] - pad_top
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError("padding would overlap the content; reduce --image-size")
    if pad_left == 0 and pad_top == 0 and pad_right == 0 and pad_bottom == 0:
        return array
    if mode == "black":
        return np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )
    numpy_mode = "reflect" if mode == "reflect" else "edge"
    if array.shape[0] <= pad_top or array.shape[0] <= pad_bottom:
        numpy_mode = "edge"
    if array.shape[1] <= pad_left or array.shape[1] <= pad_right:
        numpy_mode = "edge"
    return np.pad(
        array,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode=numpy_mode,
    )


def prepare(
    image: Image.Image,
    size: int,
    *,
    fit: FitMode = "letterbox",
    pad_mode: PadMode = "edge",
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize/pad ``image`` into a model-ready float array in ``[0, 1]``.

    Returns the array in ``CHW`` order and the transform describing how to undo
    the scale and pad.
    """

    source_width, source_height = image.size
    if source_width < 1 or source_height < 1:
        raise ValueError(f"degenerate image size {image.size}")

    if fit == "crop":
        # Legacy behaviour, kept for reproducibility. Centre-crop then resize.
        scale_side = min(source_width, source_height)
        left = (source_width - scale_side) // 2
        top = (source_height - scale_side) // 2
        crop = image.crop((left, top, left + scale_side, top + scale_side))
        resized = crop.resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        transform = LetterboxTransform(
            original_size=(source_width, source_height),
            target_size=(size, size),
            scale=size / scale_side,
            pad_left=0,
            pad_top=0,
            content_box=(0, 0, size, size),
        )
        return array.transpose(2, 0, 1), transform

    if fit == "letterbox":
        # Scale so the whole frame fits inside size x size.
        scale = min(size / source_width, size / source_height)
        model_width = max(1, int(round(source_width * scale)))
        model_height = max(1, int(round(source_height * scale)))
        target_width = target_height = size
    elif fit == "native":
        # Scale so the short edge equals `size`, keep the true aspect, then round
        # up to a multiple of 8 so the stride-2 encoder stays well behaved.
        scale = size / min(source_width, source_height)
        model_width = int(round(source_width * scale))
        model_height = int(round(source_height * scale))
        target_width = _round_up(model_width, PAD_MULTIPLE)
        target_height = _round_up(model_height, PAD_MULTIPLE)
    else:
        raise ValueError(f"unknown fit mode: {fit}")

    resized = image.resize((model_width, model_height), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0

    pad_left = (target_width - model_width) // 2
    pad_top = (target_height - model_height) // 2
    # Centre the remainder when it is odd, so the frame stays balanced.
    if (target_width - model_width) % 2:
        pad_left += 1
    if (target_height - model_height) % 2:
        pad_top += 1

    array = _pad_to(array, target_width, target_height, pad_left, pad_top, pad_mode)

    transform = LetterboxTransform(
        original_size=(source_width, source_height),
        target_size=(target_width, target_height),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        content_box=(
            pad_left,
            pad_top,
            pad_left + model_width,
            pad_top + model_height,
        ),
    )
    return array.transpose(2, 0, 1), transform


def _round_up(value: int, multiple: int) -> int:
    return int(-(-value // multiple) * multiple)


def restore(
    tensor_chw: np.ndarray,
    transform: LetterboxTransform,
    *,
    out_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Undo ``prepare``: crop away the padding, then resize to the source size.

    Because the content box is cropped before resizing, no padding bleeds into the
    result and the output is not stretched.
    """

    left, top, right, bottom = transform.content_box
    content = tensor_chw[:, top:bottom, left:right]
    height = max(1, content.shape[1])
    width = max(1, content.shape[2])

    channels_first = content.transpose(1, 2, 0)  # HWC
    image = Image.fromarray(
        np.uint8(np.clip(channels_first, 0.0, 1.0) * 255.0), mode="RGB"
    )
    target = out_size if out_size is not None else transform.original_size
    if image.size != target:
        image = image.resize(target, Image.Resampling.LANCZOS)
    return image


def fit_transform(
    image_size: tuple[int, int], size: int, fit: FitMode
) -> LetterboxTransform:
    """Describe the transform without touching pixels. Useful for reporting."""

    width, height = image_size
    if fit == "crop":
        side = min(width, height)
        return LetterboxTransform(
            original_size=(width, height),
            target_size=(size, size),
            scale=size / side,
            pad_left=0,
            pad_top=0,
            content_box=(0, 0, size, size),
        )
    if fit == "letterbox":
        scale = min(size / width, size / height)
        model_width = max(1, int(round(width * scale)))
        model_height = max(1, int(round(height * scale)))
        target_width = target_height = size
    else:
        scale = size / min(width, height)
        model_width = int(round(width * scale))
        model_height = int(round(height * scale))
        target_width = _round_up(model_width, PAD_MULTIPLE)
        target_height = _round_up(model_height, PAD_MULTIPLE)
    pad_left = (target_width - model_width) // 2
    pad_top = (target_height - model_height) // 2
    if (target_width - model_width) % 2:
        pad_left += 1
    if (target_height - model_height) % 2:
        pad_top += 1
    return LetterboxTransform(
        original_size=(width, height),
        target_size=(target_width, target_height),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        content_box=(pad_left, pad_top, pad_left + model_width, pad_top + model_height),
    )
