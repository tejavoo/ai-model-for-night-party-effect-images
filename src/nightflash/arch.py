"""Architecture registry.

``edgehat`` is the current default. ``hybrid`` and ``restormer`` remain so that
checkpoints trained with the earlier code keep loading and remain comparable.
"""

from __future__ import annotations

from torch import nn

from .config import ModelConfig
from .edgehat import EdgeHatGenerator
from .heads import ALL_HEADS, COMPOSITE_HEADS, SIMPLE_HEADS
from .models import HybridGenerator
from .swinir import SwinIRGenerator

ARCHITECTURES = ("edgehat", "swinir", "hybrid", "restormer")
CAPABILITIES = ("halation", "subject-gate", "region-exposure")
# `HEADS` is the public list: simple heads plus the composite ones.
HEADS = ALL_HEADS

ARCH_DESCRIPTIONS = {
    "edgehat": (
        "Conv trunk 48/96/192/384 with EdgeViT separable local attention "
        "(L3 x2, L4 x1) plus one HAT overlapping-cross-attention block for "
        "global lighting context."
    ),
    "swinir": (
        "Conv trunk with windowed self-attention and relative position bias "
        "(alternating shifted windows). Cleaner Core ML export, weaker "
        "cross-window communication."
    ),
    "hybrid": (
        "Original generator: conv encoder/decoder with a single pooled "
        "transformer bottleneck. Retained for checkpoint compatibility."
    ),
    "restormer": (
        "Transposed attention over channels rather than pixels. Cheapest "
        "option; attention matrix is C x C so cost is resolution independent."
    ),
}

HEAD_DESCRIPTIONS = {
    "tanh": "Bounded delta (max +/- residual_scale). Safest with little data.",
    "perpixel": (
        "Unbounded per-pixel RGB delta, clamped on read. Free to repaint "
        "pixels rather than only shifting colour."
    ),
    "freqsplit": (
        "Low / mid / high frequency bands with separate pathways. Routes grain "
        "and halation into the high band so they stop competing with tone."
    ),
    "lut3d": (
        "Learned 33^3 trilinear colour LUT plus a per-pixel delta. Cheap global "
        "grade that generalises well from a small target set."
    ),
    "lut3d+freqsplit": (
        "COMPOSITE. Global colour LUT for overall grade, plus low/mid/high band "
        "detail. The most useful combination: the two parts are complementary."
    ),
    "lut3d+perpixel": (
        "COMPOSITE. Global colour LUT plus an unconstrained per-pixel delta. "
        "Weaker than lut3d+freqsplit; both parts are per-pixel."
    ),
    "tanh+freqsplit": (
        "COMPOSITE. Band-separated detail, but every contribution is bounded. "
        "The most conservative composite."
    ),
}


def run_generator(
    model: nn.Module, x: "torch.Tensor", mask: "torch.Tensor | None" = None
) -> tuple["torch.Tensor", "torch.Tensor", dict]:
    """Forward a generator and normalise its return signature.

    The newer architectures (``edgehat``, ``swinir``) return
    ``(residual, effect_params, extras)``. The original ``hybrid`` generator
    returns ``(residual, effect_params)`` and does not accept a mask argument.
    Callers use this helper so they never have to branch on architecture.
    """

    import torch  # local import keeps this module import-light

    config = getattr(model, "config", None)
    expected = int(getattr(config, "in_channels", 3))
    if x.shape[1] == expected:
        # Caller already supplied RGB (or RGB+mask). Do not concatenate again.
        out = model(x)
    elif mask is not None and x.shape[1] == 3:
        out = model(torch.cat([x, mask], dim=1))
    else:
        raise ValueError(
            f"expected {expected} input channels, got {x.shape[1]}"
            + (" (and no mask to append)" if mask is None else "")
        )

    if isinstance(out, tuple) and len(out) == 3:
        residual, effects, extras = out
        return residual, effects, extras
    residual, effects = out[0], out[1]
    return residual, effects, {"subject": None, "region": None, "halation": None}


def build_generator(config: ModelConfig) -> nn.Module:
    """Instantiate the generator named by ``config.arch``."""

    if config.arch == "edgehat":
        return EdgeHatGenerator(config)
    if config.arch == "swinir":
        return SwinIRGenerator(config)
    if config.arch in {"hybrid", "restormer"}:
        return HybridGenerator(config)
    raise ValueError(f"unknown arch {config.arch!r}; choose from {ARCHITECTURES}")


def new_arch_config(arch: str, **overrides) -> ModelConfig:
    """A sensible default config for a named architecture."""

    if arch == "edgehat":
        defaults = dict(
            arch="edgehat",
            head="freqsplit",
            base_channels=48,
            bottleneck_channels=384,
            edgevit_blocks_l3=2,
            edgevit_blocks_l4=1,
            hat_blocks_l4=1,
            attention_heads=4,
            swin_window=8,
            use_halation=True,
            use_subject_gate=True,
            use_region_exposure=True,
            region_grid=4,
            effect_dim=8,
        )
    elif arch == "swinir":
        defaults = dict(
            arch="swinir",
            head="perpixel",
            base_channels=48,
            bottleneck_channels=384,
            swin_blocks_l3=2,
            swin_blocks_l4=2,
            attention_heads=4,
            swin_window=8,
            effect_dim=8,
        )
    elif arch in {"hybrid", "restormer"}:
        defaults = dict(arch=arch, head="tanh")
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHITECTURES}")
    defaults.update(overrides)
    return ModelConfig(**defaults)
