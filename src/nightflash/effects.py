from __future__ import annotations

import math

import torch
from torch import Tensor


def _seeded_noise(shape: tuple[int, ...], seed: int | None, device: torch.device, dtype: torch.dtype) -> Tensor:
    if seed is None:
        return torch.randn(shape, device=device, dtype=dtype)

    # CPU generators are supported on every backend. Creating the tensor on CPU
    # first also keeps MPS/Core ML inference reproducible across devices.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    return torch.randn(shape, generator=generator, dtype=dtype).to(device)


def _leak_basis(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """A smooth, inexpensive colored light-leak basis.

    This is deliberately a procedural basis rather than a neural decoder. The
    model predicts how strongly each effect should be applied; the renderer
    decides exactly how the stochastic texture is generated.
    """

    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    diagonal = torch.sigmoid((xx + 0.65 * yy + 0.35) * 5.0)
    edge = torch.sigmoid((xx - 0.25) * 6.0) * torch.sigmoid((0.85 - yy) * 4.0)
    basis = 0.65 * diagonal + 0.35 * edge
    return basis.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


def apply_stochastic_effects(
    image: Tensor,
    effect_params: Tensor,
    *,
    seed: int | None = None,
    strength: float = 1.0,
) -> Tensor:
    """Apply small, seeded effects to a deterministic styled image.

    ``effect_params`` has eight sigmoid-valued channels:

    0. grain strength
    1. light-leak strength
    2. exposure offset
    3. saturation
    4. vignette strength
    5. leak red tint
    6. leak green tint
    7. leak blue tint

    The function is differentiable with respect to ``effect_params``. The noise
    realization itself is treated as a fixed sample during a training step.
    """

    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(image.shape)}")
    if effect_params.ndim != 2 or effect_params.shape[0] != image.shape[0]:
        raise ValueError(
            f"Expected effect parameters [B,{effect_params.shape[-1]}], "
            f"got {tuple(effect_params.shape)} for image {tuple(image.shape)}"
        )
    if effect_params.shape[1] < 8:
        raise ValueError("The stochastic renderer requires eight effect parameters")

    strength = max(0.0, float(strength))
    if strength == 0.0:
        return image

    b, _, height, width = image.shape
    device = image.device
    dtype = image.dtype
    p = effect_params[:, :8]

    grain_strength = p[:, 0].view(b, 1, 1, 1)
    leak_strength = p[:, 1].view(b, 1, 1, 1)
    exposure = (p[:, 2] - 0.5).view(b, 1, 1, 1) * (0.35 * strength)
    saturation = 0.86 + p[:, 3].view(b, 1, 1, 1) * (0.30 * strength)
    vignette_strength = p[:, 4].view(b, 1, 1, 1) * (0.22 * strength)

    styled = image * torch.pow(torch.tensor(2.0, device=device, dtype=dtype), exposure)
    gray = styled.mean(dim=1, keepdim=True)
    styled = gray + (styled - gray) * saturation

    # Grain: small, high-frequency perturbation. It is intentionally bounded
    # so a bad coefficient head cannot destroy the content.
    noise = _seeded_noise((b, 3, height, width), seed, device, dtype)
    grain_amplitude = (0.012 + 0.055 * grain_strength) * strength
    styled = styled + noise * grain_amplitude

    # Light leak: a low-frequency colored overlay, not a per-pixel effect.
    basis = _leak_basis(height, width, device, dtype)
    red = 0.55 + 0.45 * p[:, 5].view(b, 1, 1, 1)
    green = 0.20 + 0.35 * p[:, 6].view(b, 1, 1, 1)
    blue = 0.08 + 0.22 * p[:, 7].view(b, 1, 1, 1)
    leak_color = torch.cat([red, green, blue], dim=1)
    styled = styled + basis * leak_color * leak_strength * (0.18 * strength)

    # Soft vignette. Coordinates are normalized to the current tensor, so the
    # effect is stable for square model inputs and tiled inference.
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    radius = torch.sqrt((xx - 0.0) ** 2 + (yy - 0.0) ** 2) / math.sqrt(2.0)
    vignette = 1.0 - radius.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0) * vignette_strength
    styled = styled * vignette

    return styled.clamp(0.0, 1.0)


def effect_parameter_names() -> tuple[str, ...]:
    return (
        "grain_strength",
        "light_leak_strength",
        "exposure_offset",
        "saturation",
        "vignette_strength",
        "leak_red",
        "leak_green",
        "leak_blue",
    )
