from __future__ import annotations

import torch

from nightflash.config import ModelConfig
from nightflash.effects import apply_stochastic_effects
from nightflash.models import HybridGenerator, PatchDiscriminator, StyleExportWrapper


def test_generator_and_effect_shapes() -> None:
    config = ModelConfig(
        base_channels=8,
        bottleneck_channels=16,
        transformer_blocks=1,
        transformer_heads=2,
        transformer_token_size=4,
        effect_dim=8,
    )
    model = HybridGenerator(config)
    image = torch.rand(2, 3, 64, 64)
    residual, effects = model(image)
    assert residual.shape == image.shape
    assert effects.shape == (2, 8)
    styled = torch.clamp(image + residual, 0.0, 1.0)
    styled_with_effects = apply_stochastic_effects(styled, effects, seed=7, strength=0.5)
    assert styled_with_effects.shape == image.shape
    assert torch.isfinite(styled_with_effects).all()


def test_mask_input_and_export_wrapper() -> None:
    config = ModelConfig(
        base_channels=8,
        bottleneck_channels=16,
        transformer_blocks=0,
        use_mask=True,
        effect_dim=8,
    )
    model = HybridGenerator(config)
    image_mask = torch.rand(1, 4, 64, 64)
    wrapper = StyleExportWrapper(model, emit_effects=True).eval()
    styled, effects = wrapper(image_mask)
    assert styled.shape == (1, 3, 64, 64)
    assert effects.shape == (1, 8)


def test_patch_discriminator() -> None:
    discriminator = PatchDiscriminator()
    output = discriminator(torch.rand(1, 3, 64, 64))
    assert output.shape[-1] > 0
    assert output.shape[-2] > 0
