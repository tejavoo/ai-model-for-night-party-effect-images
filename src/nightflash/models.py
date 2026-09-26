from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


def _group_norm(channels: int) -> nn.GroupNorm:
    """Choose a small, divisor-safe normalization layer for Core ML export."""

    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DepthwiseSeparableBlock(nn.Module):
    """Mobile-friendly local block used in the encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            _group_norm(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeTransformerBlock(nn.Module):
    """Low-resolution global context block with bounded attention cost.

    The input feature map is pooled to a fixed token grid before self-attention.
    This avoids quadratic attention over a full 512x512 feature map and makes the
    block practical for mobile inference and Core ML conversion.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        token_size: int = 16,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.token_size = token_size
        self.norm1 = nn.LayerNorm(channels)
        self.heads = heads
        self.head_dim = channels // heads
        # Explicit q/k/v matmuls are more portable to Core ML than
        # nn.MultiheadAttention, whose fused PyTorch operator is not supported
        # by every coremltools version.
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.attention_projection = nn.Linear(channels, channels, bias=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * expansion
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        pooled = F.adaptive_avg_pool2d(x, (self.token_size, self.token_size))
        tokens = pooled.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        batch, token_count, _ = normalized.shape
        qkv = self.qkv(normalized)
        qkv = qkv.reshape(batch, token_count, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim**-0.5)
        attention = torch.softmax(attention, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(batch, token_count, self.channels)
        attended = self.attention_projection(attended)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        transformed = tokens.transpose(1, 2).reshape(x.shape[0], self.channels, self.token_size, self.token_size)
        return F.interpolate(transformed, size=(height, width), mode="bilinear", align_corners=False)


class HybridGenerator(nn.Module):
    """Content-preserving unpaired image translator.

    The network predicts a bounded RGB residual and a small vector of effect
    coefficients. Stochastic film effects are rendered separately by effects.py,
    which keeps the learned representation compact and reproducible.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c1 = self.config.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        cb = self.config.bottleneck_channels
        in_channels = 4 if self.config.use_mask else 3

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            _group_norm(c1),
            nn.SiLU(inplace=True),
        )
        self.down1 = DepthwiseSeparableBlock(c1, c2, stride=2)
        self.down2 = DepthwiseSeparableBlock(c2, c3, stride=2)
        self.down3 = DepthwiseSeparableBlock(c3, cb, stride=2)

        if self.config.transformer_blocks > 0:
            self.transformer = nn.Sequential(
                *[
                    EdgeTransformerBlock(
                        cb,
                        heads=self.config.transformer_heads,
                        token_size=self.config.transformer_token_size,
                    )
                    for _ in range(self.config.transformer_blocks)
                ]
            )
        else:
            self.transformer = nn.Identity()

        self.up3 = DepthwiseSeparableBlock(cb + c3, c3)
        self.up2 = DepthwiseSeparableBlock(c3 + c2, c2)
        self.up1 = DepthwiseSeparableBlock(c2 + c1, c1)

        self.residual_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            _group_norm(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, 3, kernel_size=1),
        )
        self.effect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, self.config.effect_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got {tuple(x.shape)}")
        expected_channels = 4 if self.config.use_mask else 3
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels, got {x.shape[1]}"
            )

        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        middle = self.transformer(s4)
        # The bottleneck is one level deeper than the first skip connection.
        # Upsample before concatenating so fixed and arbitrary input sizes work.
        middle = F.interpolate(middle, size=s3.shape[-2:], mode="bilinear", align_corners=False)

        u3 = self.up3(torch.cat([middle, s3], dim=1))
        u3 = F.interpolate(u3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u3, s2], dim=1))
        u2 = F.interpolate(u2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u2, s1], dim=1))

        residual = torch.tanh(self.residual_head(u1)) * self.config.residual_scale
        effects = self.effect_head(u1)
        return residual, effects


class PatchDiscriminator(nn.Module):
    """Small PatchGAN discriminator for target-domain distribution matching."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for out_channels in (base_channels * 2, base_channels * 4, base_channels * 8):
            layers.extend(
                [
                    nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = out_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StyleExportWrapper(nn.Module):
    """Core ML-friendly wrapper returning styled RGB and effect coefficients.

    Accepts any registered generator. ``edgehat`` and ``swinir`` return
    ``(residual, effects, extras)`` while ``hybrid`` returns a 2-tuple, so the
    dispatch goes through ``run_generator`` rather than unpacking blindly.
    """

    def __init__(self, generator: nn.Module, emit_effects: bool = True) -> None:
        super().__init__()
        self.generator = generator
        self.emit_effects = emit_effects

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        from .arch import run_generator

        residual, effects, _ = run_generator(self.generator, x)
        styled = torch.clamp(x[:, :3] + residual, min=0.0, max=1.0)
        if self.emit_effects:
            return styled, effects
        return styled


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
