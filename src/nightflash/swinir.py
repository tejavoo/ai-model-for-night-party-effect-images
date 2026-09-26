"""SwinIR-style generator: windowed self-attention with relative position bias.

Kept as a comparison arm against ``edgehat``. The window partition is a pure
reshape (no ``torch.roll`` for the non-shifted case), so Core ML conversion is
cleaner than a full shifted-window implementation, at the cost of some boundary
seam risk between windows.

The shift is opt-in via ``swin_shift``; with it enabled the module uses
``torch.roll``, which Core ML lowers to gather operations — correct, but slower.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .edgehat import DepthwiseSeparableBlock, _groups
from .heads import ResidualHead


class WindowAttention(nn.Module):
    """Multi-head self-attention inside non-overlapping windows."""

    def __init__(self, channels: int, heads: int, window: int) -> None:
        super().__init__()
        if channels % heads != 0:
            heads = max(1, channels // 8)
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.window = window
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)

        # Relative position bias, the Swin trick that keeps the model
        # translation-equivariant without absolute positional embeddings.
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(window), torch.arange(window), indexing="ij"
        )).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window - 1
        relative[:, :, 1] += window - 1
        relative[:, :, 0] *= 2 * window - 1
        self.register_buffer("relative_index", relative.sum(-1), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        # The relative-position bias table is sized for self.window, so the
        # window must never shrink: pad small feature maps up instead.
        window = self.window
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        padded_h, padded_w = x.shape[-2:]
        tokens = (
            x.view(batch, channels, padded_h // window, window, padded_w // window, window)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, window * window, channels)
        )
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(t: Tensor) -> Tensor:
            return t.view(-1, window * window, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.relative_bias[self.relative_index.view(-1)].view(
            window * window, window * window, -1
        )
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, window * window, channels)
        out = self.proj(out)

        out = (
            out.view(batch, padded_h // window, padded_w // window, window, window, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, padded_h, padded_w)
        )
        return out[..., :height, :width]


class ShiftedWindowAttention(WindowAttention):
    """Window attention with a cyclic shift so windows overlap across blocks."""

    def forward(self, x: Tensor) -> Tensor:
        shifted = torch.roll(x, shifts=(-(self.window // 2), -(self.window // 2)), dims=(-2, -1))
        out = super().forward(shifted)
        return torch.roll(out, shifts=(self.window // 2, self.window // 2), dims=(-2, -1))


class SwinBlock(nn.Module):
    def __init__(self, channels: int, heads: int, window: int, shift: bool = False,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        attention = ShiftedWindowAttention if shift else WindowAttention
        self.attn = attention(channels, heads, window)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # LayerNorm normalises over the channel axis, so we move C to the last
        # position, apply the block, then move it back.
        normed = self.norm1(x.permute(0, 2, 3, 1))
        x = x + self.attn(normed.permute(0, 3, 1, 2))
        return x + self.mlp(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)


class SwinIRGenerator(nn.Module):
    """Conv trunk + windowed transformer bottlenecks + selectable head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        c1 = config.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = max(c3, config.bottleneck_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, c1, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(c1), c1),
            nn.SiLU(inplace=True),
        )
        self.down1 = DepthwiseSeparableBlock(c1, c2, stride=2)
        self.down2 = DepthwiseSeparableBlock(c2, c3, stride=2)
        self.down3 = DepthwiseSeparableBlock(c3, c4, stride=2)
        self.down4 = DepthwiseSeparableBlock(c4, c4, stride=2)

        self.l4 = nn.Sequential(*[
            SwinBlock(c4, config.attention_heads, config.swin_window,
                      shift=bool(i % 2), mlp_ratio=config.mlp_ratio)
            for i in range(config.swin_blocks_l4)
        ])
        self.l3 = nn.Sequential(*[
            SwinBlock(c3, config.attention_heads, config.swin_window,
                      shift=bool(i % 2), mlp_ratio=config.mlp_ratio)
            for i in range(config.swin_blocks_l3)
        ])

        # Level widths are stem=c1, s2=c2, s3=c3, s4=c4, s5=c4; the deep stages
        # fuse once at s3.
        self.fuse = DepthwiseSeparableBlock(c4 + c3, c3)
        self.up2 = DepthwiseSeparableBlock(c3 + c2, c2)
        self.up1 = DepthwiseSeparableBlock(c2 + c1, c1)

        self.head = ResidualHead(c1, config)
        self.effect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, config.effect_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor | None]]:
        image = x[:, :3]
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)

        s5 = self.l4(s5)
        l3_features = self.l3(s3)

        deep = F.interpolate(s5, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.fuse(torch.cat([deep, l3_features], dim=1))
        d = F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.up2(torch.cat([d, s2], dim=1))
        d = F.interpolate(d, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d, s1], dim=1))

        residual = self.head(d1, image)
        effects = self.effect_head(d1)
        return residual, effects, {"subject": None, "region": None, "halation": None}
