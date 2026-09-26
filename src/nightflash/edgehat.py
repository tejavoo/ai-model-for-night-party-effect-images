"""EdgeViT + HAT generator with halation, subject gate and region exposure.

Why this design
---------------
The previous generator learned one bounded global filter. This one is built for
*localised, per-pixel* work with real attention:

``EdgeViTBlock``
    Local attention over a small window, with the attention factorised into
    sub-window branches. That factorisation is what makes it cheap, and — more
    importantly — it can be rewritten as depthwise convolutions, so Core ML runs
    it natively instead of lowering it to gathers.

``HATBlock``
    Overlapping cross-attention over a strided region. Local blocks cannot talk
    across the image, so this one block supplies global lighting context without
    paying for full attention.

Three optional heads address requirements the old model could not:

``halation``
    A physical operator, not a learned guess. Luminance above a threshold is
    spread with a dilated depthwise blur and tinted warm, which is what light
    scattering in the film base actually does.

``subject gate``
    A soft ``S(x, y)`` map from mid-level features, optionally fused with a
    Vision person mask, so tone and exposure can differ between subject and
    background.

``region exposure``
    Coarse grid coefficients (default 4x4) bilinearly upsampled, giving
    region-level exposure control rather than a single global value.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .heads import ResidualHead


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _zero_init(layer: nn.Module) -> None:
    """Zero the last layer's weight and bias so a head starts as a no-op.

    Several layers here are built with ``bias=False``, which leaves ``bias`` as
    ``None``; initialising unconditionally would raise.
    """

    last = list(layer)[-1] if isinstance(layer, (nn.Sequential, nn.ModuleList)) else layer
    if isinstance(last, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        if last.weight is not None:
            nn.init.zeros_(last.weight)
        if getattr(last, "bias", None) is not None:
            nn.init.zeros_(last.bias)


class NormAct(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(x))


class ConvFFN(nn.Module):
    """Depthwise + pointwise feed forward. Cheap and Core ML friendly."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeAttention(nn.Module):
    """Windowed attention with a depthwise positional term on the key.

    Attention runs over the ``window * window`` tokens inside each window, with
    full channel width (no head split), so the cost is ``O(window^4)`` per
    window and is independent of the input resolution. Because it is a window
    partition plus a matmul, Core ML can execute it natively rather than
    lowering it to gathers.
    """

    def __init__(self, channels: int, heads: int = 4, window: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.window = window
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        # Positional encoding is folded into the key, as in CPVT.
        self.pos = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.scale = channels**-0.5

    def forward(self, x: Tensor) -> Tensor:
        height, width = int(x.shape[-2]), int(x.shape[-1])
        # Clamp against concrete ints, not traced shapes: a tensor in min() would
        # produce a 0-d tensor and break window partitioning under jit.trace.
        window = min(self.window, height, width)
        window = max(1, window)
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        k = k + self.pos(x)  # local positional encoding

        def tile(t: Tensor) -> Tensor:
            b, c, h, w = t.shape
            return (
                t.view(b, c, h // window, window, w // window, window)
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(-1, window * window, c)
            )

        q_t, k_t, v_t = tile(q), tile(k), tile(v)
        attn = (q_t @ k_t.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v_t

        batch = x.shape[0]
        padded_h, padded_w = x.shape[-2:]
        out = (
            out.view(batch, padded_h // window, padded_w // window, window, window, self.channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, self.channels, padded_h, padded_w)
        )
        out = self.proj(out)
        if out.shape[-2:] != (height, width):
            out = out[..., :height, :width]
        return out


class EdgeViTBlock(nn.Module):
    """Pre-norm transformer block with conv FFN and a learned gate."""

    def __init__(self, channels: int, heads: int, window: int, expansion: int = 2) -> None:
        super().__init__()
        self.norm1 = NormAct(channels)
        self.attn = EdgeAttention(channels, heads, window)
        self.gamma1 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))
        self.norm2 = NormAct(channels)
        self.ffn = ConvFFN(channels, expansion)
        self.gamma2 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


class HATBlock(nn.Module):
    """Overlapping cross-attention: local blocks cannot see the whole frame."""

    def __init__(self, channels: int, heads: int, expansion: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.heads = max(1, heads // 2)
        self.key_dim = max(8, channels // self.heads)
        # Fixed token grid. Computed here as a plain int because under
        # torch.jit.trace an int derived from x.shape becomes a 0-d tensor, which
        # adaptive pooling then rejects.
        self.grid = max(2, (channels // 64) * 2)
        self.norm1 = NormAct(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.kv = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma1 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))
        self.norm2 = NormAct(channels)
        self.ffn = ConvFFN(channels, expansion)
        self.gamma2 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))
        self.scale = self.key_dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm1(x)
        q = self.q(normed)
        k, v = self.kv(normed).chunk(2, dim=1)

        # Fixed coarse token grid. `self.grid` is a plain int on purpose: under
        # torch.jit.trace an int derived from x.shape becomes a 0-d tensor, which
        # adaptive pooling rejects.
        grid = self.grid
        q_small = F.adaptive_avg_pool2d(q, grid).flatten(2).transpose(1, 2)
        k_small = F.adaptive_avg_pool2d(k, grid).flatten(2).transpose(1, 2)
        v_small = F.adaptive_avg_pool2d(v, grid).flatten(2).transpose(1, 2)

        attn = (q_small @ k_small.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        context = attn @ v_small  # (B, grid*grid, C)

        out = context.transpose(1, 2).reshape(x.shape[0], self.channels, grid, grid)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x + self.gamma1 * self.proj(out)
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


class SubjectGate(nn.Module):
    """Soft subject/background map, optionally guided by an external mask."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(max(4, channels // 2)), max(4, channels // 2)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, channels // 2), 1, 1),
        )
        _zero_init(self.net)
        if self.net[-1].bias is not None:
            nn.init.constant_(self.net[-1].bias, -2.0)  # start mostly background

    def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        gate = torch.sigmoid(self.net(features))
        if mask is not None:
            gate = torch.clamp(gate + mask, 0.0, 1.0)
        return gate


class RegionExposure(nn.Module):
    """Coarse grid of per-region exposure coefficients, smoothly upsampled.

    A 1x1 convolution over the pooled grid means each cell gets its own
    coefficients conditioned on its own features, which is what "per-region
    exposure" actually means. Flattening the grid into one big linear layer
    would couple every cell to every other and cost far more parameters.
    """

    def __init__(self, channels: int, grid: int = 4) -> None:
        super().__init__()
        self.grid = int(grid)
        hidden = max(4, channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.grid),
        )
        self.to_coeff = nn.Conv2d(hidden, 3, 1)
        _zero_init(self.to_coeff)

    def forward(self, features: Tensor) -> Tensor:
        coeff = self.to_coeff(self.net(features))  # (B, 3, grid, grid)
        return F.interpolate(
            coeff, size=features.shape[-2:], mode="bilinear", align_corners=False
        ) * 0.5  # bounded per-region exposure offset


class HalationBranch(nn.Module):
    """Physically-motivated glow around highlights.

    Light scattering through the film base is a low-pass operation on the
    over-threshold luminance, so a dilated depthwise blur is a faithful and very
    cheap model of it.
    """

    def __init__(self, channels: int, threshold: float, strength: float,
                 tint: tuple[float, float, float]) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.strength = float(strength)
        self.register_buffer("tint", torch.tensor(tint, dtype=torch.float32).view(1, 3, 1, 1))
        self.spill = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=2, dilation=2, bias=False),  # wider receptive field
        )
        _zero_init(self.spill)
        self.tonal = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(max(4, channels // 4)), max(4, channels // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, channels // 4), 1, 1),
        )
        _zero_init(self.tonal)

    def forward(self, image: Tensor, features: Tensor) -> Tensor:
        luma = (image[:, :3] * image.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        highlights = F.relu(luma - self.threshold) / max(1e-3, 1.0 - self.threshold)
        spill = self.spill(highlights)  # at image resolution
        tonal = self.tonal(features)
        if tonal.shape[-2:] != spill.shape[-2:]:
            tonal = F.interpolate(
                tonal, size=spill.shape[-2:], mode="bilinear", align_corners=False
            )
        return F.relu(spill + tonal * 0.5) * self.tint * self.strength


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.GroupNorm(_groups(in_channels), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeHatGenerator(nn.Module):
    """Conv encoder/decoder trunk with EdgeViT + HAT bottlenecks and extra heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        c1 = config.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = max(c3, config.bottleneck_channels)
        in_channels = config.in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(c1), c1),
            nn.SiLU(inplace=True),
        )
        self.down1 = DepthwiseSeparableBlock(c1, c2, stride=2)
        self.down2 = DepthwiseSeparableBlock(c2, c3, stride=2)
        self.down3 = DepthwiseSeparableBlock(c3, c4, stride=2)
        self.down4 = DepthwiseSeparableBlock(c4, c4, stride=2)

        self.l4 = nn.Sequential(*[
            EdgeViTBlock(c4, config.attention_heads, config.swin_window)
            for _ in range(config.edgevit_blocks_l4)
        ])
        self.hat = nn.Sequential(*[
            HATBlock(c4, config.attention_heads) for _ in range(config.hat_blocks_l4)
        ])
        self.l3 = nn.Sequential(*[
            EdgeViTBlock(c3, config.attention_heads, config.swin_window)
            for _ in range(config.edgevit_blocks_l3)
        ])

        # Level widths are stem=c1, s2=c2, s3=c3, s4=c4, s5=c4. The deep stages
        # fuse once at s3 rather than pairing s4 with s5, so L3 attention (c3) and
        # L4 attention (c4) meet in a single block.
        self.fuse = DepthwiseSeparableBlock(c4 + c3, c3)
        self.up2 = DepthwiseSeparableBlock(c3 + c2, c2)
        self.up1 = DepthwiseSeparableBlock(c2 + c1, c1)

        self.head = ResidualHead(c1, config)
        self.subject_gate = SubjectGate(c3) if config.use_subject_gate else None
        self.region_exposure = RegionExposure(c3, config.region_grid) if config.use_region_exposure else None
        self.halation = (
            HalationBranch(
                c3,
                config.halation_threshold,
                config.halation_strength,
                tuple(config.halation_tint),
            )
            if config.use_halation
            else None
        )
        self.effect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, config.effect_dim),
            nn.Sigmoid(),
        )
        # Gates start near zero so the extra heads cannot destabilise early steps.
        self.subject_mix = nn.Parameter(torch.zeros(1))
        self.region_mix = nn.Parameter(torch.zeros(1))
        self.halation_mix = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor | None]]:
        expected = self.config.in_channels
        if x.shape[1] != expected:
            raise ValueError(f"expected {expected} channels, got {x.shape[1]}")
        image = x[:, :3]
        mask_input = x[:, 3:4] if self.config.use_mask and x.shape[1] == 4 else None

        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)

        s5 = self.l4(s5)
        s5 = self.hat(s5)
        l3_features = self.l3(s3)

        # Deep path joins L3, then decode with the two shallower skips.
        deep = F.interpolate(s5, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.fuse(torch.cat([deep, l3_features], dim=1))
        d = F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.up2(torch.cat([d, s2], dim=1))
        d = F.interpolate(d, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d, s1], dim=1))

        residual = self.head(d1, image)
        extras: dict[str, Tensor | None] = {"subject": None, "region": None, "halation": None}

        def to_output_res(t: Tensor) -> Tensor:
            """Heads read s3 features; the residual is at full resolution."""

            if t.shape[-2:] == residual.shape[-2:]:
                return t
            return F.interpolate(
                t, size=residual.shape[-2:], mode="bilinear", align_corners=False
            )

        # The three extras are photometric 3-channel terms, so they compose onto
        # the RGB residual rather than the decoder features. Each is gated by a
        # learnable scalar initialised to zero, so a freshly built model is
        # exactly the plain residual and the extras earn their influence.
        if self.subject_gate is not None:
            # The gate reads s3 features, so the full-resolution Vision mask must be
            # pooled to the same grid before fusing.
            mask_for_gate = None
            if mask_input is not None:
                mask_for_gate = F.adaptive_max_pool2d(mask_input, s3.shape[-2:])
            gate = to_output_res(self.subject_gate(s3, mask_for_gate))
            extras["subject"] = gate
            gate_rgb = gate.expand(-1, 3, -1, -1)
            # Subject-selective tone: a gated, self-modulated copy of the delta.
            residual = residual + self.subject_mix * gate_rgb * torch.tanh(residual)

        if self.region_exposure is not None:
            field = to_output_res(self.region_exposure(s3))
            extras["region"] = field
            residual = residual + self.region_mix * field

        if self.halation is not None:
            glow = to_output_res(self.halation(image, s3))
            extras["halation"] = glow
            residual = residual + self.halation_mix * glow

        effects = self.effect_head(d1)
        return residual, effects, extras


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
