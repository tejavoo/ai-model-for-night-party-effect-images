"""Output heads.

The head is the last stage of the generator: it turns decoder features into the
RGB delta that gets added to the input. Different heads suit different goals.

``tanh``
    Bounded delta. Most conservative, cannot repaint a pixel by more than
    ``residual_scale``. Feels like a global colour filter.

``perpixel``
    Unbounded delta, clamped only on read. This is the "actually repaint pixels"
    head: no artificial ceiling, so it can commit to strong local changes.

``freqsplit``
    Three parallel branches predicting low / mid / high frequency bands. Film
    grain and halation live in the high band, so routing them separately stops
    the main trunk from having to fight for both tone and texture.

``lut3d``
    A learned trilinear 3D colour LUT (classic mobile photo-filter
    representation) plus a per-pixel delta. Gives a cheap, globally-coherent
    grade that generalises far better than per-pixel alone on a small dataset.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F



def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _zero_init(layer: nn.Module) -> None:
    """Zero the final layer's weight and bias so the head starts as a no-op."""

    last = list(layer)[-1]
    if isinstance(last, (nn.Conv2d, nn.Linear)):
        if last.weight is not None:
            nn.init.zeros_(last.weight)
        if getattr(last, "bias", None) is not None:
            nn.init.zeros_(last.bias)


class TanhHead(nn.Module):
    """Bounded residual. The safest option when data is scarce."""

    def __init__(self, channels: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return torch.tanh(self.net(features)) * self.scale


class PerPixelHead(nn.Module):
    """Unbounded per-pixel RGB delta, clamped by the caller on read."""

    def __init__(self, channels: int, limit: float = 0.5) -> None:
        super().__init__()
        self.limit = float(limit)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
        )
        # Start near identity so early training does not shock the image.
        _zero_init(self.net)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).clamp(-self.limit, self.limit)


class FreqSplitHead(nn.Module):
    """Predict low / mid / high bands separately and recombine.

    A fixed 3-tap blur and its complement act as fixed band separators, so the
    head stays a plain convolutional network (Core ML friendly) while still
    giving each frequency band its own pathway.
    """

    def __init__(self, channels: int, limit: float = 0.5) -> None:
        super().__init__()
        self.limit = float(limit)
        self.low = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
        )
        self.high = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
        )
        for branch in (self.low, self.mid, self.high):
            _zero_init(branch)

    @staticmethod
    def _blur(x: Tensor) -> Tensor:
        kernel = x.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
        kernel = kernel / kernel.sum()
        channels = x.shape[1]
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=channels)

    def forward(self, features: Tensor) -> Tensor:
        low = self._blur(features)
        high = features - low
        delta_low = self.low(low)
        delta_mid = self.mid(features)
        delta_high = self.high(high)
        return (delta_low + delta_mid + delta_high).clamp(-self.limit, self.limit)

    def band_magnitudes(self, features: Tensor) -> dict[str, Tensor]:
        """Diagnostic: how much energy each band is contributing."""

        low = self._blur(features)
        high = features - low
        return {
            "low": self.low(low).abs().mean(),
            "mid": self.mid(features).abs().mean(),
            "high": self.high(high).abs().mean(),
        }


class Lut3DHead(nn.Module):
    """Learned trilinear 3D LUT plus a per-pixel delta.

    The ``size x size x size`` cube is stored as a single-channel image of shape
    ``(size, size * size)``: the row is the red index and the column packs
    ``green * size + blue``. A colour transform then becomes one ``grid_sample``
    with normalised coordinates, which is the standard formulation and converts
    to Core ML as a gather plus interpolation.

    ``size = 33`` costs 35,937 parameters.
    """

    def __init__(self, channels: int, size: int = 33, limit: float = 0.5) -> None:
        super().__init__()
        self.size = int(size)
        self.limit = float(limit)
        s = self.size
        ramp = torch.linspace(0.0, 1.0, s)
        r_idx = torch.arange(s).view(s, 1, 1).expand(s, s, s)
        g_idx = torch.arange(s).view(1, s, 1).expand(s, s, s)
        b_idx = torch.arange(s).view(1, 1, s).expand(s, s, s)
        identity = torch.stack([ramp[r_idx], ramp[g_idx], ramp[b_idx]], dim=-1)
        # Flat (s**3, 3) so a single index_select yields (N, 3) RGB values.
        # Per-channel gathers confuse the Core ML scatter lowering.
        self.table = nn.Parameter(identity.reshape(-1, 3).clone())
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
        )
        _zero_init(self.local)

    def _apply_table(self, rgb: Tensor) -> Tensor:
        """Trilinear interpolation over the LUT, written with explicit gathers.

        ``F.grid_sample`` is the usual way to do this, but it is not reliable
        enough to hang a learned transform on: it has been observed returning
        near-centre values regardless of the grid it is given, and Core ML
        support is uneven. Gathering the eight surrounding knots and blending
        them is exact, easy to unit test, and lowers to plain indexing.
        """

        s = self.size
        last = s - 1
        batch, _, height, width = rgb.shape
        flat = rgb.permute(0, 2, 3, 1).reshape(batch, -1, 3).clamp(0.0, 1.0)
        table = self.table  # (s**3, 3)

        axes = []
        for channel in range(3):
            position = flat[..., channel] * last
            low = position.floor().clamp(0, last)
            frac = (position - low).clamp(0.0, 1.0)
            high = (low + 1).clamp(max=last)
            axes.append((low, high, frac))

        (r0, r1, fr), (g0, g1, fg), (b0, b1, fb) = axes

        result = None
        for dr, wr in ((r0, 1.0 - fr), (r1, fr)):
            for dg, wg in ((g0, 1.0 - fg), (g1, fg)):
                for db, wb in ((b0, 1.0 - fb), (b1, fb)):
                    index = (dr * s * s + dg * s + db).long().clamp(0, s**3 - 1)
                    values = torch.index_select(table, 0, index.reshape(-1))
                    values = values.reshape(batch, -1, 3)
                    contribution = values * (wr * wg * wb).unsqueeze(-1)
                    result = contribution if result is None else result + contribution

        assert result is not None
        return result.reshape(batch, height, width, 3).permute(0, 3, 1, 2)

    def forward(self, features: Tensor, image: Tensor | None = None) -> Tensor:
        """``image`` is required: the LUT is a colour transform of the input."""

        if image is None:
            raise ValueError("Lut3DHead needs the source image to apply the LUT")
        graded = self._apply_table(image[:, :3])
        delta = graded - image[:, :3]
        return (delta + self.local(features)).clamp(-self.limit, self.limit)


SIMPLE_HEADS = ("tanh", "perpixel", "freqsplit", "lut3d")
COMPOSITE_HEADS = (
    "lut3d+freqsplit",  # global colour grade + local band detail
    "lut3d+perpixel",   # global colour grade + unconstrained local delta
    "tanh+freqsplit",   # bound everything, but keep band separation
)
ALL_HEADS = SIMPLE_HEADS + COMPOSITE_HEADS


def _build_simple(name: str, channels: int, config) -> nn.Module:  # type: ignore[no-untyped-def]
    if name == "tanh":
        return TanhHead(channels, config.residual_scale)
    if name == "perpixel":
        return PerPixelHead(channels, config.residual_limit)
    if name == "freqsplit":
        return FreqSplitHead(channels, config.residual_limit)
    if name == "lut3d":
        return Lut3DHead(channels, config.lut_size, config.residual_limit)
    raise ValueError(f"unknown head: {name!r}; choose from {SIMPLE_HEADS}")


class CompositeHead(nn.Module):
    """Sum of several simple heads.

    Combining heads is worthwhile only when they do genuinely different things.
    ``lut3d`` supplies a global colour transform while ``freqsplit`` supplies
    local per-band detail, so the pair is complementary. ``perpixel`` and
    ``freqsplit`` are both unconstrained per-pixel paths, so pairing them mostly
    just doubles the parameters.

    Each member keeps its own output bound; the sum is then clamped once, so a
    combined head is still a bounded residual no matter how many parts it has.
    """

    def __init__(self, names: tuple[str, ...], channels: int, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        if len(names) < 2:
            raise ValueError("a composite head needs at least two members")
        self.names = names
        self.limit = float(getattr(config, "residual_limit", 0.5))
        self.parts = nn.ModuleList(
            _build_simple(name, channels, config) for name in names
        )

    def forward(self, features: Tensor, image: Tensor | None = None) -> Tensor:
        total: Tensor | None = None
        for part in self.parts:
            if isinstance(part, Lut3DHead):
                if image is None:
                    raise ValueError("a composite containing lut3d needs the source image")
                contribution = part(features, image)
            else:
                contribution = part(features)
            total = contribution if total is None else total + contribution
        assert total is not None
        return total.clamp(-self.limit, self.limit)


class ResidualHead(nn.Module):
    """Dispatch to a simple or composite head named in the config."""

    def __init__(self, channels: int, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.kind = config.head
        self.needs_image = False

        if "+" in self.kind:
            names = tuple(part.strip() for part in self.kind.split("+"))
            unknown = [n for n in names if n not in SIMPLE_HEADS]
            if unknown:
                raise ValueError(
                    f"unknown head(s) {unknown} in {self.kind!r}; "
                    f"simple heads are {SIMPLE_HEADS}"
                )
            self.impl: nn.Module = CompositeHead(names, channels, config)
            self.needs_image = any(n == "lut3d" for n in names)
        elif self.kind in SIMPLE_HEADS:
            self.impl = _build_simple(self.kind, channels, config)
            self.needs_image = self.kind == "lut3d"
        else:
            raise ValueError(
                f"unknown head: {self.kind!r}; choose from {ALL_HEADS}"
            )

    def forward(self, features: Tensor, image: Tensor | None = None) -> Tensor:
        if self.needs_image and image is None:
            raise ValueError(f"head {self.kind!r} needs the source image")
        if self.needs_image:
            return self.impl(features, image)  # type: ignore[call-arg]
        return self.impl(features)  # type: ignore[call-arg]

