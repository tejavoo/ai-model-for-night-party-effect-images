from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture configuration stored in every checkpoint.

    Older fields describe the original ``hybrid`` generator and are preserved so
    existing checkpoints keep loading. The ``arch``/``head`` fields select the
    newer architectures added for the Night Flash v2 work.
    """

    base_channels: int = 48
    bottleneck_channels: int = 256
    transformer_blocks: int = 1
    transformer_heads: int = 4
    transformer_token_size: int = 16
    effect_dim: int = 8
    use_mask: bool = False
    residual_scale: float = 0.35

    # --- architecture selection -------------------------------------------
    arch: str = "hybrid"  # hybrid | restormer | edgehat | swinir
    head: str = "tanh"  # tanh | perpixel | freqsplit | lut3d

    # --- edgehat / swinir trunk -------------------------------------------
    edgevit_blocks_l3: int = 2
    edgevit_blocks_l4: int = 1
    hat_blocks_l4: int = 1
    swin_blocks_l3: int = 2
    swin_blocks_l4: int = 2
    swin_window: int = 8
    attention_heads: int = 4
    mlp_ratio: float = 4.0

    # --- optional capabilities --------------------------------------------
    use_halation: bool = False
    use_subject_gate: bool = False
    use_region_exposure: bool = False
    region_grid: int = 4
    halation_threshold: float = 0.72
    halation_tint: tuple[float, float, float] = (1.0, 0.62, 0.34)
    halation_strength: float = 0.35

    # --- output head -------------------------------------------------------
    lut_size: int = 33
    residual_limit: float = 0.5  # per-pixel clamp; tanh head uses residual_scale

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["halation_tint"] = list(self.halation_tint)
        return data

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in values.items() if k in fields}
        if isinstance(kwargs.get("halation_tint"), list):
            kwargs["halation_tint"] = tuple(kwargs["halation_tint"])
        return cls(**kwargs)

    # --- derived helpers ---------------------------------------------------

    @property
    def in_channels(self) -> int:
        return 4 if self.use_mask else 3

    def describe(self) -> str:
        """One-line human summary, used by the CLI and the notebook briefing."""

        return (
            f"{self.arch} + {self.head} "
            f"({'+'.join(self.capability_names()) if self.capabilities else 'plain'})"
        )

    def capability_names(self) -> list[str]:
        names = []
        if self.use_halation:
            names.append("halation")
        if self.use_subject_gate:
            names.append("subject-gate")
        if self.use_region_exposure:
            names.append("region-exposure")
        return names

    @property
    def capabilities(self) -> list[str]:
        return self.capability_names()


@dataclass
class TrainPaths:
    source_dir: Path
    target_dir: Path
    output_dir: Path
    mask_dir: Path | None = None

    def validate(self) -> None:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory does not exist: {self.target_dir}")
        if not any(p.is_file() for p in self.source_dir.rglob("*")):
            raise ValueError(f"No files found in source directory: {self.source_dir}")
        if not any(p.is_file() for p in self.target_dir.rglob("*")):
            raise ValueError(f"No files found in target directory: {self.target_dir}")
        if self.mask_dir is not None and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")
