from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture configuration stored in every checkpoint."""

    base_channels: int = 48
    bottleneck_channels: int = 256
    transformer_blocks: int = 1
    transformer_heads: int = 4
    transformer_token_size: int = 16
    effect_dim: int = 8
    use_mask: bool = False
    residual_scale: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in fields})


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
