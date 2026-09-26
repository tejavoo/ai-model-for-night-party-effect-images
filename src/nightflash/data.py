from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .augment import AugmentConfig, make_pair

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


def _load_rgb(path: str | Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(image)


def _load_mask(path: str | Path) -> Image.Image:
    image = Image.open(path).convert("L")
    return ImageOps.exif_transpose(image)


def _resize_short(image: Image.Image, size: int, resample: int = Image.Resampling.BICUBIC) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    resized_width = max(size, int(round(width * scale)))
    resized_height = max(size, int(round(height * scale)))
    return image.resize((resized_width, resized_height), resample)


def _choose_crop(image: Image.Image, size: int, rng: random.Random, randomize: bool) -> tuple[int, int, int, int]:
    width, height = image.size
    max_x = width - size
    max_y = height - size
    if randomize and (max_x > 0 or max_y > 0):
        left = rng.randint(0, max_x)
        top = rng.randint(0, max_y)
    else:
        left = max_x // 2
        top = max_y // 2
    return left, top, left + size, top + size


def _resize_and_crop(
    image: Image.Image,
    size: int,
    rng: random.Random,
    randomize: bool,
    resample: int = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    resized = _resize_short(image, size, resample)
    crop = _choose_crop(resized, size, rng, randomize)
    return resized.crop(crop), crop


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _find_mask(mask_dir: Path | None, source_root: Path, source_path: Path) -> Path | None:
    if mask_dir is None:
        return None
    relative = source_path.relative_to(source_root)
    candidates = [
        mask_dir / relative.with_suffix(".png"),
        mask_dir / relative.with_suffix(".mask.png"),
        mask_dir / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class UnpairedImageDataset(Dataset[dict[str, Any]]):
    """Source/target domain dataset that never requires aligned pairs.

    Geometry is sampled once per item and applied identically to the source, the
    target and the mask, so the unpaired pair still describes a consistent crop
    position. See ``augment.py`` for the rationale behind each knob.
    """

    def __init__(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        *,
        image_size: int = 256,
        mask_dir: str | Path | None = None,
        seed: int = 0,
        randomize: bool = True,
        deterministic_target: bool = False,
        augment: "AugmentConfig | None" = None,
    ) -> None:
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.target_dir = Path(target_dir).expanduser().resolve()
        self.mask_dir = Path(mask_dir).expanduser().resolve() if mask_dir else None
        self.source_paths = list_images(self.source_dir)
        self.target_paths = list_images(self.target_dir)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.randomize = bool(randomize)
        self.deterministic_target = bool(deterministic_target)
        self.augment = augment or AugmentConfig()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.source_paths)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self.epoch * 100_003 + index * 97)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_path = self.source_paths[index]
        if self.deterministic_target:
            target_path = self.target_paths[index % len(self.target_paths)]
        else:
            rng = self._rng(index)
            target_path = rng.choice(self.target_paths)

        rng = self._rng(index + 17_003)
        source = _load_rgb(source_path)
        target = _load_rgb(target_path)
        mask_image = None
        if self.mask_dir is not None:
            mask_path = _find_mask(self.mask_dir, self.source_dir, source_path)
            if mask_path is not None:
                mask_image = _load_mask(mask_path)

        if self.randomize:
            # One shared geometric warp across source, target and mask.
            source, target, mask_image = make_pair(
                source, target, self.image_size, self.augment, rng, mask_image
            )
        else:
            # Preview / evaluation path: deterministic centre crop, no jitter.
            source = _resize_and_crop(source, self.image_size, rng, False)[0]
            target = _resize_and_crop(target, self.image_size, rng, False)[0]
            if mask_image is not None:
                mask_image = _resize_and_crop(
                    mask_image, self.image_size, rng, False, Image.Resampling.NEAREST
                )[0]

        mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        if mask_image is not None:
            mask = _to_tensor(mask_image).mean(dim=0, keepdim=True)

        return {
            "source": _to_tensor(source),
            "target": _to_tensor(target),
            "mask": mask,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }


def make_dataloaders(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    image_size: int,
    batch_size: int,
    mask_dir: str | Path | None = None,
    num_workers: int = 2,
    seed: int = 0,
    augment: "AugmentConfig | None" = None,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], UnpairedImageDataset]:
    train_dataset = UnpairedImageDataset(
        source_dir,
        target_dir,
        image_size=image_size,
        mask_dir=mask_dir,
        seed=seed,
        randomize=True,
        augment=augment,
    )
    preview_dataset = UnpairedImageDataset(
        source_dir,
        target_dir,
        image_size=image_size,
        mask_dir=mask_dir,
        seed=seed + 1,
        randomize=False,
        deterministic_target=True,
        augment=augment,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) >= batch_size,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    preview_loader = DataLoader(
        preview_dataset,
        batch_size=min(batch_size, 8),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, preview_loader, preview_dataset
