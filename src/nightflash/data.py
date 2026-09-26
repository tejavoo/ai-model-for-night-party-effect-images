from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .augment import AugmentConfig, _sample_geometry, apply_geometry, make_pair
from .degrade import DegradeConfig, degraded_pair, make_config, stable_seed

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


class DegradationDataset(Dataset[dict[str, Any]]):
    """Builds a source domain by degrading images from the target pool.

    Used by the ``paired`` and ``sameset`` training modes. Unlike
    :class:`UnpairedImageDataset` this never reads a separate source directory,
    so the input distribution is fully under our control.

    Modes
    -----
    ``paired``
        ``source = degrade(image_i)``, ``target = image_i``. Exact
        correspondence, so real PSNR/SSIM are computable. Also the mode with a
        shortcut available: see the warning in :mod:`nightflash.degrade`.

    ``sameset``
        ``source = degrade(image_j)``, ``target = image_i``, ``j != i``. Both
        drawn from one pool with no correspondence, so the model cannot learn to
        invert the degradation and must learn the conditional distribution.

    In both modes the **shared geometric warp is applied first, to both images,
    and the degradation runs afterwards on the source only.** That ordering is
    what keeps the two halves consistent: any crop, flip or rotation is shared,
    and only the simulated camera differs.
    """

    def __init__(
        self,
        target_dir: str | Path,
        *,
        image_size: int = 256,
        mode: str = "paired",
        mask_dir: str | Path | None = None,
        seed: int = 0,
        randomize: bool = True,
        augment: "AugmentConfig | None" = None,
        degrade: "DegradeConfig | None" = None,
        epoch: int = 0,
        deterministic_seed: bool = False,
    ) -> None:
        if mode not in ("paired", "sameset"):
            raise ValueError(f"DegradationDataset mode must be paired or sameset, got {mode!r}")
        self.target_dir = Path(target_dir).expanduser().resolve()
        self.paths = list_images(self.target_dir)
        if not self.paths:
            raise FileNotFoundError(f"no images in {self.target_dir}")
        self.mode = mode
        self.image_size = int(image_size)
        self.mask_dir = Path(mask_dir).expanduser().resolve() if mask_dir else None
        self.seed = int(seed)
        self.randomize = bool(randomize)
        self.augment = augment or AugmentConfig()
        self.degrade = degrade if degrade is not None else make_config()
        self.epoch = int(epoch)
        # Validation sets need the same image to degrade identically in every
        # session, so the seed is derived from the filename rather than drawn
        # from the epoch-dependent stream.
        self.deterministic_seed = bool(deterministic_seed)
        self._fell_back = False
        if self.mode == "sameset" and len(self.paths) < 2:
            self._fell_back = True
            print(
                f"warning: sameset needs at least 2 images to draw a different source, "
                f"{self.target_dir} has {len(self.paths)}. Falling back to paired."
            )

    @property
    def effective_mode(self) -> str:
        return "paired" if self._fell_back else self.mode

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self.epoch * 100_003 + index * 97)

    def _stable_rng(self, path: Path) -> random.Random:
        return random.Random(stable_seed(path.name, str(self.seed), "degrade"))

    def _pick_pair(self, index: int, rng: random.Random) -> tuple[Path, Path]:
        """Return ``(source_origin_path, target_path)``."""
        target_path = self.paths[index]
        if self.effective_mode == "paired":
            return target_path, target_path
        choice = rng.randrange(len(self.paths) - 1)
        if choice >= index:
            choice += 1
        return self.paths[choice], target_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        source_origin, target_path = self._pick_pair(index, rng)
        target_image = _load_rgb(target_path)
        source_image = _load_rgb(source_origin)

        mask_path = None
        if self.mask_dir is not None:
            mask_path = _find_mask(self.mask_dir, self.target_dir, target_path)

        if self.randomize:
            # Shared warp on both, so the correspondence survives.
            source_image, target_image, mask_image = make_pair(
                source_image, target_image, self.image_size, self.augment, rng, None
            )
        else:
            source_image = _resize_and_crop(source_image, self.image_size, rng, False)[0]
            target_image = _resize_and_crop(target_image, self.image_size, rng, False)[0]
            mask_image = None

        # Degrade the source only, after the shared warp.
        config = self.degrade.resolved(self.epoch)
        if self.deterministic_seed:
            degrade_rng = self._stable_rng(source_origin)
        else:
            degrade_rng = rng
        source_image, degrade_params = degraded_pair(source_image, config, degrade_rng)

        mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        if mask_path is not None:
            mask_image = _load_mask(mask_path)
            if self.randomize:
                mask_image = apply_geometry(
                    mask_image.convert("L"), self.image_size,
                    _sample_geometry(self.augment, self._rng(index + 5)),
                    self._rng(index + 6),
                    resample=Image.Resampling.NEAREST, fill=0,
                )
            else:
                mask_image = _resize_and_crop(
                    mask_image, self.image_size, rng, False, Image.Resampling.NEAREST
                )[0]
            mask = _to_tensor(mask_image).mean(dim=0, keepdim=True)

        # Every extra field has to survive default_collate, so parameters are
        # returned as a fixed-size float tensor. The identity path yields zeros
        # plus a zero flag rather than None, which collate cannot handle.
        keys = ("resize", "blur", "contrast", "black_lift",
                "exposure", "saturation", "wb", "noise")
        if degrade_params is None:
            degrade_params = {k: 0.0 for k in keys}
            degraded_flag = 0.0
        else:
            degrade_params = {k: float(degrade_params.get(k, 0.0)) for k in keys}
            degraded_flag = 1.0

        return {
            "source": _to_tensor(source_image),
            "target": _to_tensor(target_image),
            "mask": mask,
            "source_path": str(source_origin),
            "target_path": str(target_path),
            "degrade_params": torch.tensor(
                [degrade_params[k] for k in keys], dtype=torch.float32
            ),
            "degraded": torch.tensor(degraded_flag, dtype=torch.float32),
            "is_paired": torch.tensor(
                float(source_origin == target_path), dtype=torch.float32
            ),
        }


def split_pool(
    paths: list[Path], fraction: float, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Deterministically split a pool by scene, so validation never leaks.

    Returns ``(train, validation)``. With a single image, or a fraction that
    would leave nothing to train on, the validation half is empty rather than
    silently stealing training data.
    """

    if len(paths) < 2:
        return list(paths), []
    ordered = sorted(paths)
    rng = random.Random(stable_seed("split", str(seed)))
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    count = int(round(len(shuffled) * float(fraction)))
    count = max(1, min(count, len(shuffled) - 1))
    validation = sorted(shuffled[:count])
    train = sorted(shuffled[count:])
    return train, validation


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


def make_mode_dataloaders(
    source_dir: str | Path | None,
    target_dir: str | Path,
    *,
    train_mode: str,
    image_size: int,
    batch_size: int,
    mask_dir: str | Path | None = None,
    num_workers: int = 2,
    seed: int = 0,
    augment: "AugmentConfig | None" = None,
    degrade: "DegradeConfig | None" = None,
    val_every: int = 1,
    val_count: int = 8,
    val_select: str = "fixed",
    val_split: float = 0.15,
    val_dir: str | Path | None = None,
    val_seed: int = 0,
) -> tuple[DataLoader, DataLoader, Dataset, "DataLoader | None", "Dataset | None"]:
    """Build loaders for any of the three training modes.

    Returns ``(train_loader, preview_loader, train_dataset, val_loader,
    val_dataset)``. The last two are ``None`` in ``unpaired`` mode, which has no
    held-out correspondence to score against.

    The validation half is drawn from the target pool by scene, so a validation
    image is never also a training image. In ``unpaired`` mode no validation
    loader is produced because a real quality number is not available there.
    """

    if train_mode == "unpaired":
        if not source_dir:
            raise ValueError("train_mode='unpaired' requires a source_dir")
        train_loader, preview_loader, dataset = make_dataloaders(
            source_dir, target_dir, image_size=image_size, batch_size=batch_size,
            mask_dir=mask_dir, num_workers=num_workers, seed=seed, augment=augment,
        )
        return train_loader, preview_loader, dataset, None, None

    pool = list_images(Path(target_dir).expanduser().resolve())
    if val_dir:
        validation_paths = list_images(Path(val_dir).expanduser().resolve())
        train_paths = [p for p in pool if p not in set(validation_paths)]
    else:
        train_paths, validation_paths = split_pool(pool, val_split, seed=val_seed)

    if not train_paths:
        raise ValueError(f"no training images left in {target_dir} after the validation split")

    def build(paths: list[Path], **kwargs) -> DegradationDataset:
        return _PoolDataset(paths, image_size=image_size, mode=train_mode,
                            seed=seed, augment=augment, degrade=degrade, **kwargs)

    train_dataset = build(train_paths, randomize=True, mask_dir=mask_dir)
    preview_dataset = build(train_paths, randomize=False)
    val_dataset = build(validation_paths, randomize=False, deterministic_seed=True)

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
        preview_dataset, batch_size=min(batch_size, 8), shuffle=False, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=min(val_count, 8), shuffle=False,
                            num_workers=0) if len(val_dataset) else None
    return train_loader, preview_loader, train_dataset, val_loader, val_dataset


class _PoolDataset(DegradationDataset):
    """A :class:`DegradationDataset` over an explicit path list.

    Needed because the pool has to be split before the dataset is constructed;
    the directory-based parent cannot express "these 90 paths, not all of them".
    """

    def __init__(self, paths: list[Path], **kwargs: Any) -> None:
        if not paths:
            raise ValueError("cannot build a dataset from an empty path list")
        self._paths = list(paths)
        super().__init__(paths[0].parent, **kwargs)
        self.paths = self._paths
