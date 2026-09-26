from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw

from .config import ModelConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor[:3].detach().float().clamp(0.0, 1.0).cpu()
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(array * 255.0), mode="RGB")


def save_preview(
    path: str | Path,
    source: torch.Tensor,
    styled: torch.Tensor,
    target: torch.Tensor,
    *,
    labels: tuple[str, str, str] = ("source", "generated", "target domain"),
) -> None:
    images = [tensor_to_pil(source), tensor_to_pil(styled), tensor_to_pil(target)]
    width, height = images[0].size
    label_height = 28
    canvas = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 8, 7), label, fill="black")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild whichever architecture the checkpoint was trained with.

    The architecture is read from the checkpoint's own ``model_config``, never
    from the caller's arguments, so inference always matches the weights.
    """

    from .arch import build_generator

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint.get("model_config", {}))
    model = build_generator(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def checkpoint_step(checkpoint: dict[str, Any]) -> str:
    return f"epoch-{int(checkpoint.get('epoch', 0)):04d}-step-{int(checkpoint.get('global_step', 0)):07d}"
