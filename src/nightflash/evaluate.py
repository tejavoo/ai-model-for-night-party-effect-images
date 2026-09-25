from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image, ImageOps

from .data import list_images
from .effects import apply_stochastic_effects
from .losses import edge_loss
from .utils import load_model_checkpoint, resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an unpaired Night Flash checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--mask-dir", default=None, help="Optional source mask directory for mask-aware checkpoints")
    parser.add_argument("--output-json", default="runs/nightflash/evaluation.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--effects-strength", type=float, default=0.5)
    parser.add_argument("--no-effects", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def _read_image(path: Path, size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def _read_mask(path: Path, size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(Image.open(path).convert("L"))
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def _stats(tensor: torch.Tensor) -> dict[str, list[float]]:
    value = tensor[0, :3].detach().float().clamp(0.0, 1.0)
    return {
        "mean": value.mean(dim=(1, 2)).cpu().tolist(),
        "std": value.std(dim=(1, 2), unbiased=False).cpu().tolist(),
    }


def _simple_ssim(prediction: torch.Tensor, source: torch.Tensor) -> float:
    pred = prediction[0, :3].float()
    original = source[0, :3].float()
    c1 = 0.01**2
    c2 = 0.03**2
    x = pred.mean().item()
    y = original.mean().item()
    variance_x = pred.var(unbiased=False).item()
    variance_y = original.var(unbiased=False).item()
    covariance = ((pred - x) * (original - y)).mean().item()
    numerator = (2 * x * y + c1) * (2 * covariance + c2)
    denominator = (x**2 + y**2 + c1) * (variance_x + variance_y + c2)
    return numerator / max(denominator, 1e-8)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    size = args.image_size or int(checkpoint.get("train_args", {}).get("image_size", 256))
    source_paths = list_images(args.source_dir)[: args.limit]
    target_paths = list_images(args.target_dir)[: args.limit]
    if not source_paths or not target_paths:
        raise ValueError("Both source and target directories must contain images")
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    if model.config.use_mask and mask_dir is None:
        raise ValueError("This checkpoint requires --mask-dir")
    if not model.config.use_mask and mask_dir is not None:
        raise ValueError("--mask-dir was supplied for a checkpoint that does not use a mask")

    content_values: list[float] = []
    edge_values: list[float] = []
    ssim_values: list[float] = []
    effect_means: list[list[float]] = []
    diversity_values: list[float] = []
    source_means: list[list[float]] = []
    target_means: list[list[float]] = []

    with torch.no_grad():
        for index, source_path in enumerate(source_paths):
            source = _read_image(source_path, size).to(device)
            target_path = target_paths[index % len(target_paths)]
            target = _read_image(target_path, size)
            if mask_dir is not None:
                mask_path = mask_dir / f"{source_path.stem}.png"
                if not mask_path.exists():
                    raise FileNotFoundError(mask_path)
                mask = _read_mask(mask_path, size).to(device)
                model_input = torch.cat([source, mask], dim=1)
            else:
                mask = None
                model_input = source
            residual, effect_params = model(model_input)
            deterministic = torch.clamp(source + residual, 0.0, 1.0)
            styled = (
                deterministic
                if args.no_effects
                else apply_stochastic_effects(
                    deterministic,
                    effect_params,
                    seed=args.seed + index,
                    strength=args.effects_strength,
                )
            )
            content_values.append(float((styled - source).abs().mean().cpu()))
            edge_values.append(float(edge_loss(styled, source).cpu()))
            ssim_values.append(_simple_ssim(styled, source))
            effect_means.append(effect_params[0].cpu().tolist())
            source_means.append(_stats(source)["mean"])
            target_means.append(_stats(target)["mean"])

            if index == 0:
                samples = []
                for sample_index in range(4):
                    sample = apply_stochastic_effects(
                        deterministic,
                        effect_params,
                        seed=args.seed + sample_index,
                        strength=args.effects_strength,
                    )
                    samples.append(sample)
                diversity_values.append(float(torch.stack(samples).std(dim=0).mean().cpu()))

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "source_images": len(source_paths),
        "target_images": len(target_paths),
        "image_size": size,
        "effects_enabled": not args.no_effects,
        "effects_strength": 0.0 if args.no_effects else args.effects_strength,
        "content_l1_mean": mean(content_values),
        "edge_l1_mean": mean(edge_values),
        "simple_ssim_mean": mean(ssim_values),
        "effect_parameter_mean": [mean(values) for values in zip(*effect_means)],
        "source_rgb_mean": [mean(values) for values in zip(*source_means)],
        "target_rgb_mean": [mean(values) for values in zip(*target_means)],
        "seed_diversity_l1": mean(diversity_values) if diversity_values else 0.0,
        "notes": [
            "These are unpaired diagnostics, not proof of target-domain quality.",
            "Add FID/KID, DINO/CLIP, face, and human A/B evaluation for release decisions.",
        ],
    }
    save_json(args.output_json, result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run(args)
    print("Evaluation summary")
    for key in ("content_l1_mean", "edge_l1_mean", "simple_ssim_mean", "seed_diversity_l1"):
        print(f"{key}: {result[key]:.6f}")
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
