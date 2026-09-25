from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageOps

from .data import IMAGE_EXTENSIONS, list_images
from .effects import apply_stochastic_effects
from .utils import ensure_dir, load_model_checkpoint, resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained Night Flash model on images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", required=True, help="Output image file or directory")
    parser.add_argument("--mask", default=None, help="Optional mask for a single input image")
    parser.add_argument("--image-size", type=int, default=None, help="Override checkpoint image size")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--effects-strength", type=float, default=0.5)
    parser.add_argument("--no-effects", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-comparison", action="store_true")
    return parser


def _load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


def _load_mask(path: Path, size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(Image.open(path).convert("L"))
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.NEAREST)
    import numpy as np

    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).unsqueeze(0)


def _prepare_image(image: Image.Image, size: int) -> tuple[torch.Tensor, Image.Image]:
    original = image.copy()
    resized = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    import numpy as np

    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, original


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    import numpy as np

    array = tensor[0, :3].detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(array * 255.0), mode="RGB")


def _input_paths(value: str) -> list[Path]:
    path = Path(value).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        return list_images(path)
    raise FileNotFoundError(path)


def _comparison(original: Image.Image, styled: Image.Image) -> Image.Image:
    width, height = original.size
    canvas = Image.new("RGB", (width * 2, height), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(styled, (width, 0))
    return canvas


def run(args: argparse.Namespace) -> list[Path]:
    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    size = args.image_size or int(checkpoint.get("train_args", {}).get("image_size", 256))
    use_mask = bool(model.config.use_mask)
    mask_path = Path(args.mask) if args.mask else None
    if use_mask and mask_path is None:
        raise ValueError("This checkpoint was trained with a mask input; pass --mask")

    inputs = _input_paths(args.input)
    if not inputs:
        raise ValueError("No input images found")
    output_root = Path(args.output).expanduser()
    if len(inputs) > 1 or output_root.is_dir() or not output_root.suffix:
        ensure_dir(output_root)
    written: list[Path] = []

    with torch.no_grad():
        for index, input_path in enumerate(inputs):
            original = _load_rgb(input_path)
            source, original_image = _prepare_image(original, size)
            source = source.to(device)
            mask = torch.zeros((1, size, size), device=device)
            if mask_path is not None:
                if not mask_path.exists():
                    raise FileNotFoundError(mask_path)
                mask = _load_mask(mask_path, size).to(device)
            model_input = torch.cat([source, mask], dim=1) if use_mask else source
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
            styled_image = _tensor_to_image(styled)
            if styled_image.size != original_image.size:
                styled_image = styled_image.resize(original_image.size, Image.Resampling.LANCZOS)

            if len(inputs) == 1 and output_root.suffix.lower() in IMAGE_EXTENSIONS:
                destination = output_root
            else:
                destination = output_root / f"{input_path.stem}-nightflash{input_path.suffix.lower() or '.png'}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.save_comparison:
                destination = destination.with_name(destination.stem + "-comparison" + destination.suffix)
                _comparison(original_image, styled_image).save(destination)
            else:
                styled_image.save(destination)
            written.append(destination)
            print(f"{input_path} -> {destination}")

    save_json(
        output_root / "inference.json" if output_root.is_dir() else output_root.with_suffix(".json"),
        {
            "checkpoint": str(args.checkpoint),
            "image_size": size,
            "seed": args.seed,
            "effects_strength": 0.0 if args.no_effects else args.effects_strength,
            "outputs": [str(path) for path in written],
        },
    )
    return written


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
