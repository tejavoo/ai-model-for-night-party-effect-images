from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .data import IMAGE_EXTENSIONS, list_images
from .effects import apply_stochastic_effects
from .geometry import prepare, restore
from .utils import ensure_dir, load_model_checkpoint, resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained Night Flash model on images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", required=True, help="Output image file or directory")
    parser.add_argument("--mask", default=None, help="Optional mask for a single input image")
    parser.add_argument("--image-size", type=int, default=None, help="Override checkpoint image size")
    parser.add_argument(
        "--fit",
        choices=["letterbox", "native", "crop"],
        default="letterbox",
        help=(
            "letterbox: keep every pixel, pad to square (default, matches training). "
            "native: run at the true aspect ratio. "
            "crop: legacy centre-crop, loses edges and stretches the result."
        ),
    )
    parser.add_argument(
        "--pad-mode",
        choices=["edge", "reflect", "black"],
        default="edge",
        help="How letterbox padding is filled (only used by --fit letterbox)",
    )
    parser.add_argument(
        "--effects-at",
        choices=["output", "model"],
        default="output",
        help=(
            "output (default): render grain/leak/vignette at the final resolution, "
            "so grain stays 1 pixel regardless of --image-size. "
            "model: render at model resolution (cheaper, grain is rescaled)."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--effects-strength", type=float, default=0.5)
    parser.add_argument("--no-effects", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-comparison", action="store_true")
    return parser


def _load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


def _mask_for(image: Image.Image, transform, device: torch.device) -> torch.Tensor:
    """Resize a mask through exactly the same transform as its image.

    Nearest-neighbour is used for both scale and padding so a binary person mask
    never picks up interpolated grey edges.
    """

    width, height = transform.target_size
    resized = image.resize((max(1, int(round(image.width * transform.scale))),
                            max(1, int(round(image.height * transform.scale)))),
                           Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    pad_left, pad_top = transform.pad_left, transform.pad_top
    pad_right = width - array.shape[1] - pad_left
    pad_bottom = height - array.shape[0] - pad_top
    if pad_left or pad_top or pad_right or pad_bottom:
        array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode="edge" if max(pad_left, pad_top, pad_right, pad_bottom) < min(array.shape) else "constant")
    tensor = torch.from_numpy(array[None, None].astype(np.float32))
    return tensor.to(device)


def _tensor_to_chw(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor[0, :3]
        .detach()
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )


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
    mask_image = None
    if use_mask:
        if not args.mask:
            raise ValueError("This checkpoint was trained with a mask input; pass --mask")
        mask_image = ImageOps.exif_transpose(Image.open(args.mask).convert("L"))
    elif args.mask:
        raise ValueError("--mask was supplied but this checkpoint does not use a mask")

    inputs = _input_paths(args.input)
    if not inputs:
        raise ValueError("No input images found")
    output_root = Path(args.output).expanduser()
    single_file_output = len(inputs) == 1 and output_root.suffix.lower() in IMAGE_EXTENSIONS
    if not single_file_output:
        ensure_dir(output_root)
    written: list[Path] = []

    for index, input_path in enumerate(inputs):
        original = _load_rgb(input_path)
        source_hw, transform = prepare(
            original, size, fit=args.fit, pad_mode=args.pad_mode
        )
        source = torch.from_numpy(np.ascontiguousarray(source_hw)).unsqueeze(0).to(device)

        if use_mask and mask_image is not None:
            mask = _mask_for(mask_image, transform, device)
        else:
            mask = torch.zeros((1, transform.target_size[1], transform.target_size[0]), device=device)

        model_input = torch.cat([source, mask], dim=1) if use_mask else source
        with torch.no_grad():
            residual, effect_params = model(model_input)
        deterministic = torch.clamp(source + residual, 0.0, 1.0)

        # Undo the letterbox BEFORE effects so vignette/leak are evaluated on the
        # real frame, and so grain is rendered at the delivered resolution.
        restored = restore(_tensor_to_chw(deterministic), transform)

        if args.no_effects:
            styled_image = restored
        elif args.effects_at == "model":
            styled = apply_stochastic_effects(
                deterministic, effect_params, seed=args.seed + index,
                strength=args.effects_strength,
            )
            styled_image = restore(_tensor_to_chw(styled), transform)
        else:
            rgb = torch.from_numpy(
                np.asarray(restored, dtype=np.float32).copy() / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)
            styled = apply_stochastic_effects(
                rgb, effect_params, seed=args.seed + index,
                strength=args.effects_strength,
            )
            styled_image = restore(_tensor_to_chw(styled), transform)

        if single_file_output:
            destination = output_root
        else:
            destination = output_root / f"{input_path.stem}-nightflash{input_path.suffix.lower() or '.png'}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.save_comparison:
            destination = destination.with_name(destination.stem + "-comparison" + destination.suffix)
            _comparison(original, styled_image).save(destination)
        else:
            styled_image.save(destination)
        written.append(destination)
        print(f"{input_path} {original.size} -> model {transform.target_size} -> {destination} {styled_image.size}")

    save_json(
        output_root / "inference.json" if output_root.is_dir() else output_root.with_suffix(".json"),
        {
            "checkpoint": str(args.checkpoint),
            "image_size": size,
            "fit": args.fit,
            "pad_mode": args.pad_mode,
            "effects_at": args.effects_at,
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
