from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .arch import run_generator
from .utils import load_model_checkpoint, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PyTorch and Core ML outputs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coreml", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser


def _load_pytorch_image(path: Path, size: int, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, image


def _predict_coreml(model_path: Path, tensor: torch.Tensor, source_image: Image.Image) -> np.ndarray:
    import coremltools as ct

    model = ct.models.MLModel(str(model_path))
    spec = model.get_spec()
    input_feature = spec.description.input[0]
    feature_type = input_feature.type.WhichOneof("Type")
    if feature_type == "imageType":
        output = model.predict({input_feature.name: source_image})
        value = output["styledImage"]
        if isinstance(value, Image.Image):
            array = np.asarray(value.convert("RGB"), dtype=np.float32)
            if array.max() > 1.5:
                array = array / 255.0
            return array
        array = np.asarray(value, dtype=np.float32)
        if array.max() > 1.5:
            array = array / 255.0
        return array[0, ..., :3].transpose(1, 2, 0)
    if feature_type != "multiArrayType":
        raise ValueError(f"Unsupported Core ML input type: {feature_type}")

    output = model.predict({input_feature.name: tensor.cpu().numpy()})
    value = np.asarray(output["styledImage"], dtype=np.float32)
    if value.max() > 1.5:
        value = value / 255.0
    return value[0].transpose(1, 2, 0)


def run(args: argparse.Namespace) -> dict[str, float]:
    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    size = args.image_size or int(checkpoint.get("train_args", {}).get("image_size", 256))
    source, source_image = _load_pytorch_image(Path(args.input), size, device)
    if model.config.use_mask:
        raise ValueError("Core ML validation currently supports checkpoints without a mask input")
    with torch.no_grad():
        residual, _, _ = run_generator(model, source)
        pytorch_output = torch.clamp(source + residual, 0.0, 1.0)[0].cpu().permute(1, 2, 0).numpy()

    coreml_output = _predict_coreml(Path(args.coreml), source, source_image)
    pytorch_output = np.asarray(pytorch_output, dtype=np.float32)
    coreml_output = np.asarray(coreml_output, dtype=np.float32)
    if pytorch_output.shape != coreml_output.shape:
        raise ValueError(f"Shape mismatch: PyTorch {pytorch_output.shape}, Core ML {coreml_output.shape}")
    difference = np.abs(pytorch_output - coreml_output)
    result = {
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
        "rmse": float(np.sqrt(np.mean((pytorch_output - coreml_output) ** 2))),
    }
    print("Core ML comparison")
    for key, value in result.items():
        print(f"{key}: {value:.8f}")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(coreml_output, 0.0, 1.0) * 255.0)).save(output_dir / "coreml-output.png")
        Image.fromarray(np.uint8(np.clip(pytorch_output, 0.0, 1.0) * 255.0)).save(output_dir / "pytorch-output.png")
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
