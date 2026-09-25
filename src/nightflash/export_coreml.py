from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .utils import load_model_checkpoint, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained model to Core ML.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output .mlpackage or .mlmodel path")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--convert-to", choices=["mlprogram", "mlmodel"], default="mlprogram")
    parser.add_argument("--compute-units", choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"], default="ALL")
    parser.add_argument("--half", action="store_true", help="Request float16 ML Program precision")
    parser.add_argument("--deterministic-only", action="store_true", help="Do not emit effect coefficients")
    parser.add_argument("--mask-input", action="store_true", help="Export a four-channel RGB+mask tensor input")
    parser.add_argument(
        "--input-type",
        choices=["image", "tensor"],
        default="image",
        help="Use image preprocessing or an explicit normalized tensor input",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def _compute_unit(value: str):
    import coremltools as ct

    return {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }[value]


def export(args: argparse.Namespace) -> Path:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(
            "coremltools is required. Install requirements-coreml.txt on macOS first."
        ) from exc

    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    size = args.image_size or int(checkpoint.get("train_args", {}).get("image_size", 256))
    if args.mask_input and not model.config.use_mask:
        raise ValueError("--mask-input requires a checkpoint trained with --mask-dir")

    from .models import StyleExportWrapper

    wrapper = StyleExportWrapper(model, emit_effects=not args.deterministic_only).eval()
    example = torch.zeros(1, 4 if args.mask_input else 3, size, size, dtype=torch.float32)
    if args.input_type == "tensor" or args.mask_input:
        input_type = ct.TensorType(name="inputTensor", shape=tuple(example.shape))
    else:
        input_type = ct.ImageType(
            name="inputImage",
            shape=(1, 3, size, size),
            scale=1.0 / 255.0,
            color_layout=ct.colorlayout.RGB,
        )

    styled_output: object
    if args.input_type == "tensor" or args.mask_input:
        styled_output = ct.TensorType(name="styledImage")
    else:
        styled_output = ct.ImageType(
            name="styledImage",
            color_layout=ct.colorlayout.RGB,
        )

    outputs: list[object]
    if args.deterministic_only:
        outputs = [styled_output]
    else:
        outputs = [styled_output, ct.TensorType(name="effectParams")]

    convert_kind = "neuralnetwork" if args.convert_to == "mlmodel" else "mlprogram"
    convert_kwargs = {
        "source": "pytorch",
        "inputs": [input_type],
        "outputs": outputs,
        "convert_to": convert_kind,
        "compute_units": _compute_unit(args.compute_units),
    }
    if args.half and convert_kind == "mlprogram":
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16

    print(f"Converting {args.checkpoint} at {size}x{size} to {args.convert_to} ...")
    # Recent coremltools versions require a TorchScript module or an
    # ExportedProgram rather than a plain eager PyTorch module.
    traced = torch.jit.trace(wrapper, example, strict=False)
    converted = ct.convert(traced, **convert_kwargs)
    try:
        converted.short_description = "Night Flash compact unpaired image style model"
        converted.input_description = "RGB image in [0,1], or RGB+person-mask tensor when mask input is enabled"
        converted.output_description = "Styled RGB image and optional eight effect coefficients"
    except Exception:
        # Some coremltools versions expose descriptions only through the spec.
        pass

    destination = Path(args.output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(destination))
    print(f"Saved Core ML artifact: {destination}")
    print("Validate the artifact with Xcode or a small Swift/Core ML smoke test before integration.")
    return destination


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    export(args)


if __name__ == "__main__":
    main()
