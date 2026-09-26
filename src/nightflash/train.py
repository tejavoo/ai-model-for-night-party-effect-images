from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .arch import run_generator
from .config import ModelConfig, TrainPaths
from .data import make_dataloaders
from .effects import apply_stochastic_effects
from .losses import (
    adversarial_loss,
    content_loss,
    effect_regularization,
    identity_loss,
    moment_loss,
)
from .models import HybridGenerator, PatchDiscriminator, count_parameters
from .progress import create_reporter
from .utils import ensure_dir, resolve_device, save_json, save_preview, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a compact unpaired Night Flash image-to-image model."
    )
    parser.add_argument("--source-dir", required=True, help="Directory of normal source-domain images")
    parser.add_argument("--target-dir", required=True, help="Directory of desired target-style images")
    parser.add_argument("--mask-dir", default=None, help="Optional source person-mask directory")
    parser.add_argument("--output-dir", default="runs/nightflash", help="Checkpoint and preview directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional global step limit for smoke tests")
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--bottleneck-channels", type=int, default=256)
    parser.add_argument("--transformer-blocks", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-token-size", type=int, default=16)
    parser.add_argument("--effect-dim", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--no-transformer", action="store_true", help="Use a convolution-only generator")
    parser.add_argument("--effects-strength", type=float, default=0.5)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--content-weight", type=float, default=1.5)
    parser.add_argument("--moment-weight", type=float, default=0.35)
    parser.add_argument("--identity-weight", type=float, default=0.75)
    parser.add_argument("--effect-weight", type=float, default=0.02)
    parser.add_argument("--preview-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--progress",
        choices=["auto", "tqdm", "plain", "none"],
        default="auto",
        help="progress style; 'auto' uses a live bar on TTYs and clean lines in notebooks",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny two-step sanity check")
    return parser


def _configure_smoke_test(args: argparse.Namespace) -> None:
    args.image_size = 64
    args.batch_size = 2
    args.epochs = 1
    args.max_steps = 2
    args.base_channels = 8
    args.bottleneck_channels = 16
    args.transformer_blocks = 0
    args.transformer_heads = 1
    args.transformer_token_size = 4
    args.preview_every = 1
    args.log_every = 1
    args.num_workers = 0
    args.effects_strength = 0.25


def _model_input(source: Tensor, mask: Tensor, use_mask: bool) -> Tensor:
    return torch.cat([source, mask], dim=1) if use_mask else source


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _scaler(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return None
    return torch.amp.GradScaler("cuda")


def _save_checkpoint(
    path: Path,
    *,
    model: HybridGenerator,
    discriminator: PatchDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    model_config: ModelConfig,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "model_config": model_config.to_dict(),
            "train_args": vars(args),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def _run_preview(
    model: HybridGenerator,
    batch: dict[str, Any],
    device: torch.device,
    path: Path,
    *,
    use_mask: bool,
    effects_strength: float,
) -> None:
    model.eval()
    with torch.no_grad():
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        residual, effect_params, _ = run_generator(
            model, _model_input(source, mask, use_mask)
        )
        deterministic = torch.clamp(source + residual, 0.0, 1.0)
        styled = apply_stochastic_effects(
            deterministic,
            effect_params,
            seed=424242,
            strength=effects_strength,
        )
    save_preview(path, source[0], styled[0], target[0])
    model.train()


def train(args: argparse.Namespace) -> Path:
    if args.smoke_test:
        _configure_smoke_test(args)

    paths = TrainPaths(
        source_dir=Path(args.source_dir),
        target_dir=Path(args.target_dir),
        output_dir=Path(args.output_dir),
        mask_dir=Path(args.mask_dir) if args.mask_dir else None,
    )
    paths.validate()
    output_dir = ensure_dir(paths.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    preview_dir = ensure_dir(output_dir / "previews")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_mask = paths.mask_dir is not None
    model_config = ModelConfig(
        base_channels=args.base_channels,
        bottleneck_channels=args.bottleneck_channels,
        transformer_blocks=0 if args.no_transformer else args.transformer_blocks,
        transformer_heads=args.transformer_heads,
        transformer_token_size=args.transformer_token_size,
        effect_dim=args.effect_dim,
        use_mask=use_mask,
        residual_scale=args.residual_scale,
    )
    model = HybridGenerator(model_config).to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4)

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    train_loader, preview_loader, preview_dataset = make_dataloaders(
        paths.source_dir,
        paths.target_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        mask_dir=paths.mask_dir,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if len(train_loader) == 0:
        raise ValueError("The training DataLoader is empty; reduce --batch-size or add images")

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = _scaler(device, amp_enabled)
    print(f"Device: {device}")
    print(f"Source images: {len(preview_dataset.source_paths)}")
    print(f"Target images: {len(preview_dataset.target_paths)}")
    print(f"Generator parameters: {count_parameters(model):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Previews: {preview_dir}")
    if start_epoch > 0:
        print(f"Resuming at epoch {start_epoch + 1} (global step {global_step})")

    # --epochs is an absolute target, so a resume at or past it trains nothing.
    # Fail loudly instead of printing a cheerful "training complete".
    if start_epoch >= args.epochs:
        raise SystemExit(
            f"Nothing to train: --epochs {args.epochs} is not greater than the "
            f"resume point (epoch {start_epoch + 1}).\n"
            "--epochs is an ABSOLUTE target, not a count of extra epochs.\n"
            f"Set --epochs to {start_epoch + 1} + the number of extra epochs you want."
        )

    reporter = create_reporter(mode=args.progress, log_every=args.log_every)

    for epoch in range(start_epoch, args.epochs):
        preview_dataset.set_epoch(epoch)
        model.train()
        discriminator.train()
        reporter.start_epoch(epoch + 1, args.epochs, len(train_loader))
        epoch_start = time.time()
        step_in_epoch = 0
        running: dict[str, float] = {}

        for batch in train_loader:
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            model_input = _model_input(source, mask, use_mask)

            # Discriminator update.
            optimizer_d.zero_grad(set_to_none=True)
            with _autocast(device, amp_enabled):
                residual, effect_params, _ = run_generator(model, model_input)
                deterministic = torch.clamp(source + residual, 0.0, 1.0)
                styled = apply_stochastic_effects(
                    deterministic,
                    effect_params,
                    seed=None,
                    strength=args.effects_strength,
                )
                real_logits = discriminator(target)
                fake_logits = discriminator(styled.detach())
                loss_d_real = adversarial_loss(real_logits, True)
                loss_d_fake = adversarial_loss(fake_logits, False)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
            if scaler is not None:
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                scaler.update()
            else:
                loss_d.backward()
                optimizer_d.step()

            # Generator update.
            optimizer_g.zero_grad(set_to_none=True)
            with _autocast(device, amp_enabled):
                residual, effect_params, _ = run_generator(model, model_input)
                deterministic = torch.clamp(source + residual, 0.0, 1.0)
                styled = apply_stochastic_effects(
                    deterministic,
                    effect_params,
                    seed=None,
                    strength=args.effects_strength,
                )
                fake_logits = discriminator(styled)
                loss_adv = adversarial_loss(fake_logits, True)
                loss_content = content_loss(styled, source, mask if use_mask else None)
                loss_moment = moment_loss(styled, target)
                if use_mask:
                    target_input = torch.cat([target, torch.zeros_like(mask)], dim=1)
                else:
                    target_input = target
                target_residual, _, _ = run_generator(model, target_input)
                target_identity = torch.clamp(target + target_residual, 0.0, 1.0)
                loss_identity = identity_loss(target_identity, target)
                loss_effect = effect_regularization(effect_params)
                loss_g = (
                    args.adversarial_weight * loss_adv
                    + args.content_weight * loss_content
                    + args.moment_weight * loss_moment
                    + args.identity_weight * loss_identity
                    + args.effect_weight * loss_effect
                )
            if scaler is not None:
                scaler.scale(loss_g).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer_g)
                scaler.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_g.step()

            global_step += 1
            step_in_epoch += 1
            values = {
                "loss_d": float(loss_d.detach().cpu()),
                "loss_g": float(loss_g.detach().cpu()),
                "adv": float(loss_adv.detach().cpu()),
                "content": float(loss_content.detach().cpu()),
                "identity": float(loss_identity.detach().cpu()),
            }
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + value
            reporter.update(step_in_epoch, values)
            if not args.no_preview and args.preview_every > 0 and global_step % args.preview_every == 0:
                preview_batch = next(iter(preview_loader))
                preview_path = preview_dir / f"step-{global_step:07d}.png"
                _run_preview(
                    model,
                    preview_batch,
                    device,
                    preview_path,
                    use_mask=use_mask,
                    effects_strength=args.effects_strength,
                )
                _save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=model,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    model_config=model_config,
                    args=args,
                    epoch=epoch,
                    global_step=global_step,
                )
                print(f"Preview: {preview_path}")
                print(f"Checkpoint: {checkpoint_dir / 'latest.pt'}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        reporter.finish_epoch(
            epoch + 1,
            args.epochs,
            time.time() - epoch_start,
            {key: value / max(step_in_epoch, 1) for key, value in running.items()},
        )

        epoch_path = checkpoint_dir / f"epoch-{epoch + 1:04d}.pt"
        _save_checkpoint(
            epoch_path,
            model=model,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            model_config=model_config,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )
        _save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            model_config=model_config,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )
        if not args.no_preview:
            preview_batch = next(iter(preview_loader))
            _run_preview(
                model,
                preview_batch,
                device,
                preview_dir / f"epoch-{epoch + 1:04d}.png",
                use_mask=use_mask,
                effects_strength=args.effects_strength,
            )
        print(f"Checkpoint: {epoch_path}")
        if not args.preview_every or args.preview_every <= 0:
            print(f"Previews disabled (--preview-every {args.preview_every})")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_json(
        output_dir / "run_config.json",
        {
            "model": model_config.to_dict(),
            "train_args": vars(args),
            "device": str(device),
            "source_images": len(preview_dataset.source_paths),
            "target_images": len(preview_dataset.target_paths),
            "global_step": global_step,
        },
    )
    return checkpoint_dir / "latest.pt"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = train(args)
    print(f"Training complete. Latest checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
