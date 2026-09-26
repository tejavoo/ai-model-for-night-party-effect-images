#!/usr/bin/env python3
"""Warm-start a hybrid (conv + transformer) model from a conv-only checkpoint.

This is the supported way to move from Phase 1 to Phase 2.

Why this exists
---------------
`nightflash-train --resume` performs a **strict** `load_state_dict`, so it can
only continue a checkpoint whose architecture is byte-for-byte identical. A
conv-only checkpoint has no ``transformer.*`` keys, so resuming it into a hybrid
model fails with "Missing key(s) in state_dict".

This script performs the transfer deliberately and safely:

  * loads every conv/encoder/decoder tensor from the old checkpoint
  * leaves the newly-added transformer block at its random initialisation
  * **verifies** that the only missing keys are ``transformer.*``
  **resets** both optimizer states, because the previous moments belong to a
    different parameter set
  * writes a new checkpoint that ``--resume`` can consume normally

The resulting checkpoint reports ``epoch = -1`` and ``global_step = 0`` so that
training restarts at epoch 0 with a clean step counter.

Usage
-----
    python scripts/warm_start.py \\
        --checkpoint runs/conv-pilot/checkpoints/latest.pt \\
        --output-dir runs/hybrid-warm

    # optional: also reset the discriminator weights
    python scripts/warm_start.py --checkpoint ... --output-dir ... --reset-discriminator

Then train as normal against the new checkpoint:

    python -m nightflash.train ... --resume runs/hybrid-warm/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nightflash.config import ModelConfig  # noqa: E402
from nightflash.models import HybridGenerator, PatchDiscriminator  # noqa: E402

# Optimizer hyper-parameters must match train.py exactly, otherwise the resumed
# run will not reproduce the same schedule.
GEN_LR, GEN_BETAS, GEN_WD = 2e-4, (0.5, 0.999), 1e-4
DIS_LR, DIS_BETAS, DIS_WD = 1e-4, (0.5, 0.999), 1e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="conv-only source checkpoint")
    parser.add_argument("--output-dir", required=True, help="new run directory to create")
    parser.add_argument(
        "--transformer-blocks",
        type=int,
        default=1,
        help="transformer blocks in the new hybrid model (default: 1)",
    )
    parser.add_argument(
        "--reset-discriminator",
        action="store_true",
        help="also start the discriminator from scratch instead of carrying its weights over",
    )
    parser.add_argument("--dry-run", action="store_true", help="verify only, write nothing")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    source = Path(args.checkpoint).expanduser().resolve()
    if not source.is_file():
        print(f"error: checkpoint not found: {source}", file=sys.stderr)
        return 1
    output_dir = Path(args.output_dir).expanduser().resolve()
    destination = output_dir / "checkpoints" / "latest.pt"

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    old_config = ModelConfig.from_dict(checkpoint.get("model_config", {}))
    old_state = checkpoint["model"]
    old_has_transformer = any(key.startswith("transformer.") for key in old_state)

    print("=" * 68)
    print("Warm start: conv-only  ->  hybrid")
    print("=" * 68)
    print(f"source checkpoint : {source}")
    print(f"old config        : transformer_blocks={old_config.transformer_blocks}")
    print(f"old epoch/step    : {checkpoint.get('epoch')} / {checkpoint.get('global_step')}")
    print(f"old has transformer: {old_has_transformer}")
    print()

    if old_has_transformer:
        print(
            "This checkpoint already contains a transformer. It is not conv-only.\n"
            "If you simply want to continue that run, use --resume directly:\n"
            f"  python -m nightflash.train ... --resume {source}"
        )
        return 2

    # Build the new hybrid model. Every field except the transformer count is
    # inherited from the old checkpoint, so channel widths cannot drift.
    new_config = ModelConfig.from_dict(
        {**old_config.to_dict(), "transformer_blocks": args.transformer_blocks}
    )
    model = HybridGenerator(new_config)

    incompatible = model.load_state_dict(old_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    print(f"new config        : transformer_blocks={new_config.transformer_blocks}")
    print(f"tensors transferred: {len(old_state)}")
    print(f"tensors randomly initialised: {len(missing)}")
    print()

    # --- safety checks -------------------------------------------------------
    # A bare strict=False would happily ignore genuine corruption, so verify that
    # the ONLY thing missing is the block we deliberately added.
    if unexpected:
        print("error: checkpoint has keys the new model does not expect:", file=sys.stderr)
        for key in unexpected:
            print(f"  - {key}", file=sys.stderr)
        return 3

    if not missing:
        print("error: nothing was left to initialise; expected transformer keys.", file=sys.stderr)
        return 3

    non_transformer = [key for key in missing if not key.startswith("transformer.")]
    if non_transformer:
        print("error: unexpected missing keys outside the transformer block:", file=sys.stderr)
        for key in non_transformer:
            print(f"  - {key}", file=sys.stderr)
        print(
            "\nThis usually means the source checkpoint used a different channel\n"
            "width, image size, or mask setting. Re-run Phase 1 with matching\n"
            "arguments instead of warm-starting.",
            file=sys.stderr,
        )
        return 3

    loaded = len(old_state)
    initialised = len(missing)
    print("OK  every missing key belongs to the new transformer block:")
    for key in missing[:6]:
        print(f"      {key}")
    if len(missing) > 6:
        print(f"      ... and {len(missing) - 6} more")
    print()

    # --- discriminator -------------------------------------------------------
    discriminator = PatchDiscriminator()
    if args.reset_discriminator:
        print("discriminator     : reset to random initialisation")
        disc_state = discriminator.state_dict()
    else:
        discriminator.load_state_dict(checkpoint["discriminator"])  # strict
        disc_state = discriminator.state_dict()
        print("discriminator     : weights carried over from the conv run")

    # --- fresh optimizers ---------------------------------------------------
    # The previous AdamW moments were accumulated over a different parameter
    # set, so they are discarded rather than carried into the hybrid run.
    optimizer_g = torch.optim.AdamW(
        model.parameters(), lr=GEN_LR, betas=GEN_BETAS, weight_decay=GEN_WD
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(), lr=DIS_LR, betas=DIS_BETAS, weight_decay=DIS_WD
    )
    print("optimizer state   : reset (fresh AdamW moments)")
    print()

    if args.dry_run:
        print("dry run: verified, nothing written")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": disc_state,
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "model_config": new_config.to_dict(),
            "train_args": {
                **checkpoint.get("train_args", {}),
                "no_transformer": new_config.transformer_blocks == 0,
                "transformer_blocks": new_config.transformer_blocks,
                "warm_started_from": str(source),
            },
            # -1 so train.py computes start_epoch = 0; 0 restarts preview/log counters
            "epoch": -1,
            "global_step": 0,
        },
        destination,
    )

    report = {
        "source_checkpoint": str(source),
        "new_checkpoint": str(destination),
        "transformer_blocks": new_config.transformer_blocks,
        "tensors_transferred": loaded,
        "tensors_randomly_initialised": initialised,
        "missing_keys_are_transformer_only": True,
        "optimizer_state_reset": True,
        "discriminator_reset": bool(args.reset_discriminator),
        "resumed_epoch": 0,
        "resumed_global_step": 0,
    }
    report_path = output_dir / "warm_start_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    size_mb = destination.stat().st_size / 1e6
    print(f"wrote {destination}  ({size_mb:.1f} MB)")
    print(f"wrote {report_path}")
    print()
    print("Next step — train against the NEW checkpoint:")
    print(f"  python -m nightflash.train \\")
    print(f"      --resume {destination} \\")
    print("      --transformer-blocks 1 ...")
    print()
    print("Note: the run is NOT comparable to a from-scratch hybrid run.")
    print("You inherited 40 epochs of trained conv weights, so treat this as a")
    print("separate experimental line and keep the conv run as your baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
