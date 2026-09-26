"""``nf`` — one command-line entry point for every run operation.

Subcommands
-----------
``brief``    print the full pre-flight briefing and exit (default)
``state``    one-line state summary
``train``    resume or start, with the absolute-epoch guard built in
``infer``    run the checkpoint on images, no training
``reset``    archive or wipe a run
``config``   print the resolved config for a run

The design rule is that ``train`` never silently does nothing: if the target
epoch is at or before the resume point it refuses and tells you the number to
ask for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arch import ARCHITECTURES, HEADS
from .briefing import brief
from .session import Session


def add_common(parser: argparse.ArgumentParser, *, need_data: bool = True) -> None:
    parser.add_argument("--arch", choices=ARCHITECTURES, default="edgehat")
    parser.add_argument("--head", choices=HEADS, default="freqsplit")
    parser.add_argument("--name", default="night")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50,
                        help="ABSOLUTE target epoch, not a count of extra epochs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preview-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--effects-strength", type=float, default=0.5)
    parser.add_argument("--halation", dest="halation", action="store_true", default=True)
    parser.add_argument("--no-halation", dest="halation", action="store_false")
    parser.add_argument("--subject-gate", dest="subject_gate", action="store_true", default=True)
    parser.add_argument("--no-subject-gate", dest="subject_gate", action="store_false")
    parser.add_argument("--region-exposure", dest="region_exposure", action="store_true", default=True)
    parser.add_argument("--no-region-exposure", dest="region_exposure", action="store_false")
    parser.add_argument("--bottleneck-channels", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--region-grid", type=int, default=4)
    parser.add_argument(
        "--augment",
        choices=("default", "conservative", "strong", "none"),
        default="default",
        help="augmentation preset; see docs/TRAINING.md",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="publish a Kaggle Dataset version every N epochs; 0 disables",
    )
    parser.add_argument(
        "--snapshot-dataset",
        default=None,
        help="Kaggle dataset slug for snapshots; defaults to nightflash-<run>",
    )
    if need_data:
        parser.add_argument("--source-dir", default=None)
        parser.add_argument("--target-dir", default=None)
        parser.add_argument("--val-dir", default=None)
        parser.add_argument("--mask-dir", default=None)


def make_session(args: argparse.Namespace) -> Session:
    overrides = {"attention_heads": args.attention_heads, "region_grid": args.region_grid}
    if args.bottleneck_channels:
        overrides["bottleneck_channels"] = args.bottleneck_channels
    return Session(
        arch=args.arch,
        head=args.head,
        name=args.name,
        # state/config/reset subcommands do not take data paths
        source_dir=getattr(args, "source_dir", None),
        target_dir=getattr(args, "target_dir", None),
        val_dir=getattr(args, "val_dir", None),
        mask_dir=getattr(args, "mask_dir", None),
        runs_root=args.runs_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        use_amp=not args.no_amp,
        seed=args.seed,
        num_workers=args.num_workers,
        preview_every=args.preview_every,
        log_every=args.log_every,
        effects_strength=args.effects_strength,
        halation=args.halation,
        subject_gate=args.subject_gate,
        region_exposure=args.region_exposure,
        augment_preset=args.augment,
        snapshot_every=args.snapshot_every,
        snapshot_dataset=args.snapshot_dataset,
        **overrides,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nf",
        description="Night Flash training, inference and inspection.",
    )
    sub = parser.add_subparsers(dest="command")

    brief_p = sub.add_parser("brief", help="print the pre-flight briefing (default)")
    add_common(brief_p)
    brief_p.add_argument("--extra-epochs", type=int, default=None)

    state_p = sub.add_parser("state", help="one-line state summary")
    add_common(state_p, need_data=False)

    config_p = sub.add_parser("config", help="print the resolved config")
    add_common(config_p, need_data=False)

    train_p = sub.add_parser("train", help="resume or start training")
    add_common(train_p)
    train_p.add_argument("--extra-epochs", type=int, default=None)
    train_p.add_argument("--yes", action="store_true", help="skip the briefing prompt")

    infer_p = sub.add_parser("infer", help="run a checkpoint on images")
    add_common(infer_p, need_data=False)
    infer_p.add_argument("--input", required=True, help="image file or directory")
    infer_p.add_argument("--seeds", type=int, default=1)
    infer_p.add_argument("--comparison", action="store_true")
    infer_p.add_argument(
        "--tile",
        type=int,
        default=0,
        help="tile size in px for full-resolution inference; 0 (default) = whole frame",
    )
    infer_p.add_argument("--tile-overlap", type=int, default=128)
    infer_p.add_argument(
        "--pad-mode", choices=("edge", "reflect", "black"), default="edge",
        help="what edge tiles see outside the frame",
    )
    infer_p.add_argument(
        "--max-edge",
        type=int,
        default=0,
        help="downscale the long edge to at most N px, aspect preserved; 0 = native",
    )

    reset_p = sub.add_parser("reset", help="archive or wipe a run")
    add_common(reset_p, need_data=False)
    reset_p.add_argument("--delete", action="store_true", help="wipe in place, no archive")

    snap_p = sub.add_parser("snapshot", help="publish the run as a Kaggle Dataset version")
    add_common(snap_p, need_data=False)
    snap_p.add_argument("--note", default=None)
    snap_p.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if the snapshot fails instead of warning",
    )

    resume_p = sub.add_parser("resume", help="restore an attached Kaggle snapshot if present")
    add_common(resume_p, need_data=False)
    resume_p.add_argument(
        "--required", action="store_true",
        help="fail instead of starting fresh when no snapshot is attached",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["brief", *argv]  # default to the briefing

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    session = make_session(args)

    if args.command == "brief":
        print(brief(session, mode="plan", extra_epochs=args.extra_epochs))
        return 0

    if args.command == "state":
        state = session.state()
        print(json.dumps(state.to_dict(), indent=2))
        return 0

    if args.command == "config":
        print(json.dumps({
            "run_dir": str(session.run_dir),
            "model": session.config.to_dict(),
            "train": {
                "image_size": session.image_size, "batch_size": session.batch_size,
                "epochs": session.epochs, "device": session.device,
                "amp": session.use_amp, "seed": session.seed,
            },
        }, indent=2))
        return 0

    if args.command == "train":
        # Must ask in "train" mode: would_train is only set there.
        plan = session.plan("train", args.extra_epochs)
        if not args.yes:
            print(brief(session, mode="train", extra_epochs=args.extra_epochs))
        if not plan["would_train"]:
            print(f"\nrefusing to train: {plan['action']}")
            return 2
        result = session.train(args.extra_epochs)
        print(f"\ntraining complete, {result['epochs_this_call']} epoch(s), "
              f"checkpoint {session.latest}")
        return 0

    if args.command == "infer":
        written = session.infer(
            args.input,
            seeds=args.seeds,
            save_comparison=args.comparison,
            tile=args.tile,
            tile_overlap=args.tile_overlap,
            pad_mode=args.pad_mode,
            max_edge=args.max_edge,
        )
        print(f"wrote {len(written)} file(s) to {session.inference_dir}")
        return 0

    if args.command == "snapshot":
        info = session.snapshot(note=args.note, strict=args.strict)
        return 0 if info is not None else 1

    if args.command == "resume":
        return 0 if session.auto_resume(required=args.required) else 1

    if args.command == "reset":
        destination = session.reset(archive=not args.delete)
        if destination:
            print(f"archived to {destination}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
