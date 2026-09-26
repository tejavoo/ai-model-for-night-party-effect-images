#!/usr/bin/env python3
"""Self-bootstrapping Colab training job.

Designed to be launched from a local terminal with the official Colab CLI:

    colab run --gpu T4 scripts/colab_train_job.py -- \\
        --source-dir /content/drive/MyDrive/nightflash-edge/training_photos/normal_photos \\
        --target-dir /content/drive/MyDrive/nightflash-edge/training_photos/disposable_photos \\
        --output-dir /content/drive/MyDrive/nightflash-edge/runs/job-1 \\
        --image-size 256 --batch-size 8 --epochs 30 --no-transformer --amp

The script is transmitted to a fresh Colab VM, where it then:

  1. obtains the nightflash-edge source (git clone or a previously uploaded zip)
  2. installs the package with ``pip install -e .``
  3. runs ``python -m nightflash.train`` with the supplied arguments

Nothing here depends on the local checkout being present on the VM, which is
what makes the single ``colab run`` command above possible.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_PROJECT = Path("/content/nightflash-edge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    source_group = parser.add_argument_group("getting the code onto the VM")
    source_group.add_argument("--repo-url", default="", help="Git URL to clone (e.g. https://github.com/you/nightflash-edge)")
    source_group.add_argument("--branch", default="main")
    source_group.add_argument("--zip", default="", help="Path to a zip already uploaded to the VM (e.g. /content/nightflash-edge-colab.zip)")
    source_group.add_argument("--project-dir", default=str(DEFAULT_PROJECT))

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--source-dir", required=True)
    train_group.add_argument("--target-dir", required=True)
    train_group.add_argument("--output-dir", required=True)
    train_group.add_argument("--mask-dir", default=None)
    train_group.add_argument("--image-size", type=int, default=256)
    train_group.add_argument("--batch-size", type=int, default=8)
    train_group.add_argument("--epochs", type=int, default=30)
    train_group.add_argument("--max-steps", type=int, default=0)
    train_group.add_argument("--num-workers", type=int, default=2)
    train_group.add_argument("--preview-every", type=int, default=250)
    train_group.add_argument("--log-every", type=int, default=25)
    train_group.add_argument("--seed", type=int, default=0)
    train_group.add_argument("--device", default="cuda")
    train_group.add_argument("--resume", default=None)
    train_group.add_argument("--no-transformer", action="store_true")
    train_group.add_argument("--amp", action="store_true")
    train_group.add_argument("--smoke-test", action="store_true")

    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit without touching the VM")
    return parser


def run(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def obtain_code(args: argparse.Namespace, dry_run: bool) -> Path:
    project = Path(args.project_dir)

    if (project / "pyproject.toml").exists():
        print(f"code already present at {project}")
        return project

    if args.repo_url:
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                args.branch,
                args.repo_url,
                str(project),
            ],
            dry_run=dry_run,
        )
    elif args.zip:
        zip_path = Path(args.zip)
        if not dry_run and not zip_path.exists():
            raise FileNotFoundError(
                f"{zip_path} not found on the VM. Upload it first with:\n"
                "  colab upload -s NAME nightflash-edge-colab.zip /content/nightflash-edge-colab.zip"
            )
        run(["unzip", "-q", "-o", str(zip_path), "-d", str(project.parent)], dry_run=dry_run)
    else:
        raise SystemExit(
            "No way to obtain the code. Pass --repo-url, --zip, or make sure the project "
            "already exists on the VM."
        )

    if not dry_run and not (project / "pyproject.toml").exists():
        raise SystemExit(f"failed to find pyproject.toml in {project}")
    return project


def build_train_command(args: argparse.Namespace, project: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nightflash.train",
        "--source-dir",
        args.source_dir,
        "--target-dir",
        args.target_dir,
        "--output-dir",
        args.output_dir,
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--num-workers",
        str(args.num_workers),
        "--preview-every",
        str(args.preview_every),
        "--log-every",
        str(args.log_every),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.mask_dir:
        command += ["--mask-dir", args.mask_dir]
    if args.max_steps:
        command += ["--max-steps", str(args.max_steps)]
    if args.resume:
        command += ["--resume", args.resume]
    if args.no_transformer:
        command.append("--no-transformer")
    if args.amp:
        command.append("--amp")
    if args.smoke_test:
        command.append("--smoke-test")
    return command


def main() -> None:
    args = build_parser().parse_args()
    project = obtain_code(args, args.dry_run)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=project, dry_run=args.dry_run)
    command = build_train_command(args, project)
    run(command, cwd=project, dry_run=args.dry_run)

    checkpoint = Path(args.output_dir) / "checkpoints" / "latest.pt"
    print()
    if args.dry_run:
        print(f"dry run complete. would train and write {checkpoint}")
    else:
        print(f"training finished. checkpoint: {checkpoint}")
        print("download it with:")
        print(f"  colab download -s NAME {checkpoint} ./latest.pt")


if __name__ == "__main__":
    main()
