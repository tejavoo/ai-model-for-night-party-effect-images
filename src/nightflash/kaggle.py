"""Kaggle session helpers.

Kaggle's ``/kaggle/working`` is wiped between notebook sessions, so a run that
spans more than one session has to be snapshotted out and pulled back in. These
helpers wrap the Kaggle API so the notebook body stays identical to the Colab
one and only the snapshot calls differ.

Typical flow across two sessions::

    # session 1
    s.train(extra_epochs=20)
    snapshot(s.run_dir)                    # -> a Kaggle Dataset version

    # session 2 (attach that dataset first)
    restore(s.run_dir)                     # copy the snapshot back into working
    s.train(extra_epochs=40)               # resumes where session 1 stopped
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def is_kaggle() -> bool:
    return INPUT.exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def gpus() -> list[str]:
    """Names of the GPUs Kaggle attached, in device order."""

    if not is_kaggle():
        return []
    import torch

    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def recommended_device(requested: str = "auto") -> str:
    """Pick a device, honouring an explicit request.

    On Kaggle a T4x2 kernel exposes two GPUs; a ~4M model will not saturate one,
    so the second device is better spent running a second experiment.
    """

    if requested != "auto":
        return requested
    if is_kaggle():
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    from .utils import resolve_device

    return str(resolve_device("auto"))


@dataclass
class SnapshotInfo:
    dataset: str
    version: int
    run_dir: str
    local_path: str


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def snapshot(run_dir: str | Path, *, dataset: str | None = None,
             private: bool = True, note: str | None = None) -> SnapshotInfo:
    """Publish a run directory as a new Kaggle Dataset version.

    The dataset is created on first call and versioned thereafter, so a resume
    only needs the dataset slug. Requires ``kaggle`` credentials in the usual
    ``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``.
    """

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    latest = run_dir / "checkpoints" / "latest.pt"
    if not latest.exists():
        raise FileNotFoundError(f"nothing to snapshot: {latest} does not exist")

    slug = dataset or f"nightflash-{run_dir.name}".lower().replace("_", "-")
    api = _api()
    api.dataset_create_new(
        folder=str(run_dir), name=slug, private=private, exist_ok=True
    )
    version = api.dataset_create_version(str(run_dir), note=note or "nightflash snapshot")
    return SnapshotInfo(slug, int(getattr(version, "version_number", 0) or 0),
                        str(run_dir), str(run_dir))


def restore(run_dir: str | Path, *, dataset_dir: str | Path | None = None) -> Path:
    """Copy a snapshot back into the working directory.

    Point ``dataset_dir`` at the attached dataset folder if it is not the first
    match under ``/kaggle/input``.
    """

    run_dir = Path(run_dir)
    if dataset_dir is None:
        candidates = sorted(
            (p for p in INPUT.iterdir() if (p / "checkpoints" / "latest.pt").exists()),
            key=lambda p: p.name,
        )
        if not candidates:
            raise FileNotFoundError(
                "no attached dataset contains checkpoints/latest.pt. Attach the "
                "snapshot dataset in the notebook settings, then re-run."
            )
        source = candidates[0]
    else:
        source = Path(dataset_dir)

    if run_dir.exists() and any(run_dir.iterdir()):
        index = 1
        while (candidate := run_dir.with_name(f"{run_dir.name}_archive_{index}")).exists():
            index += 1
        shutil.move(str(run_dir), str(candidate))
        print(f"archived current run -> {candidate}")

    shutil.copytree(source, run_dir)
    print(f"restored {source} -> {run_dir}")
    return run_dir


def _normalise(name: str) -> str:
    """Fold a dataset slug and a run directory name to the same comparable key.

    Kaggle dataset slugs are lowercased with ``_`` turned into ``-``, so the
    default slug for ``runs/edgehat_freqsplit_night`` is
    ``nightflash-edgehat-freqsplit-night``. Comparing raw names would never
    match, so both sides are folded to bare alphanumerics.
    """

    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_snapshot(run_dir: str | Path) -> Path | None:
    """Locate an attached snapshot that matches ``run_dir``, or ``None``.

    Matching ignores case and separator style, and accepts the ``nightflash-``
    prefix that :func:`snapshot` adds, so a notebook with several arms attached
    at once restores the right one. A dataset holding ``checkpoints/latest.pt``
    but no matching name is accepted only when it is the sole candidate, which
    keeps working when a snapshot was uploaded by hand under another name.
    """

    if not INPUT.exists():
        return None
    wanted = _normalise(Path(run_dir).name)
    named: list[Path] = []
    loose: list[Path] = []
    for entry in sorted(INPUT.iterdir()):
        if not (entry / "checkpoints" / "latest.pt").exists():
            continue
        key = _normalise(entry.name)
        if key == wanted or key.endswith(wanted) or wanted.endswith(key):
            named.append(entry)
        else:
            loose.append(entry)
    if named:
        return named[0]
    if len(loose) == 1:
        return loose[0]
    return None


def auto_restore(run_dir: str | Path, *, required: bool = False) -> Path | None:
    """Restore a snapshot if one is attached, otherwise do nothing.

    This is the "just keep going" counterpart to :func:`restore`. Call it at the
    top of every new session instead of remembering to restore by hand::

        from nightflash.kaggle import auto_restore
        auto_restore(s.run_dir)

    Returns the restored path, or ``None`` if there was nothing to restore (a
    normal first session). With ``required=True`` a missing snapshot is an
    error instead, which is what you want in the *last* session before you
    export.
    """

    if not is_kaggle():
        if required:
            raise RuntimeError("auto_restore(required=True) outside a Kaggle runtime")
        return None

    source = find_snapshot(run_dir)
    if source is None:
        if required:
            raise FileNotFoundError(
                f"no attached snapshot matches {Path(run_dir).name}. Attach the "
                "dataset in the notebook's right-hand Settings panel, or pass "
                "required=False to start fresh."
            )
        print(f"no attached snapshot for {Path(run_dir).name}; starting fresh")
        return None

    restore(run_dir, dataset_dir=source)
    return Path(run_dir)


def environment_report() -> dict[str, object]:
    """Everything worth printing at the top of a Kaggle notebook."""

    import torch

    return {
        "kaggle": is_kaggle(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpus": gpus(),
        "recommended_device": recommended_device("auto"),
        "working_dir": str(WORKING),
        "datasets": [p.name for p in sorted(INPUT.iterdir())] if INPUT.exists() else [],
    }
