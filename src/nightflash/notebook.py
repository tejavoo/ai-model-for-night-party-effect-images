"""Notebook helpers: streaming training + a live preview grid.

The two problems this solves
---------------------------
1. ``tqdm`` bars do not render in a Colab/Jupyter kernel, because the bar is
   drawn with ``\\r`` on stderr and the notebook frontend does not redraw it.
2. Reading a ``subprocess.Popen`` pipe in text mode makes Python's universal
   newlines translation turn every ``\\r`` into ``\\n``, so each bar redraw
   becomes its own printed line and floods the cell.

Both are fixed at the source: the trainer emits newline-terminated lines when it
detects a notebook, and :func:`stream_process` reads the pipe in binary and
separates real lines from in-place redraws.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .streaming import stream_process


def train(
    args: Sequence[str],
    *,
    root: str | Path = ".",
    log: str | Path | None = None,
    on_line: Callable[[str], None] | None = None,
    on_redraw: Callable[[str], None] | None = None,
    check: bool = True,
) -> int:
    """Run ``nightflash-train`` with notebook-safe live output.

    Parameters
    ----------
    args:
        Training flags, without the leading ``nightflash-train``.
    root:
        Directory to run in. Use this when the package is not pip-installed.
    log:
        Optional path to also write a full transcript of the run.
    on_line, on_redraw:
        Optional callbacks if you want to drive your own display widgets.
    """

    command = [sys.executable, "-m", "nightflash.train", *args, "--progress", "plain"]
    return stream_process(
        command,
        cwd=root,
        log_path=log,
        display="line",  # one line per update; never "\r"
        on_line=on_line,
        on_redraw=on_redraw,
        check=check,
    )


def show_previews(run_dir: str | Path, n: int = 6, width: int = 768) -> list[str]:
    """Display the most recent ``source | generated | target`` previews."""

    from IPython.display import Image, display  # type: ignore[import-not-found]

    previews = sorted(
        (Path(run_dir) / "previews").glob("*.png"),
        key=lambda p: p.stat().st_mtime,
    )
    if not previews:
        print("no previews yet")
        return []
    shown: list[str] = []
    for path in previews[-n:]:
        print(path.name)
        display(Image(filename=str(path), width=width))
        shown.append(str(path))
    return shown


def show_image(path: str | Path, width: int = 640) -> None:
    """Display a single image file."""

    from IPython.display import Image, display  # type: ignore[import-not-found]

    display(Image(filename=str(path), width=width))


def run_command(
    args: Sequence[str],
    *,
    root: str | Path = ".",
    log: str | Path | None = None,
    check: bool = True,
) -> int:
    """Stream any ``python -m nightflash.*`` command notebook-safely."""

    command = [sys.executable, "-m", *args]
    return stream_process(command, cwd=root, log_path=log, display="line", check=check)


def last_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the newest ``*.pt`` checkpoint in a run directory, if any."""

    checkpoints = sorted(
        (Path(run_dir) / "checkpoints").glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None
