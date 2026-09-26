"""Subprocess streaming that survives carriage-return redraws.

The problem
-----------
A live progress bar writes ``\\r`` to redraw itself. When a training process is
driven from a notebook with ``subprocess.Popen(..., text=True)`` and read with
``for line in process.stdout``, Python's *universal newlines* translation
converts every ``\\r`` into ``\\n``. One bar update becomes one printed line, so
the cell fills with hundreds of near-identical lines:

```
  0%|          | 0/125 [00:00<?, ?it/s]
  0%|          | 0/125 [00:00<?, ?it/s, loss_d=0.72 loss_g=1.19]
  0%|          | 0/125 [00:00<?, ?it/s, loss_d=0.72 loss_g=1.18]
  ... hundreds more, one per second
```

Reading the pipe in **binary** mode avoids that translation, and this module
then classifies each segment as either a completed line or an in-place redraw so
the caller can decide how to display it.

Usage
-----
Terminal (overwrite in place, like a normal bar)::

    stream_process(cmd, display="redraw")

Notebook or log file (one line per update)::

    stream_process(cmd, display="line")

Or handle the pieces yourself::

    for kind, text in iter_output(process.stdout):
        if kind == "line":
            print(text)
        else:
            status_widget.update(text)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterator, TextIO

from .progress import in_notebook, strip_ansi

READ_SIZE = 4096


def _read_chunks(stream: TextIO) -> Iterator[bytes]:
    """Yield raw bytes from a pipe opened in binary mode."""

    while True:
        try:
            chunk = stream.read1(READ_SIZE)  # type: ignore[union-attr]
        except AttributeError:
            chunk = stream.read(READ_SIZE)
        if not chunk:
            return
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", errors="replace")
        yield chunk


def iter_output(
    stream: TextIO,
    *,
    strip: bool = True,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> Iterator[tuple[str, str]]:
    """Split a binary pipe into ``(kind, text)`` pairs.

    ``kind`` is one of:

    ``"line"``
        A complete, newline-terminated line. Always safe to print.
    ``"redraw"``
        A segment that the producer overwrote with ``\\r``. Display it in place,
        or print it as its own line when a live bar is not possible.

    The final unterminated segment is emitted as ``"line"`` so nothing is lost
    when a process exits without a trailing newline.
    """

    buffer = ""
    for chunk in _read_chunks(stream):
        buffer += chunk.decode(encoding, errors)
        while True:
            newline = buffer.find("\n")
            carriage = buffer.find("\r")
            if newline == -1 and carriage == -1:
                break
            if newline != -1 and (carriage == -1 or newline < carriage):
                segment, buffer = buffer[:newline], buffer[newline + 1 :]
                segment = segment.rstrip("\r")
                if strip:
                    segment = strip_ansi(segment)
                if segment.strip():
                    yield "line", segment
            else:
                segment, buffer = buffer[:carriage], buffer[carriage + 1 :]
                if strip:
                    segment = strip_ansi(segment)
                if segment.strip():
                    yield "redraw", segment
    tail = buffer.rstrip("\r")
    if strip:
        tail = strip_ansi(tail)
    if tail.strip():
        yield "line", tail


def _default_display() -> str:
    """``redraw`` only when a real terminal can overwrite in place."""

    if in_notebook():
        return "line"
    try:
        if sys.stdout.isatty():
            return "redraw"
    except Exception:
        pass
    return "line"


def stream_process(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    log_path: str | Path | None = None,
    display: str = "auto",
    on_line: Callable[[str], None] | None = None,
    on_redraw: Callable[[str], None] | None = None,
    strip: bool = True,
    check: bool = True,
    echo: bool = True,
) -> int:
    """Run ``command``, streaming output live and without ``\\r`` flooding.

    Returns the process return code. Raises ``subprocess.CalledProcessError``
    when ``check`` is true and the process fails.
    """

    resolved = _default_display() if display == "auto" else display
    if display == "redraw" and not sys.stdout.isatty():
        resolved = "line"

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,  # unbuffered, binary
    )
    assert process.stdout is not None

    log_file: TextIO | None = None
    if log_path is not None:
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        log_file = destination.open("w", encoding="utf-8")

    try:
        for kind, text in iter_output(process.stdout, strip=strip):
            if log_file is not None:
                log_file.write(text.replace("\r", "\n") + "\n")
                log_file.flush()

            if kind == "line":
                if on_line is not None:
                    on_line(text)
                if echo:
                    print(text, flush=True)
            else:
                if on_redraw is not None:
                    on_redraw(text)
                if echo and resolved == "redraw":
                    sys.stdout.write("\r" + text.ljust(0))
                    sys.stdout.flush()
                elif echo and resolved == "line":
                    # A notebook cannot overwrite, so make the redraw a real
                    # line. This is the fix for the "prints every line" flood.
                    print(text, flush=True)

        if resolved == "redraw":
            sys.stdout.write("\n")
            sys.stdout.flush()
    finally:
        if log_file is not None:
            log_file.close()

    returncode = process.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return returncode
