"""Progress reporting that works in terminals *and* Jupyter/Colab.

Why this exists
---------------
``tqdm`` draws its bar with ``\\r`` carriage returns on **stderr**. In a plain
terminal that is correct and cheap. In a Jupyter/Colab kernel it causes two
visible problems:

1. The bar often does not appear at all, because the notebook frontend does not
   render stderr carriage-return redraws.
2. When output is captured through a pipe (for example ``subprocess.Popen``),
   Python's *universal newlines* translation rewrites every ``\\r`` into ``\\n``.
   Each redraw of the bar then becomes its own printed line, flooding the cell
   with dozens of near-identical lines per epoch.

The fix is to not emit ``\\r`` at all when we are not attached to a real
terminal. :func:`create_reporter` detects that situation and returns a reporter
that prints one complete, newline-terminated line per update instead.

So the same command works unchanged in both places::

    python -m nightflash.train ...            # live bar in a terminal
    python -m nightflash.train ...            # clean lines in Colab
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from typing import Any, TextIO

_ANSI_REPLACEMENTS = (
    ("\r\n", "\n"),
    ("\r", "\n"),
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colour, cursor moves) from ``text``."""

    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\x1b" and index + 1 < length:
            nxt = text[index + 1]
            if nxt == "[":  # CSI ... final byte
                cursor = index + 2
                while cursor < length and not ("\x40" <= text[cursor] <= "\x7e"):
                    cursor += 1
                index = cursor + 1
                continue
            if nxt == "]":  # OSC ... BEL or ST
                cursor = index + 2
                while cursor < length and text[cursor] != "\x07":
                    if text[cursor] == "\x1b" and cursor + 1 < length and text[cursor + 1] == "\\":
                        cursor += 2
                        break
                    cursor += 1
                index = cursor + 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def in_notebook() -> bool:
    """True when running inside an IPython kernel (Jupyter, Colab, VS Code)."""

    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    return type(shell).__name__ in {"ZMQInteractiveShell", "Shell", "TerminalInteractiveShell"}


def default_progress_mode() -> str:
    """Pick ``tqdm`` for real terminals and ``plain`` everywhere else."""

    if in_notebook():
        return "plain"
    try:
        if sys.stderr.isatty() and sys.stdout.isatty():
            return "tqdm"
    except Exception:
        pass
    return "plain"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_metrics(metrics: dict[str, float], keys: tuple[str, ...]) -> str:
    return " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


DISPLAY_KEYS: tuple[str, ...] = ("loss_d", "loss_g", "adv", "content", "identity")


class ProgressReporter(ABC):
    """Minimal progress interface used by the training loop."""

    def __init__(self, log_every: int = 25) -> None:
        self.log_every = max(1, int(log_every))
        self._metrics: dict[str, float] = {}
        self._count = 0

    @abstractmethod
    def start_epoch(self, epoch: int, epochs: int, total_steps: int) -> None:
        """Called once at the beginning of each epoch."""

    @abstractmethod
    def update(self, step_in_epoch: int, metrics: dict[str, float]) -> None:
        """Called after every optimizer step. Must not emit ``\\r``."""

    @abstractmethod
    def finish_epoch(self, epoch: int, epochs: int, elapsed: float, metrics: dict[str, float]) -> None:
        """Called once at the end of each epoch."""

    def close(self) -> None:
        return None

    def _accumulate(self, metrics: dict[str, float]) -> dict[str, float]:
        for key, value in metrics.items():
            self._metrics[key] = self._metrics.get(key, 0.0) + float(value)
        self._count += 1
        return dict(self._metrics)

    def _reset(self) -> dict[str, float]:
        averaged = {
            key: value / max(self._count, 1) for key, value in self._metrics.items()
        }
        self._metrics.clear()
        self._count = 0
        return averaged


class PlainReporter(ProgressReporter):
    """Newline-terminated progress lines. Safe for notebooks and pipes.

    Never writes a carriage return, so universal-newlines translation cannot
    inflate one bar update into a separate printed line.
    """

    def __init__(
        self,
        log_every: int = 25,
        stream: TextIO | None = None,
        show_eta: bool = True,
    ) -> None:
        super().__init__(log_every=log_every)
        self.stream = stream if stream is not None else sys.stdout
        self.show_eta = show_eta
        self._total = 0
        self._epochs = 0
        self._epoch = 0
        self._started: float | None = None
        self._epoch_started: float | None = None

    def start_epoch(self, epoch: int, epochs: int, total_steps: int) -> None:
        self._epoch, self._epochs, self._total = epoch, epochs, total_steps
        self._metrics.clear()
        self._count = 0
        now = time.time()
        self._epoch_started = now
        if self._started is None:
            self._started = now
        print(
            f"epoch {epoch}/{epochs} started "
            f"({total_steps} steps, log every {self.log_every})",
            file=self.stream,
            flush=True,
        )

    def _eta(self, step_in_epoch: int) -> str:
        if self._epoch_started is None or step_in_epoch <= 0:
            return ""
        per_step = (time.time() - self._epoch_started) / step_in_epoch
        remaining = max(0, self._total - step_in_epoch) * per_step
        return f" eta={format_duration(remaining)}"

    def update(self, step_in_epoch: int, metrics: dict[str, float]) -> None:
        self._accumulate(metrics)
        if step_in_epoch % self.log_every != 0 and step_in_epoch != self._total:
            return
        averaged = self._reset()
        percent = 100.0 * step_in_epoch / max(self._total, 1)
        eta = self._eta(step_in_epoch) if self.show_eta else ""
        print(
            f"  [{percent:5.1f}%] epoch {self._epoch}/{self._epochs} "
            f"step {step_in_epoch}/{self._total}{eta}  "
            f"{format_metrics(averaged, DISPLAY_KEYS)}",
            file=self.stream,
            flush=True,
        )

    def finish_epoch(self, epoch: int, epochs: int, elapsed: float, metrics: dict[str, float]) -> None:
        averaged = self._reset() if self._count else metrics
        print(
            f"epoch {epoch}/{epochs} complete in {format_duration(elapsed)}  "
            f"{format_metrics(averaged, DISPLAY_KEYS)}",
            file=self.stream,
            flush=True,
        )


class TqdmReporter(ProgressReporter):
    """Live ``tqdm`` bar. Only used when attached to a real terminal."""

    def __init__(self, log_every: int = 25, stream: TextIO | None = None) -> None:
        super().__init__(log_every=log_every)
        from tqdm import tqdm  # imported lazily so plain mode has no dependency

        self._tqdm = tqdm
        self.stream = stream if stream is not None else sys.stderr
        self._bar: Any = None

    def start_epoch(self, epoch: int, epochs: int, total_steps: int) -> None:
        self._metrics.clear()
        self._count = 0
        if self._bar is not None:
            self._bar.close()
        self._bar = self._tqdm(
            total=total_steps,
            desc=f"epoch {epoch}/{epochs}",
            file=self.stream,
            leave=False,
            dynamic_ncols=True,
        )

    def update(self, step_in_epoch: int, metrics: dict[str, float]) -> None:
        self._accumulate(metrics)
        assert self._bar is not None
        self._bar.update(1)
        if step_in_epoch % self.log_every == 0 and self._count >= self.log_every:
            averaged = self._reset()
            self._bar.set_postfix_str(format_metrics(averaged, DISPLAY_KEYS), refresh=False)

    def finish_epoch(self, epoch: int, epochs: int, elapsed: float, metrics: dict[str, float]) -> None:
        averaged = self._reset() if self._count else metrics
        if self._bar is not None:
            self._bar.set_postfix_str(format_metrics(averaged, DISPLAY_KEYS), refresh=True)
            self._bar.close()
            self._bar = None

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def create_reporter(
    mode: str = "auto",
    log_every: int = 25,
    stream: TextIO | None = None,
) -> ProgressReporter:
    """Build a reporter.

    Parameters
    ----------
    mode:
        ``auto`` (default), ``tqdm``, ``plain``, or ``none``.
        ``auto`` picks ``tqdm`` only when stdout and stderr are both real TTYs
        and we are not inside a notebook.
    """

    resolved = default_progress_mode() if mode == "auto" else mode
    if resolved == "none":
        return _NullReporter(log_every=log_every)
    if resolved == "tqdm":
        return TqdmReporter(log_every=log_every, stream=stream)
    return PlainReporter(log_every=log_every, stream=stream)


class _NullReporter(ProgressReporter):
    """Discards progress entirely. Useful for quiet scripted runs."""

    def start_epoch(self, epoch: int, epochs: int, total_steps: int) -> None:
        return None

    def update(self, step_in_epoch: int, metrics: dict[str, float]) -> None:
        return None

    def finish_epoch(self, epoch: int, epochs: int, elapsed: float, metrics: dict[str, float]) -> None:
        return None
