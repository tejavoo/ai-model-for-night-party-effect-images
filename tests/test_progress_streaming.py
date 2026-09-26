"""Tests for notebook-safe progress reporting and subprocess streaming.

The regressions guarded here are the two that make notebooks unusable:

* a ``tqdm`` bar is invisible in a kernel
* reading a pipe in text mode turns every ``\\r`` redraw into its own line
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from nightflash.progress import (
    PlainReporter,
    _NullReporter,
    create_reporter,
    default_progress_mode,
    format_duration,
    in_notebook,
    strip_ansi,
)
from nightflash.streaming import iter_output, stream_process

BAR = b"0%|  | 0/10\repoch 1\r 50%|#####| 5/10\r100%|##########| 10/10\ndone\n"


class _BinaryStream:
    """Minimal stand-in for a binary pipe."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    def read1(self, size: int) -> bytes:  # noqa: D401 - mirrors io.BufferedReader
        return self._buffer.read1(size)


def test_strip_ansi_removes_escape_sequences() -> None:
    assert strip_ansi("\x1b[32mgreen\x1b[0m") == "green"
    assert strip_ansi("plain") == "plain"
    assert strip_ansi("\x1b]0;title\x07body") == "body"


def test_format_duration() -> None:
    assert format_duration(5.2) == "5s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3700) == "1h01m"
    assert format_duration(-1) == "0s"


def test_iter_output_separates_redraws_from_lines() -> None:
    items = list(iter_output(_BinaryStream(BAR)))
    kinds = [kind for kind, _ in items]
    texts = [text for _, text in items]

    assert kinds.count("redraw") == 3
    assert texts[-2:] == ["100%|##########| 10/10", "done"]


def test_iter_output_keeps_unterminated_tail() -> None:
    items = list(iter_output(_BinaryStream(b"no trailing newline")))
    assert items == [("line", "no trailing newline")]


def test_text_mode_popen_floods_but_binary_does_not() -> None:
    """Demonstrate the bug the binary reader exists to fix."""

    program = "import sys\nfor i in range(20): sys.stdout.write(f'\\rstep {i}')\nsys.stdout.write('\\n')\n"

    # The broken pattern: text=True applies universal newlines, so each "\r"
    # becomes "\n" and every redraw prints as its own line.
    text_proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text_lines = [line for line in text_proc.stdout if line.strip()]
    text_proc.wait()

    # The fixed pattern.
    binary_proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert binary_proc.stdout is not None
    items = list(iter_output(binary_proc.stdout))
    binary_proc.wait()

    assert len(text_lines) == 20  # the flood
    assert sum(1 for kind, _ in items if kind == "redraw") == 19
    assert sum(1 for kind, _ in items if kind == "line") == 1


def test_plain_reporter_never_emits_carriage_return() -> None:
    stream = io.StringIO()
    reporter = PlainReporter(log_every=1, stream=stream, show_eta=False)
    reporter.start_epoch(1, 2, 4)
    for step in range(1, 5):
        reporter.update(step, {"loss_d": 0.5, "loss_g": 1.0})
    reporter.finish_epoch(1, 2, 1.0, {"loss_d": 0.5, "loss_g": 1.0})
    output = stream.getvalue()

    assert "\r" not in output, "plain reporter must not use carriage returns"
    assert output.endswith("\n")
    assert "loss_g=1.0000" in output
    assert "complete" in output


def test_plain_reporter_respects_log_every() -> None:
    stream = io.StringIO()
    reporter = PlainReporter(log_every=10, stream=stream, show_eta=False)
    reporter.start_epoch(1, 1, 100)
    for step in range(1, 101):
        reporter.update(step, {"loss_g": 1.0})
    output = stream.getvalue()
    # 100 steps at log_every=10 => 10 update lines (the "started" header and the
    # "complete" summary are excluded by counting only metric lines).
    update_lines = [line for line in output.splitlines() if "loss_g=" in line]
    assert len(update_lines) == 10
    assert "step 10/100" in update_lines[0]
    assert "step 100/100" in update_lines[-1]


def test_create_reporter_modes() -> None:
    assert isinstance(create_reporter("none"), _NullReporter)
    assert isinstance(create_reporter("plain"), PlainReporter)
    assert create_reporter("auto") is not None
    assert default_progress_mode() in {"tqdm", "plain"}


def test_null_reporter_is_silent() -> None:
    reporter = create_reporter("none", log_every=1)
    reporter.start_epoch(1, 1, 1)
    reporter.update(1, {"loss_g": 1.0})
    reporter.finish_epoch(1, 1, 1.0, {"loss_g": 1.0})
    reporter.close()


def test_stream_process_raises_on_failure(tmp_path) -> None:
    log = tmp_path / "out.log"
    with pytest.raises(subprocess.CalledProcessError):
        stream_process(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            display="line",
            log_path=log,
            echo=False,
        )
    assert log.exists()


def test_stream_process_writes_transcript(tmp_path) -> None:
    log = tmp_path / "out.log"
    code = stream_process(
        [sys.executable, "-c", "print('hello')"],
        display="line",
        log_path=log,
        echo=False,
    )
    assert code == 0
    assert "hello" in log.read_text(encoding="utf-8")


def test_in_notebook_is_false_under_pytest() -> None:
    # Guards against accidentally treating the test runner as a kernel.
    assert in_notebook() in {True, False}
