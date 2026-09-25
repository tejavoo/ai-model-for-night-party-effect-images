from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Log, Static

    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    App = object  # type: ignore[assignment,misc]
    ComposeResult = object  # type: ignore[assignment,misc]
    Horizontal = Vertical = Button = Checkbox = Footer = Header = Input = Label = Log = Static = object
    TEXTUAL_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "nightflash"


class NightFlashTUI(App):  # type: ignore[misc]
    """Interactive training dashboard for local terminals."""

    CSS = """
    Screen { background: #10131a; color: #e8eaf0; }
    #body { padding: 1 2; }
    #intro { color: #9ca3af; margin-bottom: 1; }
    #form { height: auto; border: round #3b4252; padding: 1; }
    .field { height: 3; }
    .field Label { width: 18; padding: 1 0; }
    .field Input { width: 1fr; }
    #actions { height: 3; margin-top: 1; }
    #actions Button { margin-right: 1; }
    #status { margin: 1 0; color: #8ecdf8; }
    #log { height: 1fr; border: round #3b4252; background: #090b10; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header()
        with Vertical(id="body"):
            yield Static("Train a compact unpaired Night Flash model, preview it live, and export Core ML.", id="intro")
            with Vertical(id="form"):
                with Horizontal(classes="field"):
                    yield Label("Source images")
                    yield Input(value="", placeholder="/path/to/normal/photos", id="source")
                with Horizontal(classes="field"):
                    yield Label("Target images")
                    yield Input(value="", placeholder="/path/to/disposable/photos", id="target")
                with Horizontal(classes="field"):
                    yield Label("Mask directory")
                    yield Input(value="", placeholder="optional person masks", id="mask")
                with Horizontal(classes="field"):
                    yield Label("Output directory")
                    yield Input(value=str(DEFAULT_OUTPUT), id="output")
                with Horizontal(classes="field"):
                    yield Label("Image size")
                    yield Input(value="256", id="size")
                with Horizontal(classes="field"):
                    yield Label("Batch size")
                    yield Input(value="4", id="batch")
                with Horizontal(classes="field"):
                    yield Label("Epochs")
                    yield Input(value="50", id="epochs")
                with Horizontal(classes="field"):
                    yield Label("Workers")
                    yield Input(value="2", id="workers")
                with Horizontal(classes="field"):
                    yield Label("Device")
                    yield Input(value="auto", id="device")
                with Horizontal(classes="field"):
                    yield Label("Edge transformer")
                    yield Checkbox("Enable MobileViT/EdgeViT bottleneck", value=True, id="transformer")
            with Horizontal(id="actions"):
                yield Button("Validate", id="validate", variant="primary")
                yield Button("Start training", id="start", variant="success")
                yield Button("Stop", id="stop", variant="warning")
                yield Button("Open latest preview", id="preview")
                yield Button("Export Core ML", id="export")
            yield Static("Ready.", id="status")
            yield Log(id="log", highlight=False)
        yield Footer()

    def _value(self, widget_id: str) -> str:
        return str(self.query_one(f"#{widget_id}", Input).value).strip()

    def _status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _log(self, message: str) -> None:
        self.query_one("#log", Log).write_line(message)

    def _base_command(self) -> list[str]:
        source = self._value("source")
        target = self._value("target")
        if not source or not target:
            raise ValueError("Source and target directories are required")
        output = self._value("output") or str(DEFAULT_OUTPUT)
        size = int(self._value("size") or 256)
        batch = int(self._value("batch") or 4)
        epochs = int(self._value("epochs") or 50)
        workers = int(self._value("workers") or 2)
        device = self._value("device") or "auto"
        command = [
            sys.executable,
            "-m",
            "nightflash.train",
            "--source-dir",
            source,
            "--target-dir",
            target,
            "--output-dir",
            output,
            "--image-size",
            str(size),
            "--batch-size",
            str(batch),
            "--epochs",
            str(epochs),
            "--num-workers",
            str(workers),
            "--device",
            device,
        ]
        mask = self._value("mask")
        if mask:
            command.extend(["--mask-dir", mask])
        if not self.query_one("#transformer", Checkbox).value:
            command.append("--no-transformer")
        return command

    def _spawn(self, command: list[str], label: str) -> None:
        if self.process is not None and self.process.poll() is None:
            self._log(f"{label} is already running")
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        self._log("$ " + " ".join(command))
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._status(f"{label} running (PID {self.process.pid})")
        self.thread = threading.Thread(target=self._read_process, daemon=True)
        self.thread.start()

    def _read_process(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.call_from_thread(self._log, line.rstrip())
        return_code = process.wait()
        self.call_from_thread(self._on_process_exit, return_code)

    def _on_process_exit(self, return_code: int) -> None:
        self.process = None
        self._status(f"Process exited with code {return_code}")

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        try:
            if event.button.id == "validate":
                source = Path(self._value("source"))
                target = Path(self._value("target"))
                if not source.is_dir():
                    raise ValueError(f"Source directory does not exist: {source}")
                if not target.is_dir():
                    raise ValueError(f"Target directory does not exist: {target}")
                self._log(f"Source OK: {source}")
                self._log(f"Target OK: {target}")
                self._status("Paths validated")
            elif event.button.id == "start":
                self._spawn(self._base_command(), "Training")
            elif event.button.id == "stop":
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    self._status("Stop requested")
                else:
                    self._status("No process is running")
            elif event.button.id == "preview":
                output = Path(self._value("output") or str(DEFAULT_OUTPUT))
                previews = sorted((output / "previews").glob("*.png"), key=lambda p: p.stat().st_mtime)
                if not previews:
                    self._status("No preview found yet; start training first")
                else:
                    latest = previews[-1]
                    self._log(f"Opening {latest}")
                    webbrowser.open(latest.resolve().as_uri())
            elif event.button.id == "export":
                output = Path(self._value("output") or str(DEFAULT_OUTPUT))
                checkpoint = output / "checkpoints" / "latest.pt"
                if not checkpoint.exists():
                    self._status("No latest.pt checkpoint found")
                    return
                command = [
                    sys.executable,
                    "-m",
                    "nightflash.export_coreml",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output / "NightFlashEdge.mlpackage"),
                    "--image-size",
                    self._value("size") or "256",
                ]
                self._spawn(command, "Core ML export")
        except Exception as exc:  # show errors in the UI rather than killing it
            self._status(f"Error: {exc}")
            self._log(f"Error: {exc}")


def main() -> None:
    if not TEXTUAL_AVAILABLE:
        print("The TUI requires Textual. Install it with:")
        print("  python -m pip install -e '.[tui]'")
        print("Then run:")
        print("  nightflash-tui")
        return
    NightFlashTUI().run()


if __name__ == "__main__":
    main()
