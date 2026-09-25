# Terminal UI

Install the optional Textual dependency:

```bash
python -m pip install -e '.[tui]'
nightflash-tui
```

The TUI provides:

- Source/target/mask path fields
- Image size, batch size, epoch, worker, and device fields
- Convolutional vs hybrid-transformer toggle
- Path validation
- Start/stop training subprocess
- Live stdout/stderr log
- Open latest preview
- Launch Core ML export

The TUI is intended for a local terminal. Use the CLI commands in Colab.

The UI launches training as a subprocess, so stopping the UI does not necessarily kill a detached process. The Stop button terminates the active subprocess. If a process is interrupted, resume from:

```text
<output-dir>/checkpoints/latest.pt
```
