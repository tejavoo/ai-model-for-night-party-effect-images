# Terminal UI

Install the optional Textual dependency:

```bash
python -m pip install -e '.[tui]'
nightflash-tui
```

The TUI provides:

- Source/target/mask path fields
- Image size, batch size, epoch, worker, and device fields
- An optional **Resume checkpoint** field
- Convolutional vs hybrid-transformer toggle
- Path validation
- Start/stop training subprocess
- Live stdout/stderr log
- Progress redraws shown in the status line instead of flooding the log
- Open latest preview
- Launch Core ML export

The TUI is intended for a local terminal. Use the CLI commands in Colab.

### Resuming from the TUI

Paste a checkpoint path into **Resume checkpoint** and press **Start training**.
The architecture toggles must match the run being resumed — leave
**Edge transformer** enabled for a hybrid run, disabled for a conv-only run.
Mismatching them fails fast with a state-dict error rather than silently
producing a different model.

Remember that `--epochs` is an absolute target. If the checkpoint is at epoch 40
and you enter 40, nothing trains and the run exits with an explicit error.

### Progress handling

The TUI launches the trainer with `--progress tqdm` and then reads the pipe in
**binary** mode via `nightflash.streaming.iter_output`. Complete lines are
appended to the scrollable log; `\r` redraws update the single status line. This
is the same mechanism that stops a notebook cell from being flooded with one
printed line per bar refresh — see [`TRAINING.md`](TRAINING.md#progress-output-in-notebooks).

The UI launches training as a subprocess, so stopping the UI does not necessarily kill a detached process. The Stop button terminates the active subprocess. If a process is interrupted, resume from:

```text
<output-dir>/checkpoints/latest.pt
```
