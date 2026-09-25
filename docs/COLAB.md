# Google Colab workflow

Google Colab is suitable for training this compact model. It is not a good place to run the interactive TUI because Colab notebooks do not provide a persistent terminal UI.

## 1. Create a runtime

Choose:

```text
Runtime → Change runtime type → GPU
```

A T4 is usually enough for a pilot. An L4 or A100 is preferable for 512px training.

## 2. Mount Google Drive

Run this cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Use a dedicated directory:

```text
/content/drive/MyDrive/nightflash-edge/
```

## 3. Get the project

If the project is in a Git repository:

```python
!git clone YOUR_REPOSITORY_URL /content/nightflash-edge
```

If you are uploading it manually, upload the project folder or a zip and unzip it:

```python
!unzip -q /content/nightflash-edge.zip -d /content
```

Then:

```python
%cd /content/nightflash-edge
```

## 4. Install

```python
!python -m pip install -U pip
!python -m pip install -e .
```

The default training dependencies are enough. The TUI is not needed in Colab.

## 5. Check the GPU

```python
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## 6. Prepare the dataset

Recommended Drive layout:

```text
MyDrive/nightflash-edge/
  data/
    train/
      source/
      target/
    val/
      source/
      target/
```

Set paths:

```python
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/nightflash-edge")
DATA = ROOT / "data"
RUNS = ROOT / "runs"

SOURCE = DATA / "train" / "source"
TARGET = DATA / "train" / "target"
MASK_DIR = None  # set to DATA / "train" / "source_masks" if available
```

You do not need paired filenames or aligned images.

## 7. Run a smoke test

```python
!python scripts/smoke_test.py --workdir /content/nightflash-smoke
```

If the smoke test passes, the model, effects, checkpointing, and inference path are working.

## 8. Train a short pilot

Start with a short run so you can inspect the result. In a notebook, use Python's `subprocess` so the `Path` objects are passed correctly:

```python
from pathlib import Path
import subprocess

source = DATA / "train" / "source"
target = DATA / "train" / "target"
output = RUNS / "conv-pilot"

subprocess.run([
    "python", "-m", "nightflash.train",
    "--source-dir", str(source),
    "--target-dir", str(target),
    "--output-dir", str(output),
    "--image-size", "256",
    "--batch-size", "8",
    "--epochs", "10",
    "--no-transformer",
    "--preview-every", "100",
    "--log-every", "25",
    "--amp",
], check=True)
```

If you prefer a shell command, define shell variables first:

```python
!export SOURCE=/content/drive/MyDrive/nightflash-edge/data/train/source
!export TARGET=/content/drive/MyDrive/nightflash-edge/data/train/target
!export RUNS=/content/drive/MyDrive/nightflash-edge/runs
!python -m nightflash.train --source-dir "$SOURCE" --target-dir "$TARGET" --output-dir "$RUNS/conv-pilot" --image-size 256 --batch-size 8 --epochs 10 --no-transformer --preview-every 100 --log-every 25 --amp
```

The Python `subprocess` form is less error-prone.

## 9. Test a mid-training checkpoint

The training command writes:

```text
runs/conv-pilot/checkpoints/latest.pt
runs/conv-pilot/previews/step-*.png
```

After a checkpoint exists, run inference from another cell:

```python
checkpoint = RUNS / "conv-pilot" / "checkpoints" / "latest.pt"
test_image = DATA / "val" / "source" / "example.jpg"

subprocess.run([
    "python", "-m", "nightflash.infer",
    "--checkpoint", str(checkpoint),
    "--input", str(test_image),
    "--output", str(RUNS / "conv-pilot" / "mid-training.png"),
    "--seed", "12345",
    "--device", "cuda",
], check=True)
```

If you want to inspect a mid-training state without waiting for the full run, use a step limit:

```python
subprocess.run([
    "python", "-m", "nightflash.train",
    "--source-dir", str(source),
    "--target-dir", str(target),
    "--output-dir", str(RUNS / "step-test"),
    "--image-size", "256",
    "--batch-size", "8",
    "--epochs", "1",
    "--max-steps", "1000",
    "--preview-every", "250",
    "--amp",
], check=True)
```

Then infer from:

```text
runs/step-test/checkpoints/latest.pt
```

Resume the longer run from that checkpoint:

```python
subprocess.run([
    "python", "-m", "nightflash.train",
    "--source-dir", str(source),
    "--target-dir", str(target),
    "--output-dir", str(RUNS / "step-test"),
    "--resume", str(RUNS / "step-test" / "checkpoints" / "latest.pt"),
    "--image-size", "256",
    "--batch-size", "8",
    "--epochs", "30",
    "--preview-every", "500",
    "--amp",
], check=True)
```

## 10. Enable the edge transformer

After the convolutional baseline works:

```python
subprocess.run([
    "python", "-m", "nightflash.train",
    "--source-dir", str(source),
    "--target-dir", str(target),
    "--output-dir", str(RUNS / "hybrid-pilot"),
    "--image-size", "256",
    "--batch-size", "4",
    "--epochs", "20",
    "--transformer-blocks", "1",
    "--transformer-token-size", "16",
    "--preview-every", "500",
    "--amp",
], check=True)
```

Do not enable the transformer and change the dataset, resolution, and all loss weights simultaneously. Compare one change at a time.

## 11. Colab storage rules

Store these on Drive:

```text
checkpoints/
previews/
run_config.json
evaluation.json
```

Do not rely on `/content` after the runtime ends.

Keep at least:

```text
latest.pt
best-known-good.pt
```

The current trainer saves `latest.pt` and epoch checkpoints. Copy a known-good checkpoint to a separate name before starting a new experiment.

## 12. TUI in Colab

The Textual TUI is intended for a local terminal:

```bash
nightflash-tui
```

It is not useful inside a Colab notebook. Use the Python/CLI commands for Colab.

## 13. Export Core ML

Core ML conversion should be done on macOS with `coremltools` installed, not in a typical Colab runtime.

Download these from Drive:

```text
checkpoint .pt
```

Then on the Mac:

```bash
python -m pip install -e '.[coreml]'
nightflash-export-coreml \
  --checkpoint /path/to/latest.pt \
  --output /path/to/NightFlashEdge.mlpackage \
  --image-size 256 \
  --half
```

See [`COREML.md`](COREML.md).
