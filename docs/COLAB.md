# Google Colab workflow

Google Colab is suitable for training this compact model. It is **not** a good place to run the interactive TUI, because Colab notebooks do not provide a persistent terminal UI.

The Kaggle quickstart ([`QuickStartKaggle.md`](QuickStartKaggle.md)) covers the
whole flow end to end, including both-GPU setup and the command cookbook. This
document is Colab-specific: paths, Drive persistence, the Colab CLI, and the
Colab MCP.

---

## Choose your path

There are three supported ways to drive Colab. All of them end up running the
same training code; path A drives it through a `Session` object, paths B and C
through the `nf` command or the legacy `nightflash-train`.

| Path | Use when | Docs |
|---|---|---|
| **A. Notebook** (recommended) | You want to watch training, inspect previews inline, iterate interactively | [`colab/NightFlash_Colab.ipynb`](../colab/NightFlash_Colab.ipynb) |
| **B. Colab CLI** | You want to script it from your Mac terminal, or run unattended overnight jobs | [Colab CLI](#colab-cli-terminal-workflow) below |
| **C. Colab MCP** | You want your coding agent to drive the Colab session for you | [Colab MCP](#colab-mcp-agent-workflow) below |

The notebook is **seven cells**: install, data, configuration, train, inspect,
keep. Only cell 3 needs editing.

Do not use the TUI in Colab. Use `nightflash-tui` on your Mac instead.

### What each path assumes about your data

Training data does **not** travel with the code. In every path, put `training_photos/` on Google Drive:

```text
MyDrive/nightflash-edge/
  training_photos/
    normal_photos/       # source domain
    disposable_photos/   # target domain
    validation_photos/   # held-out source images
```

### ⚠️ Current dataset imbalance

| Domain | Count | Status |
|---|---:|---|
| `normal_photos` (source) | 1000 | fine |
| `disposable_photos` (target) | 935 | **wrong domain, not just small — see below** |
| `validation_photos` (source) | 20 | fine |
| held-out target | 0 | **target quality is unmeasurable** |

This is enough to prove the pipeline runs end to end. It is not enough to judge
model quality.

The target folder has 935 images but they are not night-flash photographs.
Measured mean luminance is 0.350 (genuine night flash is below 0.20); only 9.3%
of the folder is in the target range and 45.2% is daylit above 0.35. There is
also no `manifest.csv`, so licensing is unrecorded.

A PatchGAN learns whatever distribution the target folder contains, so training
on this teaches a slightly hazy daylight look rather than Night Flash. Fix the
folder before drawing any conclusion from the run — see
[`DATASET_SOURCING.md`](DATASET_SOURCING.md) and
[`QuickStartKaggle.md`](QuickStartKaggle.md#before-you-start-the-target-domain)
for a copy-pasteable measurement you can run first.

Also hold out ~15 target images and point `--target-dir` at them when
evaluating, otherwise target quality stays unmeasurable.

See [`DATASET_SOURCING.md`](DATASET_SOURCING.md) for how the dataset was built
and its licensing obligations.

---

# Path A — Notebook workflow

The ready-made notebook [`colab/NightFlash_Colab.ipynb`](../colab/NightFlash_Colab.ipynb) is the fastest route. Upload it to Colab, then run the cells top to bottom.

To create a zip of the repo for upload (code + docs only; dataset and checkpoints are excluded):

```bash
./scripts/make_colab_bundle.sh
# → dist/nightflash-edge-colab.zip
```

Then in Colab:

1. Runtime → Change runtime type → **GPU** (T4 for a pilot, L4/A100 for 512px)
2. Open the file browser (folder icon) and upload `nightflash-edge-colab.zip` to `/content/`
3. Upload `training_photos/` to Google Drive
4. Run all cells

The notebook handles code acquisition, install, GPU checks, dataset counting, training, inline preview display, multi-seed inference, evaluation, and checkpoint download.

### Path A alternative — GitHub

If you push this repo to GitHub, the notebook's first code cell accepts a `REPO_URL` instead of a zip. Set:

```python
REPO_URL = "https://github.com/you/nightflash-edge"
REPO_BRANCH = "main"
```

To push from your Mac (this repo currently has no remote):

```bash
cd /Users/tejavu/Documents/opencode-experiments/disposable-cam-model-train
git remote add origin https://github.com/you/nightflash-edge.git
git add -A
git commit -m "Add Colab notebook, dataset sourcing, and training scaffold"
git push -u origin main
```

Add a `.gitignore` check before pushing — `training_photos/`, `runs/`, `dist/`, and `*.pt` are already ignored.

---

# Manual cell workflow

The sections below are the underlying manual steps. Use them if you want to build your own notebook, or if you prefer explicit control. Path A already automates all of it.

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

---

# Colab CLI (terminal workflow)

Run training on a Colab GPU **from your Mac terminal**, without opening a notebook. This is the best option for long unattended runs and for scripting.

This is Google's official CLI: [`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli).

**Platform support:** Linux and macOS. Windows is not supported.

## Install

```bash
# with uv (recommended)
uv tool install google-colab-cli

# or with pip
pip install google-colab-cli

colab version
```

Authentication defaults to Application Default Credentials (`--auth adc`). If you prefer browser OAuth, create an OAuth client in Google Cloud Console with the `https://www.googleapis.com/auth/colaboratory` scope, save it to `~/.colab-cli-oauth-config.json`, and run with `--auth oauth2`.

### Progress output in Colab

Do not drive training with a bare `subprocess.run(cmd)` and expect a live progress
bar. `tqdm` draws with `\r` on stderr, and Python's universal-newlines translation
turns every redraw into a separate printed line, so the cell fills with hundreds of
near-identical lines. The bar is also usually invisible in the first place, because
the kernel does not redraw stderr.

Use the package helper, which forces newline-terminated output and reads the pipe in
binary:

```python
from nightflash.notebook import train, show_previews, last_checkpoint

train([...same flags as the CLI...],
      root="/content/nightflash-edge",
      log=f"{OUTPUT_DIR}/train.log")
show_previews(OUTPUT_DIR)
```

Or force the style directly with `--progress plain | tqdm | none`. The trainer
auto-detects notebooks, so a plain `!python -m nightflash.train ...` shell call is
fine too. Full details in
[`TRAINING.md`](TRAINING.md#progress-output-in-notebooks).

### Resuming and warm starting

`--epochs` is an **absolute target**, not a count of extra epochs. Resuming at or
below the checkpoint's epoch trains nothing; the trainer now exits with an explicit
error instead of reporting false success.

To move from a trained conv model to a hybrid one, `--resume` cannot be used directly
because it is a strict state-dict load. Run the warm-start helper first:

```python
subprocess.run([sys.executable, "/content/scripts/warm_start.py",
                "--checkpoint", f"{RUNS}/conv-pilot/checkpoints/latest.pt",
                "--output-dir",  f"{RUNS}/hybrid-warm",
                "--transformer-blocks", "1"], check=True)
```

Then resume the **new** checkpoint. See [`WARM_START.md`](WARM_START.md).

## Flow 1 — full training run from your Mac (recommended)

This provisions a GPU VM, mounts Drive, installs the project, trains, fetches the checkpoint, and tears the VM down.

```bash
# 1. Package the code on your Mac
./scripts/make_colab_bundle.sh          # → dist/nightflash-edge-colab.zip

# 2. Provision a GPU session
colab new -s nightflash --gpu T4

# 3. Mount Drive (this is the one interactive auth step)
colab drivemount -s nightflash

# 4. Upload the code bundle
colab upload -s nightflash dist/nightflash-edge-colab.zip /content/nightflash-edge-colab.zip

# 5. Upload the dataset (once; ~1000 + 45 + 20 images)
colab upload -s nightflash training_photos /content/training_photos

# 6. Train. colab exec -c takes Python, so we shell out to the job script.
colab exec -s nightflash -c "import subprocess,sys; sys.exit(subprocess.call([sys.executable,'/content/colab_train_job.py','--zip','/content/nightflash-edge-colab.zip','--source-dir','/content/training_photos/normal_photos','--target-dir','/content/training_photos/disposable_photos','--output-dir','/content/runs/cli-pilot','--image-size','256','--batch-size','8','--epochs','30','--no-transformer','--amp']))"

# 7. Fetch the checkpoint back to your Mac
colab download -s nightflash /content/runs/cli-pilot/checkpoints/latest.pt ./latest.pt

# 8. Release the VM (important — keep-alive keeps burning compute units)
colab stop -s nightflash
```

To keep checkpoints across sessions, point `--output-dir` at Drive instead of `/content`:

```text
/content/drive/MyDrive/nightflash-edge/runs/cli-pilot
```

`scripts/colab_train_job.py` is a self-bootstrapping entry point: when run on a VM that does not have the repo yet, it unzips or clones the code and runs `pip install -e .` before training. Use `--dry-run` locally to print the exact plan without doing anything.

## Flow 2 — one-shot ephemeral job

`colab run` provisions a fresh VM, runs a **local script path** with forwarded arguments, and tears the runtime down automatically. Note it takes a script path, not a `-m` module flag, which is why `scripts/colab_train_job.py` exists.

```bash
colab run --gpu T4 --keep scripts/colab_train_job.py -- \
  --repo-url https://github.com/you/nightflash-edge \
  --source-dir /content/data/normal_photos \
  --target-dir /content/data/disposable_photos \
  --output-dir /content/runs/job-1 \
  --image-size 256 --batch-size 8 --epochs 30 --no-transformer --amp
```

**Important caveat:** a fresh ephemeral VM has no interactive Drive mount. Use this flow only when the dataset is already reachable without a browser prompt — for example a public URL, a public bucket, or GCS after running `colab auth` on the VM. If your dataset is only on your personal Drive, use Flow 1 instead.

Add `--keep` to preserve the VM so you can inspect it afterwards:

```bash
colab ls -s NAME /content/runs/job-1/checkpoints
colab url -s NAME --open
```

## Flow 3 — run the bundled notebook on a VM

```bash
colab new -s nb --gpu T4
colab drivemount -s nb
colab exec -s nb -f colab/NightFlash_Colab.ipynb
colab log -s nb -o colab_run.md
colab stop -s nb
```

`colab log` exports the session history as markdown, a notebook, plain text, or JSONL — handy for keeping a record of a run.

The Drive-mount cell in the notebook may block under `colab exec` because Google auth is interactive in a non-browser context. If that happens, either run `colab auth` on the VM first, or use Flow 1.

## Tiled inference on Colab

A 256px model letterboxes a 12-megapixel phone photo to roughly 0.08× scale.
Nothing is cropped or stretched, but high-frequency detail is lost — and detail is
what the halation and `freqsplit` heads exist to produce.

```python
s.infer(photo_path)                    # whole frame
s.infer(photo_path, tile=256)          # near-native resolution, crossfaded
s.infer(photo_path, tile=256, tile_overlap=128)   # smoother, ~2x the cost
s.infer(photo_path, max_edge=1024)     # downscale first, aspect preserved
s.infer(photo_path, max_edge=1024, tile=256)      # downscale AND tile
```

**Match `tile` to the `image_size` you trained at.** A model trained on 256px
crops has learned 256px-crop feature statistics; a different tile size is a
distribution shift, and `Session.infer` warns when they differ.

Measured cost on a 4032×3024 source:

| tile | overlap ratio | tiles | MP processed |
|---:|---:|---:|---:|
| 256 | 25% | 336 | 22.0 |
| 256 | 50% | 713 | 46.7 |
| 512 | 25% | 88 | 23.1 |

What matters is the overlap **ratio**, not its absolute size. Use 25% for speed
on a batch, 50% when quality matters.

Borders do not fade to black: the Hann window has a positive floor and edge tiles
are padded, so the frame border is never a tile border. A 255-valued border strip
survives at 254.6/255. `pad_mode="reflect"` is available if the default
edge-replication is visible on strong vertical structure.

Use the whole-frame path for bulk work and tiling for final images. Tiled
inference makes `region-exposure` **tile-local**, so train with
`--no-region-exposure` if you need global regions. Full detail in
[`INFERENCE.md`](INFERENCE.md#tiled-inference--optional-for-real-detail).

## Augmentation on Colab

Set it once in cell 3:

```python
s = Session(..., augment_preset="default")
```

| preset | scale | rotation | flips | colour jitter | holes |
|---|---|---|---|---|---|
| `default` | 0.8–1.25 | ±12° | h + v | off | off |
| `conservative` | 0.9–1.15 | ±6° | h + v | off | off |
| `strong` | 0.7–1.4 | ±15° | h + v | ±0.1 | 2 |
| `none` | — | — | — | off | off |

One geometric warp per item, applied **identically to the source, the target and
the mask**, so an unpaired batch still describes a consistent crop position.

Colour jitter is off by default on purpose: the target domain *is* a colour look,
and perturbing it teaches the discriminator the style is fuzzy. `strong`
includes it so you can measure what it does. Full rationale in
[`TRAINING.md`](TRAINING.md#augmentation).

## Choosing a head

`--head` accepts simple heads, and composites joined with `+`:

```python
s = Session(head="freqsplit",      ...)   # default: low / mid / high bands
s = Session(head="lut3d",           ...)   # global trilinear LUT + per-pixel delta
s = Session(head="lut3d+freqsplit", ...)   # best default: global + band detail
s = Session(head="tanh+freqsplit",  ...)   # band detail, fully bounded
```

`lut3d+freqsplit` is the most useful composite because the two parts do
genuinely different things: a global colour grade and local per-band detail.
Parameter costs are in [`ARCHITECTURE.md`](ARCHITECTURE.md#output-heads).

## Persistence on Colab

Colab Drive survives a disconnect, so there is nothing to snapshot. Point
`runs_root` at Drive and leave `snapshot_every` at 0:

```python
s = Session(..., runs_root="/content/drive/MyDrive/nightflash/runs",
            device="cuda", snapshot_every=0)
s.auto_resume()      # reports "not a Kaggle runtime" and does nothing
```

The same notebook cell is safe to run in either environment. Colab does
disconnect idle sessions after ~90 minutes, so a shorter `extra_epochs` per run
gives you more save points. To move a finished run to Kaggle, zip the run
directory and upload it as a Dataset — `s.auto_resume()` there picks it up.

## Useful commands

| Command | Purpose |
|---|---|
| `colab new -s NAME --gpu L4` | Allocate a session (T4, L4, A100, H100) |
| `colab sessions` | List active sessions |
| `colab status -s NAME` | Show GPU, memory, and state |
| `colab exec -s NAME -f script.py` | Run a local script on the VM (no manual upload) |
| `colab exec -s NAME -f nb.ipynb` | Run a local notebook on the VM |
| `colab upload/download -s NAME SRC DST` | Move files both ways |
| `colab ls -s NAME /content` | List remote files |
| `colab ssh -s NAME --gpu L4` | SSH into the runtime for IDE/remote-dev |
| `colab url -s NAME --open` | Open the session in a browser |
| `colab log -s NAME -o log.md` | Export session history as markdown/notebook |
| `colab usage` | Check remaining compute units |
| `colab stop -s NAME` | Terminate the VM |

Run `colab <command> --help` for full options.

## Notes and gotchas

- **Keep-alive:** the CLI runs a background daemon that pings the VM every 60s so it does not idle out. That means an abandoned session keeps burning compute units. Always finish with `colab stop`.
- **Free tier:** roughly a 12-hour cap per session, plus daily GPU limits. A 30-epoch 256px run on T4 is comfortably inside that.
- **Free vs Pro:** some shapes and GPUs (A100, H100) need Colab Pro. Check with `colab usage`.
- **Transparent execution:** `colab exec -f local_script.py` reads the file locally and transmits it — you do not need to upload it first. The same applies to `-f notebook.ipynb`.
- **`repl` and `console` need a local TTY.** In scripts, pipe stdin instead: `echo "print(1)" | colab repl`.

---

# Colab MCP (agent workflow)

If you want a coding agent to operate your Colab session — open the notebook, edit cells, run them, read outputs — use Google's official Colab MCP server: [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp).

This bridges your **local** agent to a Colab session **in your browser**. It is the right tool when you want to say "run the pilot, show me the previews" and have the agent do it. It is overkill for a single scripted training run — use the CLI for that.

## Setup

Requires `uv`, and a client that runs locally and supports `notifications/tools/list_changed` (Gemini CLI, Claude Code, Windsurf, OpenCode).

```bash
pip install uv
```

Add to your MCP client config:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

If you are behind a non-standard package index, add `--index https://pypi.org/simple` to the args.

## How the connection works

1. The MCP server starts locally and opens a browser connection to `colab.research.google.com`.
2. You accept the connection in the browser.
3. An injected `open_colab_browser_connection` tool unlocks the notebook editing tools.
4. The agent can then read and write cells and execute them against your live session.

The server tracks connection state and notifies your client if the session drops, so the agent does not silently act on a dead runtime.

## Using it for this project

A sensible division of labour:

| Task | Tool |
|---|---|
| Explore previews, tweak weights, re-run | **Colab MCP** — interactive, iterative |
| Overnight 30-epoch run, fetch checkpoint | **Colab CLI** — unattended, scripted |
| Interactive training on your Mac with the TUI | **`nightflash-tui`** — local |
| Core ML export and benchmarking | **Your Mac** — `coremltools` is host-dependent |

Because Core ML export should happen on macOS, use MCP/Colab for training only, then download the `.pt` and export locally.

## Troubleshooting

- **"Connection attempt failed"** — the browser tab must be open and you must accept the connection prompt. Re-run the check tool.
- **Tools not appearing** — confirm the client is running locally and supports tool list-change notifications.
- **Drive mount cell hangs under `colab exec`** — interactive Google auth; use `colab auth` first, or skip the notebook path and use the manual cell workflow.

---

## Summary: which path should I use?

- **Watching a run and iterating on it** → Path A, the notebook
- **Running it overnight from the terminal** → Path B, `colab run`
- **Having your agent drive Colab for you** → Path C, Colab MCP
- **All of them, eventually** → export Core ML on your Mac, then integrate into the iOS app

Whatever path you use, the dataset lives on Drive, checkpoints are written to Drive continuously, and the training command is the same.
