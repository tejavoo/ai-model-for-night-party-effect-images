# Training guide

## Install

```bash
cd /Users/tejavu/Documents/opencode-experiments/disposable-cam-model-train
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

For the TUI:

```bash
python -m pip install -e '.[tui]'
```

## Validate the installation

```bash
python -m pip install -e '.[dev]'
pytest -q
```

A dependency-free pipeline check is also available:

```bash
python scripts/smoke_test.py
```

The smoke test creates synthetic images, runs two training steps, saves a checkpoint, and runs inference.

## Phase 1: convolutional baseline

Start without the transformer:

```bash
nightflash-train \
  --source-dir data/train/source \
  --target-dir data/train/target \
  --output-dir runs/conv-baseline \
  --image-size 256 \
  --batch-size 8 \
  --epochs 30 \
  --no-transformer \
  --preview-every 250 \
  --log-every 25
```

What to watch:

- Does the output acquire the target color palette?
- Are faces and geometry preserved?
- Does the model merely darken everything?
- Do outputs contain generic GAN artifacts?
- Is the effect coefficient head saturating at 0 or 1?

## Phase 2: hybrid transformer

After the baseline works:

```bash
nightflash-train \
  --source-dir data/train/source \
  --target-dir data/train/target \
  --output-dir runs/hybrid \
  --image-size 256 \
  --batch-size 4 \
  --epochs 50 \
  --transformer-blocks 1 \
  --transformer-token-size 16 \
  --preview-every 500
```

The transformer is used at the low-resolution bottleneck. It is not applied globally to every pixel.

## Phase 3: mask-aware training

```bash
nightflash-train \
  --source-dir data/train/source \
  --target-dir data/train/target \
  --mask-dir data/train/source_masks \
  --output-dir runs/mask-aware \
  --image-size 256 \
  --batch-size 4 \
  --epochs 50
```

The model then accepts:

```text
RGB[3] + person mask[1] = 4 channels
```

If the target domain is not mask-aware, the model still uses a zero mask for the target identity pass.

## Phase 4: deployment resolution

Start at 256 for fast iteration. Once the architecture is stable, train at the resolution you plan to export:

```bash
nightflash-train \
  --source-dir data/train/source \
  --target-dir data/train/target \
  --output-dir runs/hybrid-512 \
  --image-size 512 \
  --batch-size 2 \
  --epochs 50 \
  --transformer-blocks 1 \
  --preview-every 500
```

## Resume training

Every preview and epoch writes:

```text
runs/hybrid/checkpoints/latest.pt
runs/hybrid/checkpoints/epoch-0001.pt
runs/hybrid/checkpoints/epoch-0002.pt
```

Resume with:

```bash
nightflash-train \
  --source-dir data/train/source \
  --target-dir data/train/target \
  --output-dir runs/hybrid \
  --resume runs/hybrid/checkpoints/latest.pt \
  --image-size 256 \
  --batch-size 4 \
  --epochs 100
```

The checkpoint contains:

- Generator weights
- Discriminator weights
- Optimizer state
- Model configuration
- Training arguments
- Epoch and global step

## Mid-training inference

Training writes previews to:

```text
runs/hybrid/previews/
```

Each preview is a three-panel image:

```text
source | generated | target-domain reference
```

To test a mid-training checkpoint manually:

```bash
nightflash-infer \
  --checkpoint runs/hybrid/checkpoints/latest.pt \
  --input data/validation/source/example.jpg \
  --output outputs/mid-training.png \
  --seed 12345
```

Test several seeds:

```bash
for seed in 1 2 3 4 5; do
  nightflash-infer \
    --checkpoint runs/hybrid/checkpoints/latest.pt \
    --input data/validation/source/example.jpg \
    --output "outputs/seed-${seed}.png" \
    --seed "${seed}"
done
```

Inspect the deterministic component alone:

```bash
nightflash-infer \
  --checkpoint runs/hybrid/checkpoints/latest.pt \
  --input data/validation/source/example.jpg \
  --output outputs/deterministic.png \
  --no-effects
```

If the deterministic result is good but stochastic results are bad, reduce:

```bash
--effects-strength 0.25
```

## Training controls

### More target realism

Increase:

```bash
--adversarial-weight 1.5
```

Watch for hallucinations and identity drift.

### More content preservation

Increase:

```bash
--content-weight 2.0
--identity-weight 1.0
```

Watch for weak style transfer.

### More color-statistics matching

Increase:

```bash
--moment-weight 0.7
```

This is still an unpaired batch-level signal, not a paired pixel loss.

### Less extreme stochastic effects

Increase:

```bash
--effect-weight 0.05
```

or lower:

```bash
--effects-strength 0.25
```

### Disable the transformer for debugging

```bash
--no-transformer
```

This is the first thing to try if Core ML export fails.

## Recommended training schedule

1. 256px convolutional baseline: 20–30 epochs
2. 256px hybrid transformer: 30–50 epochs
3. Mask-aware model: 20–40 epochs
4. 512px final model: 30–50 epochs
5. Export only after held-out evaluation

Do not assume more epochs automatically improve quality. Unpaired GANs can overfit the target domain and damage content preservation.

## Progress output in notebooks

`tqdm` draws its bar with carriage returns on **stderr**. That works in a
terminal but not in a notebook, where it produces two problems:

1. The bar is often invisible, because the kernel frontend does not redraw
   stderr `\r` output.
2. When the trainer is driven through `subprocess.Popen(..., text=True)`, Python's
   *universal newlines* translation rewrites every `\r` into `\n`. Each redraw of
   the bar becomes its own printed line, flooding the cell with hundreds of
   near-identical lines.

`train.py` handles this at the source. `--progress auto` (the default) picks:

| Environment | Behaviour |
|---|---|
| Real TTY, not a notebook | live `tqdm` bar |
| Notebook, or piped output | one clean newline-terminated line per update |

Force it either way:

```bash
--progress plain     # always one line per update (notebook-safe)
--progress tqdm      # always a live bar
--progress none      # silent, for scripted runs
```

Plain output looks like this:

```text
epoch 3/50 started (125 steps, log every 25)
  [ 20.0%] epoch 3/50 step 25/125 eta=1m12s  loss_d=0.5310 loss_g=1.3765 adv=0.9213
  [ 40.0%] epoch 3/50 step 50/125 eta=48s    loss_d=0.5050 loss_g=1.2898 adv=0.8762
epoch 3/50 complete in 3m01s  loss_d=0.5182 loss_g=1.3310
```

`--log-every` controls the update cadence for both styles.

### Driving the trainer from a notebook

Use the helper rather than raw `Popen`. It reads the pipe in **binary** mode, so
carriage returns are never rewritten, and it separates real lines from in-place
redraws:

```python
from nightflash.notebook import train, show_previews, last_checkpoint

train([
    "--source-dir", f"{ROOT}/training_photos/normal_photos",
    "--target-dir", f"{ROOT}/training_photos/disposable_photos",
    "--output-dir", f"{ROOT}/runs/hybrid-warm",
    "--image-size", "256", "--batch-size", "16", "--epochs", "50",
    "--num-workers", "2", "--preview-every", "25", "--log-every", "25",
    "--device", "cuda", "--amp",
    "--transformer-blocks", "1",
], root="/content", log=f"{ROOT}/runs/hybrid-warm/train.log")

show_previews(f"{ROOT}/runs/hybrid-warm", n=6)
```

If you must use `Popen` yourself, the two rules are:

```python
# 1. binary mode, so universal newlines cannot split redraws
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

# 2. split redraws from lines yourself
from nightflash.streaming import iter_output
for kind, text in iter_output(proc.stdout):
    if kind == "line":
        print(text)                   # safe to print
    else:
        status_widget.update(text)    # overwrite in place
```

A `text=True` pipe plus a plain `for line in process.stdout` is the broken
combination. `tests/test_progress_streaming.py` asserts the difference.

---

## Resuming and warm starting

`--epochs` is an **absolute target**, not a count of extra epochs. Resuming with
`--epochs` at or below the checkpoint's epoch trains nothing; the trainer now
exits with an explicit error instead of reporting false success. To move from a
trained conv model to a hybrid one, use `scripts/warm_start.py`.

Both are covered in [`WARM_START.md`](WARM_START.md).

---

## When training looks unstable

Use these steps in order:

1. Lower the learning rate by 2×.
2. Increase `content_weight`.
3. Increase `identity_weight`.
4. Reduce `effects_strength`.
5. Disable the transformer.
6. Use more diverse source images.
7. Reduce the number of training steps and inspect checkpoints.
8. Check for duplicate or corrupted target images.

A target-domain model can produce plausible images while changing the source. Inspect the source/generated pair, not only the generated image.
