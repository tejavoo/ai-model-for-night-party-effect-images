# Warm starting, resuming, and changing resolution

Three questions come up constantly once Phase 1 is running, so this document
answers all three in one place.

| Goal | Use | Safe? |
|---|---|---|
| Continue the same architecture for more epochs | `nightflash-train --resume` | Yes |
| Add the transformer to a trained conv model | `scripts/warm_start.py` | Yes, with caveats |
| Change image size (256 → 512) on an existing checkpoint | `--resume` with a new `--image-size` | Yes |

---

## 1. Why `--resume` failed on your hybrid run

You hit this:

```
RuntimeError: Error(s) in loading state_dict for HybridGenerator:
    Missing key(s) in state_dict: "transformer.0.norm1.weight", ...
```

`--resume` performs a **strict** `load_state_dict`. Strict means every parameter
in the new model must exist in the checkpoint with an identical shape.

- Phase 1 ran with `--no-transformer`, so `self.transformer` was `nn.Identity()`,
  which has **zero parameters** and therefore contributes **no keys** to
  `state_dict`.
- Phase 2 dropped `--no-transformer`, so the model gained a real
  `EdgeTransformerBlock` registered as `transformer.0`, adding 12 parameter
  tensors.

Twelve missing keys, all under `transformer.` — exactly the error you saw.

### Reading this class of error

| Error | Meaning |
|---|---|
| **Missing key(s)** | The new model has a module the old one had no parameters for — something was *added* |
| **Size mismatch** | A parameter exists in both but with a different shape — channels, resolution, or mask input changed |
| **Unexpected key(s)** | A module was *removed* |

You got missing-keys only, which tells you every conv/encoder/decoder tensor
matched perfectly. Only the transformer was new.

### The root cause is a design gap, not a user error

`train.py` rebuilds the model from **CLI arguments** and then loads the
checkpoint. It never compares those arguments against the `model_config` that is
stored inside the checkpoint file. The information needed for a friendly error
was present the whole time — it just was not checked.

---

## 2. Warm start: conv-only → hybrid

`scripts/warm_start.py` performs the transfer deliberately and verifies it.

```bash
python scripts/warm_start.py \
    --checkpoint runs/conv-pilot/checkpoints/latest.pt \
    --output-dir  runs/hybrid-warm \
    --transformer-blocks 1
```

What it does:

1. Loads the conv checkpoint and confirms it really has no `transformer.*` keys.
2. Builds a new `HybridGenerator` with **every other field inherited** from the
   old `model_config`, so channel widths cannot silently drift.
3. Calls `load_state_dict(strict=False)` and then **verifies** that the only
   missing keys start with `transformer.`. A bare `strict=False` would happily
   swallow real corruption, so this check is the important part.
4. Rejects any `unexpected_keys` and any missing key outside the transformer.
5. **Resets both optimizer states.** The old AdamW moments were accumulated over
   a different parameter set and must not be carried over.
6. Writes `epoch = -1` and `global_step = 0`, so `train.py` computes
   `start_epoch = 0` and the preview/log counters restart.

Expected output:

```
old has transformer: False
tensors transferred: 46
tensors randomly initialised: 12
OK  every missing key belongs to the new transformer block
optimizer state   : reset (fresh AdamW moments)
wrote .../runs/hybrid-warm/checkpoints/latest.pt
wrote .../runs/hybrid-warm/warm_start_report.json
```

If the number of randomly initialised tensors is anything other than the size of
one transformer block, stop and read the error — it means something other than
the transformer changed.

Options:

- `--dry-run` verifies without writing anything.
- `--reset-discriminator` also starts the discriminator from scratch instead of
  carrying its weights over.

Then train normally against the **new** checkpoint:

```bash
python -m nightflash.train \
    --source-dir ... --target-dir ... \
    --output-dir runs/hybrid-warm \
    --transformer-blocks 1 \
    --resume runs/hybrid-warm/checkpoints/latest.pt
```

### What a warm start is and is not

**Is:** a legitimate technique. You inherit 40 epochs of trained conv weights
and only the transformer learns from scratch. It often converges in fewer total
epochs than training everything from zero.

**Is not:** a clean architecture comparison. Because the conv weights were
pretrained, a warm-started hybrid is **not** comparable to a from-scratch
hybrid. Keep `runs/conv-pilot/` as your untouched baseline. If you want to know
whether the transformer helps, train a second hybrid from scratch with no
`--resume` and compare those two against each other.

### Expect a rough start

A randomly initialised transformer inside an otherwise-trained network will
make the first preview look **worse** than the conv baseline. That is expected,
not a bug. Give it a few hundred steps before judging.

---

## 3. Resuming the same run

```bash
python -m nightflash.train \
    --source-dir ... --target-dir ... \
    --output-dir runs/hybrid-warm \
    --transformer-blocks 1 \
    --resume runs/hybrid-warm/checkpoints/latest.pt \
    --epochs 80
```

The architecture flags **must match the run you are resuming**:

- keep `--transformer-blocks 1`, or
- keep `--no-transformer` for a conv run.

Dropping the flag is what caused the original error.

### `--epochs` is an absolute target, not a count

This is the single most common resume mistake.

`train.py` does:

```python
start_epoch = checkpoint["epoch"] + 1
for epoch in range(start_epoch, args.epochs):
```

So if the checkpoint is at epoch 40 and you pass `--epochs 40`, the range
`range(41, 40)` is **empty** and **zero epochs train**.

Older versions of this codebase then printed a cheerful
`Training complete. Latest checkpoint: …` and exited 0, which looks like success.
`train.py` now detects this and exits non-zero with an explicit message:

```
Nothing to train: --epochs 40 is not greater than the resume point (epoch 41).
--epochs is an ABSOLUTE target, not a count of extra epochs.
Set --epochs to 41 + the number of extra epochs you want.
```

Rule of thumb: `new_epochs = checkpoint["epoch"] + 1 + extra_epochs_you_want`.

### The same rule in `Session` and `nf`

`Session` applies the identical rule, and it is the one to watch on Kaggle where
sessions are short:

| call | meaning | at epoch 40 it |
|---|---|---|
| `s.train()` | **absolute** — go to `Session.epochs` | trains nothing if `epochs <= 40` |
| `s.train(extra_epochs=20)` | **relative** — 20 more | trains epochs 41–60 |
| `nf train --epochs 80` | **absolute** | trains 41–80 |
| `nf train --extra-epochs 20` | **relative** | trains 41–60 |

`extra_epochs=0` is a deliberate no-op, not a rerun. Both `Session` and `nf`
refuse a no-op rather than pretending to succeed:

```
refusing to train: NO-OP GUARD - already at epoch 40; target 40 would train
zero epochs. Ask for more epochs.
```

Use `extra_epochs` for the resume path. It cannot overshoot, because it is
computed from the checkpoint's own epoch at call time.

### Across Kaggle sessions

On Kaggle `/kaggle/working` is wiped, so the resume path is
`snapshot_every=N` plus `s.auto_resume()`. See
[`KAGGLE.md`](KAGGLE.md#resuming-across-sessions) and
[`QuickStartKaggle.md`](QuickStartKaggle.md#session-2-resume).

### What correctly survives a resume

Verified behaviour:

| Item | Behaviour |
|---|---|
| Optimizer momentum | Carried over. AdamW continues rather than restarting. |
| Epoch counter | Continues at `last_epoch + 1`; no repeats. |
| Global step | Preserved, so `--preview-every` keeps its cadence. |
| Checkpoint history | `epoch-NNNN.pt` files accumulate in the same folder. |

---

## 4. Changing resolution on resume

**You can resume a 256px checkpoint at 512px.** No special handling is needed.

The model has no height- or width-indexed parameters:

- normalisation is `GroupNorm` and `LayerNorm`, both per-channel
- the effect head uses `AdaptiveAvgPool2d(1)`
- the decoder upsamples with dynamic `F.interpolate`
- there are no learned positional embeddings

So the same weights load cleanly at any resolution.

```bash
python -m nightflash.train \
    --output-dir runs/hybrid-512 \
    --image-size 512 --batch-size 4 \
    --transformer-blocks 1 \
    --resume runs/hybrid-warm/checkpoints/latest.pt
```

Practical notes:

- Drop the batch size. A T4 has 16 GB; `--batch-size 16` at 512px will OOM. Use
  4, or 2 if it still fails.
- If the DataLoader workers crash, set `--num-workers 0`.
- A checkpoint is resolution-agnostic, but a *preview* written at 256 and one
  written at 512 are not directly comparable. Compare like with like.

---

## 5. Notebook quick reference

See [`../colab/NightFlash_Colab.ipynb`](../colab/NightFlash_Colab.ipynb) for the
full workflow. Warm start and resume in one cell:

```python
import torch
from nightflash.notebook import train, show_previews, last_checkpoint

ROOT = "/content/drive/MyDrive/nightflash-edge"
CONV = f"{ROOT}/runs/conv-pilot/checkpoints/latest.pt"
WARM = f"{ROOT}/runs/hybrid-warm"

# conv-only -> hybrid (only needed once)
import subprocess, sys
subprocess.run([sys.executable, "/content/scripts/warm_start.py",
                "--checkpoint", CONV, "--output-dir", WARM,
                "--transformer-blocks", "1"], check=True)

# resume, extending by N epochs
ck = torch.load(last_checkpoint(WARM), map_location="cpu", weights_only=False)
target = ck["epoch"] + 1 + 50

train([
    "--source-dir", f"{ROOT}/training_photos/normal_photos",
    "--target-dir", f"{ROOT}/training_photos/disposable_photos",
    "--output-dir", WARM,
    "--image-size", "256", "--batch-size", "16",
    "--epochs", str(target),
    "--num-workers", "2", "--preview-every", "25",
    "--log-every", "25", "--seed", "0",
    "--device", "cuda", "--amp",
    "--transformer-blocks", "1",
    "--resume", last_checkpoint(WARM),
], root="/content", log=f"{WARM}/train.log")

show_previews(WARM, n=6)
```

`nightflash.notebook.train` forces `--progress plain` and reads the subprocess
pipe in binary, which is what makes output render correctly in Colab. See
[`TRAINING.md`](TRAINING.md#progress-output-in-notebooks) for the details.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Missing key(s) ... transformer.0.*` | Resuming a conv checkpoint into a hybrid model. Use `scripts/warm_start.py`. |
| `Nothing to train: --epochs N ...` | `--epochs` is not past the resume point. Use `last_epoch + 1 + extra`. |
| `size mismatch` for `stem.0.weight` | A mask-aware checkpoint is being resumed into a non-mask model, or vice versa. Keep `--mask-dir` consistent. |
| `size mismatch` for conv weights | Channel widths differ between runs. Use identical `--base-channels` / `--bottleneck-channels`, or warm start (which inherits them automatically). |
| `CUDA out of memory` at 512px | Lower `--batch-size`, or set `--num-workers 0`. |
| First hybrid preview looks worse than conv | Expected. The transformer is randomly initialised. Give it a few hundred steps. |
| Hybrid worse than conv after full training | The transformer is not helping, or the 45-image target set is overfitting. Check the dataset before blaming the architecture. |
