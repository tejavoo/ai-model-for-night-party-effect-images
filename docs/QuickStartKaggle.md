# QuickStart — Kaggle

Everything needed to go from a fresh Kaggle notebook to a Core ML model, with
the reasoning behind each choice. For the notebook itself see
`notebook/nightflash_kaggle.ipynb` (7 cells).

- [The 30-second version](#the-30-second-version)
- [Before you start: the target domain](#before-you-start-the-target-domain)
- [Session 1: set up](#session-1-set-up)
- [Session 1: train](#session-1-train)
- [Which mode?](#which-mode)
- [Validation strips](#validation-strips)
- [Session 2: resume](#session-2-resume)
- [Inference](#inference)
- [Using both GPUs](#using-both-gpus)
- [Export to Core ML](#export-to-core-ml)
- [Colab instead](#colab-instead)
- [Command cookbook](#command-cookbook)
- [Why these settings](#why-these-settings)
- [Troubleshooting](#troubleshooting)

---

## The 30-second version

Cells 1-2 are setup and are identical every session. Cells 3-6 are the work.

```python
# cell 1  (once per session)
!pip install -q -e .            # or: sys.path.insert(0, "/kaggle/input/<bundle>")
from nightflash import Session, brief

# cell 2  (once per session)
TRAIN_MODE = "paired"           # "paired" | "sameset" | "unpaired"
s = Session(
    train_mode=TRAIN_MODE,
    source_dir="/kaggle/input/datasets/levelupai/night-flash-images/training_photos/normal_photos" if TRAIN_MODE == "unpaired" else None,
    target_dir="/kaggle/input/datasets/levelupai/night-flash-images/training_photos/disposable_photos",
    runs_root="/kaggle/working/runs",
    arch="edgehat", head="freqsplit",
    image_size=256, batch_size=8, epochs=60,
    device="cuda",
    degrade_profile="moderate",  # the source is synthesised by degrading the pool
    val_count=8,                # held-out strip every epoch
    snapshot_every=1,            # publish a Dataset version every epoch
)
s.auto_resume()                 # pull in the last snapshot, or start fresh
print(brief(s, mode="train"))   # read this before training

# cell 3  train
s.train(extra_epochs=20)

# cell 4  look at the output
s.infer("/kaggle/input/datasets/levelupai/night-flash-images/training_photos/normal_photos", seeds=2, save_comparison=True)
# runs/<name>/validations/epoch-0020.png   input | output | target
# runs/<name>/validations/history.csv      psnr / ssim / l1 per epoch

# cell 5  export the BEST checkpoint, not latest.pt
!python -m nightflash.export_coreml \
    --checkpoint {s.checkpoint_dir / "best.pt"} \
    --output /kaggle/working/model.mlpackage

# cell 6  final snapshot
s.snapshot(note="done")
```

Two settings matter most on Kaggle. `snapshot_every=1`, because `/kaggle/working`
is wiped when the session ends — without it a disconnect at epoch 41 of 60 throws
away 41 epochs, with it you lose at most one. And export `best.pt` rather than
`latest.pt`, because a run that peaks early and decays has already overwritten
its good weights in `latest.pt`.

---

## Before you start: the target domain

**This matters more than any hyperparameter here.** The model learns whatever
your target folder contains. If that folder is not genuine night-flash
photography, training will not converge on the look you want no matter how long
you run it.

A quick sanity check you can run on the folder before uploading:

```python
import numpy as np
from pathlib import Path
from PIL import Image

lums, sizes = [], []
for p in sorted(Path("/kaggle/input/datasets/levelupai/night-flash-images/training_photos/disposable_photos").glob("*")):
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".heic", ".webp"):
        continue
    im = Image.open(p).convert("L")
    lums.append(np.asarray(im, float).mean() / 255.0)
    sizes.append(max(im.size))

lums, sizes = np.array(lums), np.array(sizes)
print(f"images            {len(lums)}")
print(f"mean luminance    {lums.mean():.3f}  (night flash: < 0.20)")
print(f"under 0.20        {(lums < 0.20).mean():.0%}   (want: most of them)")
print(f"over 0.35         {(lums > 0.35).mean():.0%}   (want: very few)")
print(f"median long edge  {np.median(sizes):.0f}px  (want: >= 512)")
```

If `over 0.35` is high, the folder holds daylight photos and the model will
learn the opposite of the intended look. Either fix the folder or generate a
correct target set — see `docs/DATASET_SOURCING.md`.

Also worth knowing: `disposable_photos` in this repo currently contains 206
daylit travel and outdoor stock images. It is not a valid target domain yet.

---

## Session 1: set up

### 1. Upload the data

Create a Kaggle Dataset from your photo folders. Keep the two domains in
separate subfolders:

```
night-flash-images/training_photos/
  normal_photos/        # the source domain: ordinary daylight photos
  disposable_photos/    # the target domain: real night-flash photos
```

Attach it in the notebook's **Add Data** panel, right-hand side, as *Input*.

### 2. Get the code in

Upload `nightflash-edge-kaggle.zip` as a Kaggle Dataset. Kaggle auto-extracts
it into `/kaggle/input/`. Attach it in the **Add Data** panel, then install:

```python
!pip install -q -e /kaggle/input/datasets/levelupai/nightflash-edge-kaggle
```

### 3. Pick the GPU

Right-hand sidebar → **Notebook options** → **Accelerator**.

| Accelerator | Verdict |
|---|---|
| **GPU T4 x2** | Best. 16 GB each. Run two arms, see [Using both GPUs](#using-both-gpus). |
| GPU T4 x2 (RAPIDS) | Same, plus a large RAM disk. Useful for big datasets. |
| GPU P100 | Fine, ~1.5x slower than T4 for fp16. |
| **TPU** | **Not supported.** See below. |
| CPU | Unusable for training. Fine for export. |

**Why no TPU:** the model uses `grid_sample`-free explicit gathers, `torch.jit`
tracing for Core ML, and a PatchGAN discriminator. TPU support would need a
full XLA rewrite of the augmentation pipeline, and Core ML export would still
have to happen on CPU. Two T4s are strictly more useful.

### 4. Authenticate for snapshots

Snapshots use the Kaggle API, which needs credentials:

```python
import json, os
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump({"username": "<user>", "key": "<key>"}, f)
```

Get the key from **Kaggle → Settings → API → Create New Token**. Without this,
`snapshot_every` prints a warning each epoch and training continues unaffected.

---

## Session 1: train

```python
from nightflash import Session, brief

# Pick a mode. paired and sameset build the source by degrading TARGET_DIR,
# so SOURCE_DIR is not used in those modes.
TRAIN_MODE = "paired"     # "paired" | "sameset" | "unpaired"

s = Session(
    train_mode=TRAIN_MODE,
    source_dir="/kaggle/input/datasets/levelupai/night-flash-images/training_photos/normal_photos" if TRAIN_MODE == "unpaired" else None,
    target_dir="/kaggle/input/datasets/levelupai/night-flash-images/training_photos/disposable_photos",
    runs_root="/kaggle/working/runs",
    arch="edgehat",
    head="freqsplit",
    image_size=256,
    batch_size=8,
    epochs=60,               # ABSOLUTE target epoch
    device="cuda",
    augment_preset="default",
    degrade_profile="moderate",
    val_count=8,             # images per validation strip
    val_select="fixed",      # same photos every epoch
    keep_last=3,
    snapshot_every=1,
)
s.auto_resume()
print(brief(s, mode="train"))
s.train(extra_epochs=20)
```

Read the briefing before training. It prints the plan, the flags, the paths, the
flow, the current state, and a numbered decision menu. It never offers a
no-op "continue" — if there is nothing to do it says so and gives you the
command that would actually do something.

`brief(s)` returns the text; `print()` it. `nf brief` prints it too.

### Which mode?

| mode | what it does | when |
|---|---|---|
| `paired` | source = `degrade(image_i)`, target = `image_i` | you want real PSNR/SSIM and `best.pt` |
| `sameset` | source = `degrade(image_j)`, `j ≠ i`, target = `image_i` | you want the model unable to invert your degradation |
| `unpaired` | two separate pools | you already have a matching source domain |

**The trade-off in one sentence:** `paired` can learn to invert your specific
degradation instead of learning the look, and `sameset` cannot — but `sameset`
gives you no real quality number and no `best.pt`.

**With two GPUs, run both.** Copy the notebook, set `TRAIN_MODE = "sameset"` and
`name="b"` in the second, and compare the validation strips. That is the
comparison worth making and it costs one session.

**How to spot the failure:** if the model looks great on freshly-degraded images
but poor on ordinary photographs, it learned the inverse rather than the
aesthetic. The validation strip makes it obvious within a few epochs.

### Validation strips

After every epoch:

```
runs/<name>/validations/
  epoch-0007.png   input | output | target, one image per row
  latest.png
  history.csv      epoch,images,psnr,ssim,l1,effect
```

`checkpoints/best.pt` holds the highest-PSNR weights seen, and `--keep-last`
prunes old checkpoints. **Export `best.pt`, not `latest.pt`**, unless PSNR is
still rising at the final epoch — that is the whole reason `best.pt` exists.

Seeds are derived from each filename with SHA-256, so a strip rendered this
session matches one rendered last week and epoch 7 is comparable to epoch 8.

### `--epochs` vs `extra_epochs`

These mean different things and mixing them up is how a run trains forever:

| | meaning | use |
|---|---|---|
| `epochs=60` | **absolute**. Stop at epoch 60, whatever you have done so far. | the CLI, and setting up a session |
| `s.train(extra_epochs=20)` | **relative**. Train 20 more from wherever the run is. | resuming, and Kaggle's fixed time budget |

Calling `s.train(extra_epochs=20)` three times on a fresh run gives you 60
epochs. Calling it on a run already at epoch 60 gives you 80. Neither can
overshoot silently.

### The equivalent CLI

```bash
nf train --source-dir ... --target-dir ... --epochs 60 --snapshot-every 1
nf train --source-dir ... --target-dir ... --extra-epochs 20
```

`nf train` without `--yes` prints the briefing and waits. Add `--yes` to skip it
in a scripted run.

---

## Session 2: resume

Kaggle sessions have a hard wall-clock limit (usually 12h). When yours ends,
start a new one on the same notebook.

1. **Attach the snapshot dataset.** In **Add Data**, attach the dataset the
   snapshot created (`nightflash-edgehat-freqsplit-night`). Your original data
   dataset must be attached too.
2. **Run cells 1-2 again**, with one change:

```python
s = Session(..., snapshot_every=1)
s.auto_resume()      # finds the attached snapshot and restores it
```

`s.auto_resume()` replaces the old two-step dance. It:

- looks for an attached dataset whose name matches this run, ignoring case and
  `-`/`_` differences, so the auto-generated slug is recognised;
- accepts a sole candidate under any name, so a hand-uploaded snapshot works;
- refuses to guess when two snapshots are attached and neither matches;
- archives any existing run directory first, so nothing is overwritten;
- prints `starting fresh` and does nothing when there is no snapshot, which is
  exactly right for a first session.

It is a no-op outside Kaggle, so the same cell is safe in Colab.

3. **Keep training.**

```python
s.train(extra_epochs=40)
```

### Snapshot vs restore, explicitly

If you prefer the explicit calls:

```python
from nightflash.kaggle import snapshot, restore, environment_report

print(environment_report())     # gpus, torch version, attached datasets
snapshot(s.run_dir)             # publish
restore(s.run_dir)              # pull back (requires the dataset attached)
```

`restore` picks the first attached dataset containing `checkpoints/latest.pt`.
`auto_restore` is the smarter version. CLI equivalents: `nf snapshot`,
`nf resume`.

### If a snapshot fails

Snapshot failures never stop training. You will see:

```
epoch 12: snapshot failed (HTTPError 429); training continues
```

Common causes: API rate limit (lower `snapshot_every` to 5), missing
`kaggle.json`, or the dataset name already taken by a different account. Your
checkpoint on disk is unaffected — download `/kaggle/working` output before the
session ends as a backstop.

---

## Inference

Whole-frame is the default and the right first choice. The image is letterboxed
to `image_size`, styled, and restored to its exact original dimensions. Nothing
is cropped or stretched.

```python
s.infer("photo.jpg")                                  # 1:1, original size
s.infer("photos/", seeds=3, save_comparison=True)     # 3 variants + side-by-side
```

### Tiled inference

For native-resolution detail — grain, halation, local texture — process
overlapping tiles and crossfade:

```python
s.infer("photo.jpg", tile=256)                        # match image_size
s.infer("photo.jpg", tile=256, tile_overlap=64)       # 50% overlap, smoother
```

**Match `tile` to `image_size`.** This is not a suggestion; a mismatch is a
distribution shift:

| training | inference tile | result |
|---|---|---|
| 256 crops | 256 tiles | **Correct.** Same feature statistics. |
| 256 crops | 512 tiles | Degraded. Larger context than the model saw. |
| 1024+ images | 256 tiles | Worst. Objects appear cropped in ways never seen in training. |

The code warns when they differ, but matching them is the reliable choice.

**Borders do not fade.** Two mechanisms handle it, so you do not need to
pre-pad:

1. the Hann window has a positive floor (1e-3), so weights never reach zero;
2. edge tiles extend past the frame and are edge-replicated, so the frame
   border is never a tile border.

Verified: a 255-valued border strip comes through tiled inference at 254.6/255,
worst-case error 1.0. A fade-to-black would show near 0.

`pad_mode` controls what the model sees outside the frame:

| mode | behaviour | when |
|---|---|---|
| `edge` (default) | replicate the border row/column | safe, never invents a gradient |
| `reflect` | mirror the border | images with strong vertical structure |
| `black` | zero fill | only if trained on padded black borders |

```python
s.infer("photo.jpg", tile=256, pad_mode="reflect")
```

### Downscaling while keeping the aspect ratio

`max_edge` shrinks the long edge, preserving aspect, before anything else. Use
it to batch many images, or to match a model trained at a small `image_size`:

```python
s.infer("12mp.jpg", max_edge=1024)                  # 4032x3024 -> 1024x768
s.infer("12mp.jpg", max_edge=1024, tile=256)       # downscale AND tile
s.infer("huge/", max_edge=2048, seeds=2)            # many images, 2 variants
```

It never upscales: `max_edge=4096` on a 300x200 photo returns 300x200.

### Cost of tiling

Whole-frame is dramatically cheaper because the model only ever sees
`image_size` pixels, no matter how large the source is. Tiling cost scales with
the source, and overlap has a large effect:

| source | whole-frame scale | tiles @ overlap 64 | MP | tiles @ overlap 128 | MP |
|---|---|---|---|---|---|
| 600x800 | 0.43x | 12 | 0.8 | 24 | 1.6 |
| 1920x1080 | 0.24x | 60 | 3.9 | 112 | 7.3 |
| 4032x3024 | 0.08x | 336 | 22.0 | 713 | 46.7 |

Read that last row carefully. Tiling a 12MP photo at 50% overlap means running
the network over 46.7 megapixels — more than eleven full passes over the whole
image. This is fine for one hero shot and terrible for a batch.

`tile_overlap` defaults to 128 (50% of a 256 tile). Drop to 64 for a 4x speedup
with slightly more visible crossfade seams. Use it for hero shots, not batches.

### One caveat about tiling

`region-exposure` becomes tile-local when you tile: a 4x4 exposure grid over a
256px tile is not the same as over the whole frame. If you enabled
`--region-exposure`, the model expects global regions — either infer
whole-frame, or train without it. The code prints this note when relevant.

---

## Using both GPUs

A 4M-parameter model will not saturate a T4. The second GPU is better spent on a
second experiment, which is also the only way to get a real comparison.

### The comparison you actually want

At 4M parameters a single T4 is not saturated, so the second GPU is best spent
answering a question rather than going faster. There are two worth asking.

**Question 1 — does the model learn the look, or the inverse of my degradation?**
This one is specific to the paired/sameset modes and is the more important of
the two. Duplicate the notebook and change cell 3 only:

```python
# notebook A
TRAIN_MODE = "paired"
s = Session(..., train_mode=TRAIN_MODE, name="a", device="cuda")
s.train(extra_epochs=20)

# notebook B
TRAIN_MODE = "sameset"
s = Session(..., train_mode=TRAIN_MODE, name="b", device="cuda")
s.train(extra_epochs=20)
```

Then put `runs/*/validations/epoch-0020.png` side by side. If the `paired` arm
looks good on degraded inputs while the `sameset` arm looks better on ordinary
photographs, that is the shortcut showing itself.

**Question 2 — which architecture?**

```python
# notebook A
s = Session(..., arch="edgehat", head="freqsplit", name="a", device="cuda")

# notebook B
s = Session(..., arch="swinir",  head="freqsplit", name="b", device="cuda")
```

Because `name` differs, the two runs live in separate directories and neither
can clobber the other. Compare them on the same held-out images with the
inference settings held constant, so the only variable is the architecture.

To pin a specific GPU inside one notebook:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
s = Session(..., device="cuda:0")   # this is now physical GPU 1
```

`recommended_device("auto")` returns plain `"cuda"` on Kaggle, which lets
PyTorch use whichever device it likes. For a genuine two-arm comparison, run
two notebooks rather than fighting over one device.

### What does *not* help

- **DDP / `DataParallel`.** At 4M parameters the step is dominated by kernel
  launch overhead, not compute. You would add sync cost and get nothing.
- **Larger batch on one GPU.** `--batch-size 16` on a T4 is fine and slightly
  faster per epoch. Beyond ~24 you will hit OOM at 256px.

---

## Export to Core ML

Export is a script, not a `Session` method:

```bash
python -m nightflash.export_coreml \
    --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
    --output model.mlpackage \
    --compute-units CPU_AND_GPU \
    --half
```

Then confirm it against PyTorch before you trust it:

```bash
python -m nightflash.validate_coreml \
    --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
    --coreml model.mlpackage \
    --input /kaggle/input/datasets/levelupai/night-flash-images/training_photos/normal_photos \
    --image-size 256
```

Useful flags:

| flag | why |
|---|---|
| `--half` | fp16 weights, roughly halves the file. Verify with `validate_coreml` after. |
| `--compute-units CPU_AND_GPU` | Neural Engine + GPU. Best for live preview. |
| `--compute-units ALL` | Adds CPU. Most compatible, largest file. |
| `--input-type tensor` | Feeds a 1x3xHxW tensor instead of an image. Lets Core ML own preprocessing. |
| `--mask-input` | Adds the person-mask input, needed if you trained with masks. |
| `--deterministic-only` | Drops the stochastic grain/halation sampling. Smaller and faster, but a fixed look. |

All 14 architecture x head combinations have been verified to convert, with
tensor-I/O MAE 0.00030 against PyTorch.

Things that were specifically done to make conversion work, so you do not
"fix" them:

- `nn.MultiheadAttention` is not used. It lowers to a fused op that fails
  conversion; the attention uses explicit q/k/v matmuls instead.
- `Lut3DHead.table` is a flat `(s**3, 3)` tensor. Per-channel gathers break Core
  ML scatter lowering.
- `grid_sample` is not used anywhere. It was returning near-centre values
  regardless of the grid in testing, and the 3D LUT uses explicit trililinear
  gathers.
- `HATBlock.self.grid` and the `EdgeAttention` window clamp use precomputed
  Python `int`s from `__init__`. Under `torch.jit.trace`, `x.shape[1]` becomes
  a 0-d tensor, which breaks `max()`, `min()` and `adaptive_avg_pool2d`.

Download the `.mlpackage` from `/kaggle/working` output when the session ends.

---

## Colab instead

Two differences, everything else is identical.

**Persistence.** Drive survives; `/kaggle/working` does not.

```python
from google.colab import drive
drive.mount('/content/drive')
runs_root = "/content/drive/MyDrive/nightflash/runs"
```

**No snapshots.** Point `runs_root` at Drive and the problem disappears. Leave
`snapshot_every=0` (the default) and `auto_resume()` will tell you it is not a
Kaggle runtime and do nothing.

```python
s = Session(..., runs_root="/content/drive/MyDrive/nightflash/runs",
            device="cuda", snapshot_every=0)
```

Colab's T4 is one GPU, so drop the two-GPU section. Colab also disconnects idle
sessions after ~90 minutes; a shorter `extra_epochs` per cell gives you more
save points.

See `docs/COLAB.md` and `colab/NightFlash_Colab.ipynb`.

---

## Command cookbook

Every line below is a complete, runnable command. The `--why` column is the
part worth remembering.

### Training

```bash
# Start a run, read the briefing first (recommended for any new experiment)
nf train --target-dir POOL --name a --epochs 60

# Non-interactive (scripts, re-runs)
nf train --target-dir POOL --name a --epochs 60 --yes

# Add epochs to an existing run
nf train --target-dir POOL --name a --extra-epochs 20

# The shortcut-free variant, for your second GPU
nf train --target-dir POOL --name b --epochs 60 --train-mode sameset

# Train across two separate pools instead (the pre-degradation behaviour)
nf train --source-dir SRC --target-dir TGT --name c --epochs 60

# Kaggle: publish a snapshot every epoch so a disconnect costs one epoch
nf train --target-dir POOL --name a --epochs 60 --snapshot-every 1

# Snapshot less often to dodge API rate limits
nf train ... --snapshot-every 5
```

| Why | Command |
|---|---|
| Wider or gentler corruption | `--degrade-profile strong` / `subtle` |
| Override one degradation bound | `--degrade-contrast-min 0.3 --degrade-contrast-max 0.4` |
| A reusable or "virtual camera" profile | `--degrade-config my-camera.json` |
| Turn the ramp off | `--no-degrade-curriculum` |
| Feed clean targets as sources more often | `--degrade-none-prob 0.2` |
| Suppress the supervised term | `--paired-weight 0` |
| Fewer / smaller validation strips | `--val-count 4`, `--val-cols 2` |
| Sample a fresh validation set each epoch | `--val-select random` |
| Supply your own held-out folder | `--val-holdout-dir /kaggle/input/.../val_night` |
| Keep more or fewer checkpoints | `--keep-last 5` |
| Don't keep a best checkpoint | `--no-keep-best` |
| Reduce frame-to-frame flicker | `--ema-decay 0.999` |

| Why | Command |
|---|---|
| Second architecture on the second GPU | `--arch swinir --name b` |
| Composite head, usually the best quality | `--head lut3d+freqsplit` |
| Turn a capability off to isolate its effect | `--no-halation` / `--no-subject-gate` / `--no-region-exposure` |
| Conservative augmentation, small dataset | `--augment conservative` |
| No augmentation, to measure its effect | `--augment none` |
| Bigger crops, more detail, ~4x the memory | `--image-size 512 --batch-size 4` |
| Fresh start, keep the old run | `nf reset --name a` then retrain |
| Genuinely delete the run | `nf reset --name a --delete` |

`nf reset` archives to `<name>_archive_N` by default. It never deletes without
`--delete`.

### Inspection

```bash
nf brief  --name a                    # full pre-flight plan and decision menu
nf state  --name a                    # one-line JSON state
nf config --name a                    # resolved config, no side effects
```

These take no data paths — they read the run directory only.

### Inference

```bash
# Whole frame, exact original size (default, fastest)
nf infer --name a --input photo.jpg

# Side-by-side original | styled
nf infer --name a --input photo.jpg --comparison

# Three stochastic variants (grain, halation, leak are seeded per variant)
nf infer --name a --input photo.jpg --seeds 3

# Native-resolution detail; tile MUST match --image-size
nf infer --name a --input photo.jpg --tile 256

# Smoother crossfade, roughly 2x the tiles
nf infer --name a --input photo.jpg --tile 256 --tile-overlap 128

# Downscale the long edge first, aspect preserved
nf infer --name a --input photo.jpg --max-edge 1024

# Downscale AND tile
nf infer --name a --input photo.jpg --max-edge 1024 --tile 256

# A whole folder
nf infer --name a --input photos/ --seeds 2
```

| Why | Command |
|---|---|
| Vertical structures, mirrored padding reads better | `--pad-mode reflect` |
| You want the output smaller than the input | `--max-edge 1024` |
| Batch throughput on a big folder | `--input photos/ --max-edge 1024` |
| Compare two arms on identical inputs | same `--input`, different `--name` |

### Kaggle lifecycle

```bash
nf snapshot --name a --note "epoch 40"   # publish
nf resume   --name a                     # restore if attached
nf resume   --name a --required          # fail instead of starting fresh
```

`--required` is what you want in the last session before exporting: it turns a
forgotten attachment into a loud error instead of a silently fresh run that
overwrites your weights with epoch 1.

### Evaluation and export

You need a **held-out target set** — night-flash photos the model never trained
on — or you have no way to tell whether the model learned the look. Keep it out
of `target_dir`. `/kaggle/input/datasets/levelupai/night-flash-images/training_photos/val_night` is a good place.

```bash
# Score a checkpoint against the held-out set
python -m nightflash.evaluate \
    --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
    --source-dir /kaggle/input/datasets/levelupai/night-flash-images/training_photos/normal_photos \
    --target-dir /kaggle/input/datasets/levelupai/night-flash-images/training_photos/val_night \
    --output-json runs/edgehat_freqsplit_night/evaluations/score.json
```

It reports `content_l1_mean`, `edge_l1_mean`, `simple_ssim_mean`,
`seed_diversity_l1` and mean RGB for source vs target. It is careful to say in
its own output that these are **unpaired diagnostics, not proof of
target-domain quality**. For a release decision you also want FID/KID, a
perceptual metric, and human A/B. See `docs/EVALUATION.md`.

A note on `--val-dir`: `Session` accepts it and records it in `config.json` and
prints it in the briefing, but the training loop does not yet consume it. Run
`nightflash.evaluate` explicitly.

```bash
# Export
python -m nightflash.export_coreml \
    --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
    --output model.mlpackage

# Verify the exported model against PyTorch
python -m nightflash.validate_coreml \
    --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
    --coreml model.mlpackage --input photo.jpg

# Deliberately transfer conv weights from an older conv-only checkpoint
python scripts/warm_start.py \
    --checkpoint old_conv_only.pt \
    --output-dir runs/edgehat_freqsplit_night
```

`warm_start.py` is the supported way to move from a conv-only checkpoint to a
transformer one. Plain `load_state_dict` is strict and will refuse; this script
copies only the layers whose shapes still match and reports what it skipped.

---

## Why these settings

**`image_size=256` with `batch_size=8`.** 256 is the sweet spot on a T4: it
divides evenly by the 8x8 window attention, and it keeps grain and halation
large enough in frame to be learnable. 512 quadruples memory and roughly
quadruples step time, and on a small dataset it overfits sooner. Only move up
once the previews at 256 look good rather than merely plausible.

**`head="freqsplit"`.** Your look lives in two bands: colour and exposure in the
low band, grain and halation in the high band. Splitting the residual by
frequency lets each head specialise, and it is only 63k parameters. `lut3d` is
the strongest single head at 129k, and `lut3d+freqsplit` is the best
combination at 192k — worth trying once `freqsplit` alone looks right.

**`--epochs 60` with `batch_size 8`.** On a few hundred images that is a few
thousand steps. Watch `previews/` rather than the loss: the discriminator loss
is not a quality signal. If previews stop changing after ~40 epochs, more epochs
will not help; the dataset is the limit.

**`augment_preset="default"`.** Scale 0.8-1.25, rotation +/-12 degrees, both
flips, no colour jitter. Scale jitter matters most: a 256px crop of a 12MP night
photo is a close-up, the same crop of a 3MP wide shot is a whole room, and
without jitter the model silently learns only the first case.

Colour jitter is **off by default on purpose.** Your target domain *is* a colour
look. Perturbing target colour teaches the discriminator that the style is
fuzzy, and it can wash out exactly the grade you are trying to learn. It is
available behind `apply_colour_to_target` for when you have already memorised
the target set.

**`snapshot_every=1`.** A Kaggle session ends without warning. One epoch of loss
is a good price for not losing the run.

**Whole-frame inference by default.** It is 20-50x cheaper, it keeps
`region-exposure` global, and for a phone camera preview it is more than enough.
Tiling is for hero shots.

**Two separate runs, not DDP.** At 4M parameters, DataParallel adds
synchronisation cost without adding throughput, and two arms answer a real
question that one arm cannot.

---

## Troubleshooting

**`/kaggle/working` is empty after reconnecting.** Expected — it is wiped.
That is what `snapshot_every` is for. If you have no snapshot, the run is gone;
start again with a lower `extra_epochs` so you save more often.

**`no attached snapshot matches <name>` with datasets attached.** The snapshot
dataset is not in **Add Data**. Attaching the data alone is not enough. If the
snapshot was uploaded under a different name and two datasets are attached,
`auto_restore` refuses to guess — attach only the snapshot and re-run.

**`refusing to train: NO-OP GUARD`.** You asked for an absolute target you have
already reached. Use `--extra-epochs N` to add epochs.

**`Missing key(s) in state_dict` on load.** The checkpoint was saved by a
different architecture. `nn.Identity()` has no parameters, so an old conv-only
checkpoint has zero `transformer.*` keys — that is "added module", not a
corrupt file. Start a new run directory with a different `--name`, or use
`scripts/warm_start.py` to transfer the conv layers deliberately.

**`size mismatch for ...` on load.** A layer's shape changed between the
checkpoint and now — usually a different `--head` or `--region-grid`. Same fix.

**Training is slow, or loss_d sits at 0.** The source and target domains are too
similar for the discriminator to find a boundary. That is a dataset problem, not
a hyperparameter problem.

**Previews look fine but tiled inference looks worse.** Almost always a tile
size mismatch. Set `tile` equal to `image_size`.

**`RuntimeError: Adaptive pool MPS: input sizes must be divisible...`.** An
MPS-only limitation, and it means `image_size` or `tile` is smaller than the
architecture's window. Use at least 64, or pass `--device cpu`.

**Snapshot spam every epoch.** Lower `--snapshot-every`. Rate limits are
per-hour and an epoch on a small dataset can be under a minute.
