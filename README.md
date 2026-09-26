# Night Flash Edge

A small, open-source training and deployment toolkit for learning a **disposable-camera / Night Flash look from unpaired image domains**.

The project is designed around the existing native app:

```text
/Users/tejavu/Documents/opencode-experiments/disposable-cam
```

The app already has a deterministic Core ML color transform plus Core Image/Vision finishing. This repository adds a separate, compact learned model that can learn a more adaptive style without requiring paired source/target images.

## What this repository contains

- `edgehat` generator: EdgeViT-style local windowed attention plus one HAT
  overlapping cross-attention block
- Bounded RGB residual prediction
- Four head families and three composites (see [Heads](#heads))
- Three switchable capabilities: halation, subject gate, region exposure
- Small stochastic-effect coefficient head
- Seeded procedural grain, leak, exposure, saturation, and vignette renderer
- Unpaired PatchGAN training with content, identity, edge, and batch-statistics
  losses
- Geometric augmentation with a shared warp across source, target and mask
- Mid-training preview generation
- Checkpoint/resume, plus Kaggle snapshot/restore for cross-session survival
- Whole-frame, tiled, and downscale-then-tiled inference
- Evaluation diagnostics
- Core ML export to `.mlpackage` and `.mlmodel`, with a PyTorch-vs-Core-ML
  validator
- Optional Textual terminal UI
- Colab, Kaggle, and local documentation

## Design decision

The model predicts a **bounded residual** on top of the input, then a seeded
stochastic renderer adds grain and light leaks:

```text
source image
→ convolutional encoder
→ local windowed attention + one global cross-attention block
→ head (tanh | per-pixel | freq-split | 3D LUT | composite)
→ learned style residual
→ seeded stochastic effects
→ existing Core Image / Vision finishing
```

Two decisions are worth defending:

- **Bounded residual.** The output is `clamp(source + residual)`, so the model
  learns a *modification* rather than a whole image. That is what protects
  faces, text, and geometry, and it is why identity stays stable even when the
  style is aggressive.
- **Not a pure per-pixel network.** Grain and halation live in the high
  frequency band, which a per-pixel head represents poorly. The default
  `freqsplit` head predicts low / mid / high bands separately and recombines
  them, so each band gets the resolution it needs.

The model is not intended to replace the entire camera pipeline. Keep the
existing Vision person mask and Core Image effects in the app.

## Quick start

```bash
cd /Users/tejavu/Documents/opencode-experiments/disposable-cam-model-train
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[tui]'
```

Create tiny synthetic data and run a smoke test:

```bash
python scripts/smoke_test.py
```

Everything else goes through one object:

```python
from nightflash import Session
from nightflash.briefing import brief

s = Session(
    source_dir="/path/to/normal-photos",
    target_dir="/path/to/disposable-photos",
    arch="edgehat", head="freqsplit", name="night",
    image_size=256, batch_size=8, epochs=50,
)

print(brief(s, mode="plan"))   # read this before spending GPU hours
s.train()                       # go to epochs=50, resuming if a checkpoint exists
s.infer("photo.jpg", seeds=3)   # inference only, never touches the trainer
```

The same from a terminal:

```bash
nf brief  --source-dir SRC --target-dir TGT          # the pre-flight briefing
nf train  --source-dir SRC --target-dir TGT --epochs 50
nf infer  --input photo.jpg --seeds 3
```

Each run gets its own folder, so switching architecture or head never
overwrites another run:

```text
runs/edgehat_freqsplit_night/
  config.json  checkpoints/  previews/  inferences/  evaluations/  logs/
```

Legacy standalone entry points still exist for older checkpoints and are
documented in [`docs/TRAINING.md`](docs/TRAINING.md): `nightflash-train`,
`nightflash-infer`, `nightflash-evaluate`, `nightflash-export-coreml`,
`nightflash-validate-coreml`, `nightflash-tui`. Prefer `nf` and `Session` for
anything new.

## Documentation

1. [`docs/QuickStartKaggle.md`](docs/QuickStartKaggle.md) — **start here for Kaggle**: full flow, both GPUs, command cookbook
2. [`docs/PLAN.md`](docs/PLAN.md) — recommended implementation phases
3. [`docs/DATA.md`](docs/DATA.md) — unpaired dataset preparation
4. [`docs/DATASET_SOURCING.md`](docs/DATASET_SOURCING.md) — how `training_photos/` was built, and the target-domain gap
5. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the `edgehat` model, heads, and capabilities
6. [`docs/CLI.md`](docs/CLI.md) — `Session`, the pre-flight briefing, and the `nf` command
7. [`docs/TRAINING.md`](docs/TRAINING.md) — augmentation, loss weights, mid-training inference
8. [`docs/COLAB.md`](docs/COLAB.md) — Google Colab notebook, CLI, and MCP
9. [`docs/KAGGLE.md`](docs/KAGGLE.md) — Kaggle setup, T4×2, cross-session resume
10. [`docs/WARM_START.md`](docs/WARM_START.md) — resuming, warm starting, changing resolution
11. [`docs/INFERENCE.md`](docs/INFERENCE.md) — whole-frame, tiled, downscaled inference
12. [`docs/EVALUATION.md`](docs/EVALUATION.md) — quality and performance evaluation
13. [`docs/COREML.md`](docs/COREML.md) — Core ML export and iOS integration
14. [`docs/TUI.md`](docs/TUI.md) — the optional terminal UI

## Quick start: one object

Everything is driven by a single `Session`. It owns a run directory, so
experiments never collide and resuming is just calling it again.

```python
from nightflash import Session
from nightflash.briefing import brief

s = Session(
    arch="edgehat", head="freqsplit", name="night",
    source_dir="data/source", target_dir="data/target",
    image_size=256, batch_size=8, epochs=50,
)

print(brief(s, mode="plan"))   # read this before training
s.train()                       # absolute: go to epochs=50
s.train(extra_epochs=50)        # relative: add 50 from wherever it is
s.infer("photo.jpg", seeds=3)   # inference only, no training
s.state()                       # one-line status, writes nothing
s.reset()                       # archive, never delete
```

`epochs=50` is an **absolute** target: asking for it again at epoch 50 is a
no-op, not another 50 epochs. `extra_epochs=N` is **relative**, which is the
resume path. Mixing those up is how a run silently trains forever, so the
briefing prints both before you start.

## Architectures and heads

| arch | params | attention |
|---|---:|---|
| `edgehat` (default) | 4.02 M | local windows + one HAT global block |
| `swinir` | 4.90 M | windowed, relative position bias |
| `restormer` | 0.76 M | over channels (legacy) |
| `hybrid` | 0.76 M | pooled tokens, keeps older checkpoints (legacy) |

### Heads

| head | learns | params |
|---|---|---:|
| `freqsplit` (default) | low / mid / high bands separately | 63 k |
| `perpixel` | per-pixel RGB delta, unclamped until read | 42 k |
| `lut3d` | 33³ trilinear colour LUT **+ per-pixel delta** | 129 k |
| `tanh` | bounded delta, most conservative | 21 k |

Composite heads combine with `+`:

| composite | params | when |
|---|---:|---|
| `lut3d+freqsplit` | 192 k | **best default** — global grade + local band detail |
| `lut3d+perpixel` | 171 k | global grade + unconstrained local delta |
| `tanh+freqsplit` | 84 k | band-separated, everything bounded |

`perpixel+freqsplit` is deliberately absent: both are per-pixel, so it adds
parameters without adding a distinct inductive bias.

The 3D LUT uses **explicit trilinear gathers**, not `grid_sample` — see
[`docs/COREML.md`](docs/COREML.md) for why.

### Capabilities

Three, each independently switchable, each gated by a zero-initialised mix
scalar so a fresh model starts as a no-op:

| capability | what it does |
|---|---|
| **halation** | thresholded highlight, physically dilated spread, warm tint |
| **subject gate** | soft `S(x,y)`, guided by a Vision mask when supplied |
| **region exposure** | 4×4 coefficients, bilinearly upsampled |

The zero-init gate is what makes them safe to enable by default: the model
cannot regress on a capability it has not yet learned to use.

## Augmentation

One geometric warp per item, applied **identically to the source, the target and
the mask** — otherwise an unpaired batch describes two different scenes and the
content losses contradict each other.

```bash
--augment default        # scale 0.8-1.25, rot +/-12deg, hflip + vflip
--augment conservative   # scale 0.9-1.15, rot +/-6deg,  hflip + vflip
--augment strong         # wider, + colour jitter + 2 cutout holes
--augment none           # deterministic centre crop
```

Colour jitter is **off by default, on purpose**. The target domain *is* a colour
look; perturbing it teaches the discriminator that the style is fuzzy. Full
rationale and every bound in [`docs/TRAINING.md`](docs/TRAINING.md#augmentation).

## Common tasks

### Inference

```bash
# Whole frame (default): nothing cropped or stretched, exact original size
nf infer --input photo.jpg

# Side-by-side original | styled
nf infer --input photo.jpg --comparison

# Tiled, for real high-frequency detail. Match --tile to --image-size.
nf infer --input big.jpg --tile 256 --tile-overlap 64

# Downscale the long edge, aspect preserved, then tile
nf infer --input big.jpg --max-edge 1024 --tile 256

# A whole folder, two stochastic variants each
nf infer --input photos/ --seeds 2 --max-edge 1024
```

`--tile` should equal the `image_size` the model was trained at. A mismatch is a
distribution shift and the code warns about it. Measured cost on a 12 MP source:
whole-frame sees 0.08× scale, tiling at 25% overlap processes ~22 MP. Use
tiling for hero shots, not batches.

Full reference in [`docs/INFERENCE.md`](docs/INFERENCE.md).

### Training

```bash
# Two arms on two GPUs, in two notebook copies
nf train --source-dir ... --target-dir ... --name a --arch edgehat --epochs 60
nf train --source-dir ... --target-dir ... --name b --arch swinir  --epochs 60

# Isolate a capability's effect
nf train ... --no-halation

# Kaggle: publish a Dataset version every epoch, so a disconnect costs 1 epoch
nf train ... --epochs 60 --snapshot-every 1

# Add the transformer to a trained conv-only model (carries conv weights,
# randomises only the transformer, resets optimizer state)
python scripts/warm_start.py \
  --checkpoint runs/conv-pilot/checkpoints/latest.pt \
  --output-dir  runs/hybrid-warm

# Change resolution on an existing checkpoint (safe at any size)
nightflash-train ... --image-size 512 --batch-size 4 \
  --resume runs/hybrid/checkpoints/latest.pt
```

### Export

```bash
python -m nightflash.export_coreml \
  --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
  --output model.mlpackage --compute-units CPU_AND_GPU --half

python -m nightflash.validate_coreml \
  --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
  --coreml model.mlpackage --input photo.jpg
```

Always run the validator. All 28 arch × head combinations have been verified to
convert, with tensor-I/O MAE 0.00030 against PyTorch.

### Inspecting a run

```bash
nf brief  --name night      # full plan + numbered decision menu
nf state  --name night      # machine-readable JSON
nf config --name night      # resolved config, no side effects
```

`nf reset --name night` archives to `nightflash_freqsplit_night_archive_N`; it
never deletes without `--delete`.

See [`docs/WARM_START.md`](docs/WARM_START.md) for resume and warm-start detail.

## Running on Google Colab and Kaggle

Both notebooks are **seven cells**: install, data, configuration, train,
inspect, keep. Only the data plumbing and the snapshot lines differ.

| Platform | Notebook | Data | Cross-session survival |
|---|---|---|---|
| Colab | [`colab/NightFlash_Colab.ipynb`](colab/NightFlash_Colab.ipynb) | Google Drive | `runs_root` on Drive, nothing to snapshot |
| Kaggle | [`notebook/nightflash_kaggle.ipynb`](notebook/nightflash_kaggle.ipynb) | attached Dataset | `snapshot_every=1` + `s.auto_resume()` |

On Kaggle, `/kaggle/working` is wiped between sessions, so `snapshot_every=1`
publishes a Dataset version after every epoch and `s.auto_resume()` restores the
last one at the start of the next. Start with
[`docs/QuickStartKaggle.md`](docs/QuickStartKaggle.md).

Package the code for upload (datasets, runs and checkpoints are excluded):

```bash
./scripts/make_colab_bundle.sh     # → dist/nightflash-edge-colab.zip
```

The official Colab CLI (`pip install google-colab-cli`) and Colab MCP
(`uvx git+https://github.com/googlecolab/colab-mcp`) are both documented in
[`docs/COLAB.md`](docs/COLAB.md). The TUI does not run in either — use
`nightflash-tui` on your Mac.

## Dataset status

`training_photos/normal_photos/` and `validation_photos/` are populated by
`scripts/scrape_dataset.py`, which pulls openly-licensed images from Openverse,
screens them on measured pixel statistics, and records licence and attribution in
a `manifest.csv`.

| Folder | Images | `manifest.csv` | Status |
| --- | ---: | --- | --- |
| `normal_photos/` | 1000 | yes | Built — source domain |
| `validation_photos/` | 20 | yes | Built — held-out source images |
| `disposable_photos/` | 935 | **no** | **Not a usable target domain — see below** |

### ⚠️ The target domain is the blocker, not the model

`disposable_photos/` is the only folder the model learns a look *from*, and it is
not currently that look. Measured over all 935 images:

| mean luminance | images | share | reads as |
|---|---:|---:|---|
| 0.00 – 0.10 | 19 | 2.0% | near-black, probably astronomy |
| **0.10 – 0.20** | **68** | **7.3%** | **genuine night flash** |
| 0.20 – 0.35 | 425 | 45.5% | dim indoor / dusk |
| 0.35 – 0.50 | 307 | 32.8% | overcast daylight |
| 0.50 – 1.00 | 116 | 12.4% | full daylight |

Mean luminance is **0.350**, median **0.334**. Genuine night flash sits below
0.20. So:

- only **9.3%** (87 of 935) of the folder is in the target range;
- **45.2%** (423 of 935) is above 0.35, i.e. daylit;
- filenames are daylight travel and outdoor stock —
  `young-woman-exploring-the-grand-palace-in-bangkok.jpg`,
  `scenic-desert-road-along-colorado-river.jpg`;
- median long edge is 579 px, and 27% are under 512 px.

A PatchGAN learns whatever distribution this folder contains. Training on it
teaches the **opposite** of Night Flash — a slightly hazy, desaturated daylight
look — and no amount of epochs, architecture, or tiling will fix that.

Two further gaps:

- **No `manifest.csv`**, so licensing and attribution for those 935 images are
  unrecorded. Resolve before anything ships.
- **No held-out target set.** Without one, target-domain quality is
  unmeasurable and you cannot tell progress from memorisation.

The recommended fix is to **generate** the target set by running the app's
existing deterministic `NightFlashProcessor` over `normal_photos/`. That gives
unlimited correctly-labelled, correctly-licensed pairs-by-construction, and it
converges far faster than scraping. See
[`docs/DATASET_SOURCING.md`](docs/DATASET_SOURCING.md).

You can re-run the measurement yourself before spending GPU hours — the snippet
is in [`docs/QuickStartKaggle.md`](docs/QuickStartKaggle.md#before-you-start-the-target-domain).

Note: `training_photos/disposable_photos/README.md` still describes the folder
as intentionally empty. It is the user's file and has been left untouched.

## Important constraints

- A Core ML LoRA file is small, but the diffusion base model is still large. This repository instead trains a compact standalone model.
- Unpaired training cannot mathematically guarantee exact identity preservation. Use content, identity, face, and edge metrics.
- Train and evaluate with the same color pipeline used in deployment.
- Keep random grain and light leaks seeded and outside the neural network when possible.
- The model is most useful for a narrow, fixed Night Flash aesthetic. Use separate checkpoints or a style embedding for multiple distinct looks.

## License

The project code is intended to be released under the MIT license. Check the licenses of all datasets, pretrained weights, and model-derived assets separately before shipping a product.
