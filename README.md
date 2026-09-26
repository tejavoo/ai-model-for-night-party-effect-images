# Night Flash Edge

A small, open-source training and deployment toolkit for learning a **disposable-camera / Night Flash look from unpaired image domains**.

The project is designed around the existing native app:

```text
/Users/tejavu/Documents/opencode-experiments/disposable-cam
```

The app already has a deterministic Core ML color transform plus Core Image/Vision finishing. This repository adds a separate, compact learned model that can learn a more adaptive style without requiring paired source/target images.

## What this repository contains

- Hybrid convolutional + MobileViT/EdgeViT-style generator
- Bounded RGB residual prediction
- Small stochastic-effect coefficient head
- Seeded procedural grain, leak, exposure, saturation, and vignette renderer
- Unpaired PatchGAN training
- Content, identity, edge, and batch-statistics losses
- Mid-training preview generation
- Checkpoint/resume support
- Single-image and batch-directory inference
- Evaluation diagnostics
- Core ML export to `.mlpackage` and `.mlmodel`
- Optional Textual terminal UI
- Colab and local documentation

## Design decision

The recommended architecture is deliberately hybrid:

```text
source image
→ convolutional encoder
→ small low-resolution global transformer
→ residual decoder
→ learned style residual
→ seeded stochastic effects
→ existing Core Image/Metal finishing
```

The model is not intended to replace the entire camera pipeline. Keep the existing Vision person mask and Core Image effects in the app.

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

Train on real unpaired data:

```bash
nightflash-train \
  --source-dir /path/to/normal-photos \
  --target-dir /path/to/disposable-photos \
  --output-dir runs/nightflash \
  --image-size 256 \
  --batch-size 8 \
  --epochs 50
```

Run inference from a checkpoint:

```bash
nightflash-infer \
  --checkpoint runs/nightflash/checkpoints/latest.pt \
  --input /path/to/test-photo.jpg \
  --output outputs/test-photo-nightflash.png \
  --seed 12345
```

Run the terminal UI:

```bash
nightflash-tui
```

## Documentation

Read these in order:

1. [`docs/PLAN.md`](docs/PLAN.md) — recommended implementation phases
2. [`docs/DATA.md`](docs/DATA.md) — unpaired dataset preparation
3. [`docs/DATASET_SOURCING.md`](docs/DATASET_SOURCING.md) — how `training_photos/` was built, and the target-domain gap
4. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the `edgehat` model, heads, and capabilities
5. [`docs/CLI.md`](docs/CLI.md) — `Session`, the pre-flight briefing, and the `nf` command
6. [`docs/TRAINING.md`](docs/TRAINING.md) — local training and mid-training inference
7. [`docs/COLAB.md`](docs/COLAB.md) — Google Colab notebook, CLI, and MCP
8. [`docs/KAGGLE.md`](docs/KAGGLE.md) — Kaggle setup, T4×2, cross-session resume
9. [`docs/WARM_START.md`](docs/WARM_START.md) — resuming, warm starting, changing resolution
10. [`docs/INFERENCE.md`](docs/INFERENCE.md) — testing, seeds, aspect ratio
11. [`docs/EVALUATION.md`](docs/EVALUATION.md) — quality and performance evaluation
12. [`docs/COREML.md`](docs/COREML.md) — Core ML export and iOS integration

## Quick start: one object

```python
from nightflash import Session
from nightflash.briefing import brief

s = Session(
    arch="edgehat", head="freqsplit", name="night",
    source_dir="data/source", target_dir="data/target",
    image_size=256, batch_size=8, epochs=50,
)

print(brief(s, mode="plan"))   # read this before training
s.train()                       # resume, or start fresh
s.train(extra_epochs=50)        # resume and add 50 more
s.infer("photo.jpg", seeds=3)   # inference only, no training
```

The same thing from a terminal:

```bash
nf brief
nf train --source-dir data/source --target-dir data/target --epochs 50
nf infer --input photo.jpg --seeds 3
```

Each run gets its own folder, so switching architectures never overwrites
another run:

```text
runs/edgehat_freqsplit_night/
  config.json  checkpoints/  previews/  inferences/  evaluations/  logs/
```

## Architectures and heads

| arch | params | attention |
|---|---:|---|
| `edgehat` (default) | 4.02 M | local windows + one HAT global block |
| `swinir` | 4.90 M | windowed, relative position bias |
| `restormer` | 0.76 M | over channels |
| `hybrid` | 0.76 M | pooled tokens, keeps older checkpoints |

| head | learns | params |
|---|---|---:|
| `freqsplit` (default) | low / mid / high bands separately | 63 k |
| `perpixel` | unbounded per-pixel delta | 42 k |
| `lut3d` | 33³ colour LUT **+ per-pixel conv** | 129 k |
| `tanh` | bounded delta, most conservative | 21 k |

Composite heads combine with `+`:

| composite | params | when |
|---|---:|---|
| `lut3d+freqsplit` | 192 k | **best default** — global grade + local band detail |
| `lut3d+perpixel` | 171 k | global grade + unconstrained local delta |
| `tanh+freqsplit` | 84 k | band-separated, everything bounded |

Three capabilities, each independently switchable:

- **halation** — thresholded highlight, dilated spread, warm tint
- **subject gate** — soft `S(x,y)`, guided by a Vision mask when supplied
- **region exposure** — 4×4 coefficients, bilinearly upsampled

## Common tasks

```bash
# Inference: whole frame (nothing cropped, but downscaled to --image-size)
nf infer --input photo.jpg --output out.png

# Inference: tiled, for real high-frequency detail
nf infer --input big.jpg --output big-t512.png --tile 512 --tile-overlap 128

# Train with a composite head
nf train --source-dir ... --target-dir ... --head lut3d+freqsplit --epochs 50
```

```bash
# Train
nightflash-train --source-dir ... --target-dir ... --output-dir runs/hybrid

# Notebook / piped output: one clean line per update, no tqdm bar
--progress plain            # or: --progress tqdm | --progress none

# Continue the same architecture (--epochs is ABSOLUTE, not "extra epochs")
nightflash-train ... --resume runs/hybrid/checkpoints/latest.pt --epochs 80

# Add the transformer to a trained conv-only model (carries conv weights,
# randomises only the transformer, resets optimizer state)
python scripts/warm_start.py \
  --checkpoint runs/conv-pilot/checkpoints/latest.pt \
  --output-dir  runs/hybrid-warm

# Change resolution on an existing checkpoint (safe at any size)
nightflash-train ... --image-size 512 --batch-size 4 --resume runs/hybrid/checkpoints/latest.pt
```

See [`docs/WARM_START.md`](docs/WARM_START.md) for the full explanation of each.

## Running on Google Colab and Kaggle

Both notebooks are **six cells**, and cells 3 to 6 are byte-identical. Only the
data plumbing differs.

| Platform | Notebook | Data | Cross-session resume |
|---|---|---|---|
| Colab | [`colab/NightFlash_Colab.ipynb`](colab/NightFlash_Colab.ipynb) | Google Drive | writes straight to Drive |
| Kaggle | [`notebook/nightflash_kaggle.ipynb`](notebook/nightflash_kaggle.ipynb) | attached Dataset | `snapshot()` / `restore()` |

Package the code for upload (dataset and checkpoints are excluded):

```bash
./scripts/make_colab_bundle.sh     # → dist/nightflash-edge-colab.zip
```

The official Colab CLI (`pip install google-colab-cli`) and Colab MCP
(`uvx git+https://github.com/googlecolab/colab-mcp`) are both documented in
[`docs/COLAB.md`](docs/COLAB.md). Kaggle specifics, including the T4×2 tip, are in
[`docs/KAGGLE.md`](docs/KAGGLE.md). The TUI does not run in either — use
`nightflash-tui` on your Mac.

## Dataset status

`training_photos/normal_photos/` and `validation_photos/` are populated by
`scripts/scrape_dataset.py`, which pulls openly-licensed images from Openverse,
screens them on measured pixel statistics, and records license and attribution in
a `manifest.csv`.

| Folder | Images | Status |
| --- | ---: | --- |
| `normal_photos/` | 1000 | Built — source domain |
| `validation_photos/` | 20 | Built — held-out source images |
| `disposable_photos/` | 45 | **Target domain, but unverified and too small** |

> **⚠️ Open question on the target domain.** `docs/DATASET_SOURCING.md` and
> `training_photos/disposable_photos/README.md` both state that this folder is
> intentionally empty, because a calibration pass over 253 scraped images kept
> only ~5–10% genuine night-flash snapshots and the rest were astronomy and macro
> imagery. 45 image files are now present there, but no `manifest.csv` records
> their provenance, and the folder README still describes it as empty.
>
> Confirm those 45 images really are the Night Flash look before training on
> them. A PatchGAN learns whatever distribution this folder contains, so if it is
> the rejected scrape, the model will learn "black background with one bright
> object" rather than the target aesthetic.
>
> 45 images is also too few to avoid overfitting. See
> [`docs/DATASET_SOURCING.md`](docs/DATASET_SOURCING.md).

## Important constraints

- A Core ML LoRA file is small, but the diffusion base model is still large. This repository instead trains a compact standalone model.
- Unpaired training cannot mathematically guarantee exact identity preservation. Use content, identity, face, and edge metrics.
- Train and evaluate with the same color pipeline used in deployment.
- Keep random grain and light leaks seeded and outside the neural network when possible.
- The model is most useful for a narrow, fixed Night Flash aesthetic. Use separate checkpoints or a style embedding for multiple distinct looks.

## License

The project code is intended to be released under the MIT license. Check the licenses of all datasets, pretrained weights, and model-derived assets separately before shipping a product.
