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
3. [`docs/TRAINING.md`](docs/TRAINING.md) — local training and mid-training inference
4. [`docs/COLAB.md`](docs/COLAB.md) — Google Colab workflow
5. [`docs/INFERENCE.md`](docs/INFERENCE.md) — testing, seeds, and effect controls
6. [`docs/EVALUATION.md`](docs/EVALUATION.md) — quality and performance evaluation
7. [`docs/COREML.md`](docs/COREML.md) — Core ML export and iOS integration

## Important constraints

- A Core ML LoRA file is small, but the diffusion base model is still large. This repository instead trains a compact standalone model.
- Unpaired training cannot mathematically guarantee exact identity preservation. Use content, identity, face, and edge metrics.
- Train and evaluate with the same color pipeline used in deployment.
- Keep random grain and light leaks seeded and outside the neural network when possible.
- The model is most useful for a narrow, fixed Night Flash aesthetic. Use separate checkpoints or a style embedding for multiple distinct looks.

## License

The project code is intended to be released under the MIT license. Check the licenses of all datasets, pretrained weights, and model-derived assets separately before shipping a product.
