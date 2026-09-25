# Recommended plan

## Goal

Build a small, deterministic, unpaired image-to-image model that improves the existing Night Flash camera look while preserving the subject and working on Apple hardware.

The model should learn:

- Adaptive color and exposure behavior
- Subject-aware tonal treatment
- Nonlinear film-like response

It should not be responsible for every camera artifact. Keep deterministic optical effects in Core Image/Metal.

## Phase 0 — Establish the baseline

Before training a neural model, measure the current app:

```text
original image
→ existing 1×1 Core ML transform
→ existing Vision/Core Image pipeline
```

Record:

- Mean/std RGB values
- Highlight and shadow clipping
- Processing time
- Output dimensions
- Behavior on dark, bright, portrait, and landscape scenes

This becomes the quality/performance baseline.

## Phase 1 — Build the data domains

You need two **unpaired** directories:

```text
source/
  normal photographs
  faces, streets, interiors, daylight, night, etc.

target/
  desired disposable-camera photographs
  varied scenes, exposure levels, grain, and lighting
```

Do not make the target set contain only one person, one camera, or one scene. The model will learn those biases.

Recommended initial collection:

| Split | Target images | Source images |
|---|---:|---:|
| Pilot | 100–500 | 1,000–5,000 |
| Normal training | 500–2,000 | 5,000–20,000 |
| Larger dataset | 2,000+ | 20,000+ |

These are starting points, not hard requirements.

See [`DATA.md`](DATA.md).

## Phase 2 — Train the convolutional baseline

Start with `--no-transformer`.

Why:

- It is easy to debug
- It exports cleanly
- It establishes whether unpaired learning works at all
- It gives a fair speed/quality baseline

Pilot command:

```bash
nightflash-train \
  --source-dir data/source \
  --target-dir data/target \
  --output-dir runs/conv-baseline \
  --image-size 256 \
  --batch-size 8 \
  --epochs 30 \
  --no-transformer \
  --preview-every 250
```

Inspect previews in:

```text
runs/conv-baseline/previews/
```

Test a checkpoint before continuing:

```bash
nightflash-infer \
  --checkpoint runs/conv-baseline/checkpoints/latest.pt \
  --input data/source/example.jpg \
  --output outputs/conv-baseline.png
```

## Phase 3 — Add global transformer context

If the convolutional model handles color but misses global relationships, enable the low-resolution transformer bottleneck:

```bash
nightflash-train \
  --source-dir data/source \
  --target-dir data/target \
  --output-dir runs/hybrid-transformer \
  --image-size 256 \
  --batch-size 4 \
  --epochs 50 \
  --transformer-blocks 1 \
  --transformer-token-size 16
```

The transformer is pooled to a small fixed token grid. Do not apply full attention to a 512×512 feature map.

Compare against the convolutional baseline using the same checkpoint step, data split, and inference settings.

## Phase 4 — Add person-mask conditioning

The existing app already has a Vision person mask. For a mask-aware model:

1. Export person masks for the source images.
2. Store them in a parallel directory.
3. Train with `--mask-dir`.
4. Feed RGB + mask as a four-channel model input.

```bash
nightflash-train \
  --source-dir data/source \
  --target-dir data/target \
  --mask-dir data/source_masks \
  --output-dir runs/mask-aware \
  --image-size 256 \
  --batch-size 4
```

A missing mask becomes an all-zero mask. Prefer explicit mask errors in a production data pipeline.

## Phase 5 — Tune stochastic effects

The model predicts eight effect coefficients. The renderer applies them with a seed.

Test multiple seeds:

```bash
for seed in 1 2 3 4 5; do
  nightflash-infer \
    --checkpoint runs/hybrid-transformer/checkpoints/latest.pt \
    --input data/source/example.jpg \
    --output outputs/seed-${seed}.png \
    --seed "${seed}"
done
```

Use `--no-effects` to inspect only the deterministic learned residual.

If effects are too strong:

```bash
--effects-strength 0.25
```

If effects are too subtle:

```bash
--effects-strength 0.75
```

Do not increase the strength before checking the effect coefficient distribution.

## Phase 6 — Train at deployment resolution

Once the 256px model is stable, train at 512px:

```bash
nightflash-train \
  --source-dir data/source \
  --target-dir data/target \
  --output-dir runs/hybrid-512 \
  --image-size 512 \
  --batch-size 2 \
  --epochs 50 \
  --resume runs/hybrid-transformer/checkpoints/latest.pt
```

A model trained at 256 and exported at 512 is not automatically valid. Either train at the deployment resolution or explicitly validate resolution transfer.

## Phase 7 — Export and benchmark Core ML

On macOS:

```bash
python -m pip install -e '.[coreml]'

nightflash-export-coreml \
  --checkpoint runs/hybrid-512/checkpoints/latest.pt \
  --output runs/hybrid-512/NightFlashEdge.mlpackage \
  --image-size 512 \
  --half
```

Benchmark on:

- M3 Ultra Mac
- Target iPhone generation
- Cold-start model load
- p50/p95 latency
- Peak memory
- Camera frame rate
- Battery/thermal behavior

See [`COREML.md`](COREML.md).

## Phase 8 — Integrate into the existing app

Recommended app order:

```text
AVFoundation frame
→ resize/normalize
→ Vision person mask
→ NightFlashEdge.mlpackage
→ effect coefficients
→ Core Image/Metal finishing
→ Photos/share
```

The current app's 1×1 model can remain in place for the first integration, or be replaced after the new model has been validated.

## Decision gates

Do not move to the next phase unless:

- The model beats the current pipeline on style similarity.
- Faces and object structure remain recognizable.
- The model does not create obvious color artifacts.
- Different seeds create plausible variation.
- The model is stable across held-out scenes.
- Core ML output matches PyTorch output within a chosen tolerance.

## Recommended final stack

```text
Small hybrid residual model
+ existing Vision person segmentation
+ seeded procedural grain/light leaks
+ existing Core Image/Metal finishing
```

This gives you a small learned model without the cost and unpredictability of a full diffusion pipeline.
