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
