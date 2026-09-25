# Inference and mid-training testing

## Single image

```bash
nightflash-infer \
  --checkpoint runs/hybrid/checkpoints/latest.pt \
  --input data/validation/source/photo.jpg \
  --output outputs/photo-nightflash.png \
  --seed 12345
```

The command:

1. Loads the checkpoint and model configuration.
2. Applies EXIF orientation.
3. Fits the image to the model resolution.
4. Runs the generator.
5. Applies seeded stochastic effects.
6. Resizes the result back to the original dimensions.
7. Saves the output and a small JSON inference record.

## Directory inference

```bash
nightflash-infer \
  --checkpoint runs/hybrid/checkpoints/latest.pt \
  --input data/validation/source \
  --output outputs/validation \
  --seed 12345
```

## Deterministic-only inference

Inspect the learned color/tone component without procedural effects:

```bash
nightflash-infer \
  --checkpoint runs/hybrid/checkpoints/latest.pt \
  --input data/validation/source/photo.jpg \
  --output outputs/deterministic.png \
  --no-effects
```

This is the most useful test when debugging whether a problem comes from the model or the stochastic renderer.

## Seeded stochastic inference

The effect parameters are deterministic for a given input. The random grain/leak realization changes with the seed.

```bash
for seed in 1 2 3 4 5; do
  nightflash-infer \
    --checkpoint runs/hybrid/checkpoints/latest.pt \
    --input data/validation/source/photo.jpg \
    --output "outputs/seed-${seed}.png" \
    --seed "${seed}"
done
```

Use one seed per captured shot if you want reproducible results. For video, smooth or interpolate the effect seed over time rather than choosing a new unrelated seed every frame.

## Effect strength

```bash
--effects-strength 0.0
--effects-strength 0.25
--effects-strength 0.50
--effects-strength 0.75
```

`0.0` is equivalent to `--no-effects` for the rendered result.

## Mid-training checkpoints

Training writes:

```text
runs/hybrid/checkpoints/latest.pt
runs/hybrid/checkpoints/epoch-0001.pt
runs/hybrid/checkpoints/epoch-0002.pt
```

The `latest.pt` file is intentionally overwritten so it is easy to test from the TUI. Copy a known-good checkpoint before starting a new experiment:

```bash
cp runs/hybrid/checkpoints/latest.pt runs/hybrid/checkpoints/best-known-good.pt
```

## Effect diagnostics

The effect vector contains eight values:

```text
0 grain strength
1 light-leak strength
2 exposure offset
3 saturation
4 vignette strength
5 leak red
6 leak green
7 leak blue
```

They are sigmoid-valued and normally should not remain pinned at 0 or 1.

If every value is identical across all images, the effect head has probably learned a constant style. That may be acceptable for a fixed look, but it is not learning input-dependent variation.

## Comparison workflow

For every candidate checkpoint, run the same validation set:

```bash
for checkpoint in runs/hybrid/checkpoints/*.pt; do
  name=$(basename "$checkpoint" .pt)
  nightflash-infer \
    --checkpoint "$checkpoint" \
    --input data/validation/source \
    --output "outputs/${name}" \
    --seed 12345
done
```

Compare:

- Target color
- Face identity
- Text and edges
- Highlight clipping
- Grain realism
- Variation across seeds
- Temporal stability if testing video

## Performance testing

Measure inference separately from camera capture. Record:

- Model load time
- First inference latency
- p50/p95 latency
- Peak memory
- Device temperature
- Battery impact
- Frame rate under continuous capture

The same checkpoint can behave very differently on M3 Ultra, MacBook, and iPhone.
