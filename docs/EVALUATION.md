# Evaluation guide

Unpaired training requires more than one metric. A model can match target color statistics while destroying the subject.

## Held-out validation during training

`paired` and `sameset` hold out part of the target pool and score it every epoch.
This is a different mechanism from `nightflash.evaluate`: it runs automatically,
writes a strip you look at, and feeds checkpoint selection.

```bash
nf train --target-dir POOL --epochs 60 --val-every 1 --val-count 8
```

```
runs/<name>/validations/
  epoch-0007.png   input | output | target, one image per row
  latest.png       a copy, for quick viewing
  history.csv      epoch,images,psnr,ssim,l1,effect
```

Read the strip, not the loss curve. The discriminator loss is not a quality
signal and will fall while the output gets worse.

### What each number means

| column | meaning | trustworthy in |
|---|---|---|
| `psnr` | peak signal-to-noise ratio of the deterministic output against the true target | `paired` only |
| `ssim` | structural similarity, same comparison | `paired` only |
| `l1` | mean absolute difference, same comparison | `paired` only |
| `effect` | mean magnitude of the degradation parameters in the batch | all modes |

In `sameset` the target is a *different* photograph, so `psnr`, `ssim` and `l1`
are computed between unrelated images and mean nothing as quality measures. They
are still recorded so the columns line up across arms, but **do not rank
checkpoints on them** — which is why `best.pt` is not written in `sameset`.

For `sameset`, the strip is the evaluation. Read it, and use distribution-level
metrics if you need a number.

### Two determinisms

- `--val-select fixed` (default) keeps the same images every epoch, so per-image
  progress is readable. `random` answers "is the model still healthy on
  *arbitrary* images", which is what you want when you suspect memorisation.
- The degradation and effect seeds are pinned per image via SHA-256 of the
  filename, unconditionally. Without this, epoch 7 and epoch 8 would be different
  problems and any improvement would be noise.

### Model selection

`best.pt` is written whenever validation PSNR improves, and `--keep-last` prunes
old per-epoch checkpoints.

This exists because the alternative is unrecoverable: with only `latest.pt`, a run
that peaks at epoch 40 and decays by 60 has already thrown away its best weights.

**Export `best.pt`, not `latest.pt`**, unless the history shows PSNR still rising
at the final epoch.

---

## Built-in diagnostics

Run:

```bash
nightflash-evaluate \
  --checkpoint runs/hybrid/checkpoints/latest.pt \
  --source-dir data/validation/source \
  --target-dir data/validation/target \
  --output-json runs/hybrid/evaluation.json \
  --limit 200
```

The report includes:

- Content L1
- Edge L1
- Simple global SSIM approximation
- Mean effect parameters
- Source/target RGB means
- Variation across several seeds

These are diagnostics, not release criteria.

## Content preservation

Measure:

- DINO or CLIP image similarity
- Face embedding similarity
- Landmark error
- Edge/gradient correlation
- OCR accuracy on text-heavy images
- Human identity judgments

A style model should preserve content even when the target domain has strong artistic changes.

## Target-domain quality

You need a **held-out target set** — night-flash photographs the model never
trained on. Without one, target-domain quality is unmeasurable and you cannot
distinguish progress from memorisation. Keep it out of `target_dir`; the built-in
diagnostics take a separate `--target-dir` for exactly this.

```bash
python -m nightflash.evaluate \
  --checkpoint runs/edgehat_freqsplit_night/checkpoints/latest.pt \
  --source-dir training_photos/normal_photos \
  --target-dir training_photos/val_night \
  --output-json runs/edgehat_freqsplit_night/evaluations/score.json
```

A note on `--val-dir`: `Session` accepts it, records it in `config.json` and
prints it in the briefing, but the training loop does not consume it. Run
`nightflash.evaluate` explicitly.

Then use:

- FID/KID against the target domain
- LPIPS or DISTS against target references
- Color histogram distance
- Mean/std color statistics
- Highlight and shadow clipping
- Human pairwise preference

FID alone does not tell you whether the model preserved the input scene.

**Check the held-out set is actually the target domain** before trusting any of
these numbers. FID between two distributions is low when both are *equally
wrong* — it measures similarity, not correctness. The luminance check in
[`DATA.md`](DATA.md#check-the-target-domain-before-you-train) applies to the
held-out set too.

## Stochastic quality

Evaluate:

- Variation across seeds
- Variation across different input scenes
- Grain spatial frequency
- Light-leak spatial smoothness
- Effect amplitude distribution
- Mode collapse
- Temporal flicker

For video, add:

- Optical-flow warping error
- Feature variance across frames
- Face identity stability
- Color stability
- Ghosting around motion

## Suggested release gates

A practical first release gate:

```text
No severe face or object changes
No obvious text corruption
Target color/style preferred in blind A/B testing
No mode collapse across at least 20 validation scenes
Stable output across five seeds
Core ML matches PyTorch within visual tolerance
Real-time latency meets the device target
```

## Comparing checkpoints

Use the same:

- Validation images
- Image size
- Seed
- Effect strength
- Color management
- Output format

Do not compare a 256px preview with a 512px export, or a deterministic model with a high-effect-strength model.

## Human test set

Create a small fixed set containing:

- Day portraits
- Night portraits
- Groups
- Bright skies
- Dark interiors
- Text/signs
- Motion
- Reflections
- Low-light noise
- Different skin tones

Ask reviewers to answer separately:

1. Does the output resemble the target camera/style?
2. Is the original subject/content preserved?
3. Are the stochastic effects plausible?
4. Would you save this image?
