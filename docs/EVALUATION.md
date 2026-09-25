# Evaluation guide

Unpaired training requires more than one metric. A model can match target color statistics while destroying the subject.

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

Use:

- FID/KID against the target domain
- LPIPS or DISTS against target references
- Color histogram distance
- Mean/std color statistics
- Highlight and shadow clipping
- Human pairwise preference

FID alone does not tell you whether the model preserved the input scene.

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
