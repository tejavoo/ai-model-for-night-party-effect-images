# Unpaired data guide

## Directory layout

```text
dataset/
  source/
    0001.jpg
    0002.jpg
    nested/
      0003.jpg

  target/
    0001.jpg
    0002.jpg
    nested/
      0003.jpg

  source_masks/       # optional
    0001.png
    nested/
      0003.png
```

The loader accepts:

```text
.jpg .jpeg .png .webp .bmp .tif .tiff .heic
```

HEIC support depends on the Pillow build installed in your environment.

## What belongs in `source/`

Use ordinary images representing the content domain you expect in the app:

- Portrait photos
- Group photos
- Streets and buildings
- Interiors
- Day and night scenes
- Near and far subjects
- Bright highlights and dark shadows
- Motion and low-light images
- Text, signs, faces, and small objects

Do not use only web images that already contain the target look.

## What belongs in `target/`

Use images that represent the desired output domain:

- Real disposable-camera photographs, if available
- Scans or photographs of a desired film aesthetic
- A consistent camera/film treatment
- Varied scenes and exposure levels
- Natural grain and highlight behavior

Avoid:

- Instagram filters with logos
- Borders or date stamps if those are not part of the target
- Heavy JPEG compression
- Images all from one person or location
- Scans with scanner color casts that you do not want

### Check the target domain before you train

This is the single most common way to waste GPU hours. A PatchGAN learns whatever
distribution the target folder contains, so a wrong folder does not fail — it
quietly teaches the wrong look.

For a Night Flash target, **mean luminance should be below 0.20**. Daylit
photography sits above 0.35. Measure before training:

```python
import numpy as np
from pathlib import Path
from PIL import Image

lums, sizes = [], []
for p in sorted(Path("target").glob("*")):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
        continue
    im = Image.open(p).convert("L")
    lums.append(np.asarray(im, np.float32).mean() / 255.0)
    sizes.append(max(im.size))

lums, sizes = np.array(lums), np.array(sizes)
print(f"images            {len(lums)}")
print(f"mean luminance    {lums.mean():.3f}  (night flash: < 0.20)")
print(f"under 0.20        {(lums < 0.20).mean():.0%}")
print(f"over 0.35         {(lums > 0.35).mean():.0%}   (want: very few)")
print(f"median long edge  {np.median(sizes):.0f}px  (want: >= 512)")
```

If `over 0.35` is high, the folder holds daylight photos and no amount of
training will fix it. If the median long edge is under 512, the target is also
too small to carry grain and halation detail.

The current state of `training_photos/disposable_photos/` fails this check — see
the dataset status table in [`../README.md`](../README.md#dataset-status) and
[`DATASET_SOURCING.md`](DATASET_SOURCING.md).

### Generating the target set instead

If you cannot source enough genuine photographs, generate them by running the
app's existing deterministic processor over the source domain. That gives
unlimited, correctly-labelled, correctly-licenced targets by construction, and it
converges much faster than scraping. See
[`DATASET_SOURCING.md`](DATASET_SOURCING.md#what-will-work).

## No paired images

The training code does not require:

```text
source[i] ↔ target[i]
```

For every source image, the loader samples a random target image. The discriminator learns the target-domain distribution, while content/identity losses constrain the source-to-output relationship.

This is unpaired domain translation, not paired regression.

One consequence worth knowing: because the two images are unrelated, the loader
applies a **single shared geometric warp** to the source, the target and the
mask each time. If they were warped independently the discriminator would see two
different scenes and the content losses would contradict each other. See
[`ARCHITECTURE.md`](ARCHITECTURE.md#data-pipeline).

## Train/validation split

Keep a held-out set before training:

```text
dataset/
  train/
    source/
    target/
  val/
    source/
    target/
```

Split by scene or recording session. Do not randomly split adjacent video frames.

A simple rule:

- If the same person appears in train and validation, results will look better than they will in production.
- If adjacent frames are split across train and validation, metrics will be misleading.
- Hold out entire scenes, cameras, or days where possible.

## Masks

Masks are optional. When used, each source mask should align with the source image:

```text
source/photo.jpg
source_masks/photo.png
```

The mask is resized with nearest-neighbor interpolation and aligned to the source crop.

The existing iOS app generates masks with Vision. For Colab, precompute masks with a compatible model such as a person/foreground segmentation network and save them as grayscale PNGs.

Missing masks are treated as all-zero masks.

## Image preprocessing

The training loader:

1. Applies EXIF orientation.
2. Converts to RGB.
3. Resizes the shorter side to the selected model size.
4. Crops to a square.
5. Randomly crops during training.
6. Randomly flips source and target independently.
7. Converts pixels to `[0, 1]` tensors.

The inference command resizes with a center fit and restores the original output dimensions.

Keep one consistent preprocessing policy for training, validation, and deployment.

## Dataset quality checklist

Before training, verify:

- [ ] Source and target directories contain different images
- [ ] Target set represents one coherent aesthetic
- [ ] Source set represents the app's expected content domain
- [ ] No accidental duplicates dominate either domain
- [ ] No corrupted images
- [ ] Consistent color profile where possible
- [ ] Held-out scenes exist
- [ ] Target images have enough lighting and subject variation
- [ ] Dataset usage rights are clear

## Optional dataset balancing

If one target category dominates, balance it:

- Cap repeated subjects/cameras
- Sample categories uniformly
- Keep a separate style-reference subset
- Report results per category

A model trained on only night portraits may look excellent on night portraits and fail on daylight landscapes.
