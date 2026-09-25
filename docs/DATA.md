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

## No paired images

The training code does not require:

```text
source[i] ↔ target[i]
```

For every source image, the loader samples a random target image. The discriminator learns the target-domain distribution, while content/identity losses constrain the source-to-output relationship.

This is unpaired domain translation, not paired regression.

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
