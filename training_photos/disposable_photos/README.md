# Target domain — needs real photographs

This folder is intentionally **empty**. It is the `target/` domain for the
unpaired Night Flash training run, and it could not be filled by scraping.

## Why it is empty

The requested look — cheap-film direct flash at night — is a *visual*
aesthetic, not a keyword. Searching CC-licensed image indexes for it returns
astronomy, macro studio shots, infrared trail-camera wildlife, fireworks and
neon instead. A calibration pass that downloaded and pixel-screened 253 images
kept 60, and inspecting them showed only about 5–10% were genuine flash
snapshots.

Training on that set would actively hurt the model: the PatchGAN discriminator
learns the distribution of this folder, so it would learn "black background with
one bright object" (the astronomy/macro look) rather than the Night Flash look.

Full measurements and reasoning: [`docs/DATASET_SOURCING.md`](../../docs/DATASET_SOURCING.md).

## What to put here

Real disposable-camera photographs taken at night. 300–500 genuine frames will
outperform any scraped set. Scan or photograph a roll if you have one.

```text
disposable_photos/
  roll01_001.jpg
  roll01_002.jpg
  ...
```

Aim for variety — mix subjects, distances, indoor/outdoor, and flash intensity.
A roll of near-identical flash portraits will teach one narrow style.

## Before training

Check the folder is what you think it is:

```bash
python scripts/scrape_dataset.py --domain disposable --count 500
open training_photos/.cache/sheet_disposable.jpg   # contact sheet of the set
```

The screen used here removes daylight, blown-out and macro-on-black frames. It
is a sanity check on real photos, not a way to manufacture the look from search
results.
