# Dataset sourcing: what works and what does not

This file records how `training_photos/` was populated, what was measured, and —
importantly — which part of the dataset could **not** be built by scraping and
why. Read this before training.

## Layout produced

```text
training_photos/
  normal_photos/        # source domain
  disposable_photos/    # target domain
  validation_photos/    # held-out source images
  .cache/               # API response cache + contact sheets (safe to delete)
```

Every kept image has a row in `manifest.csv` next to it recording the source
URL, creator, license and license URL. `CC BY` images require attribution if you
redistribute the dataset or ship it with a model.

## Sources used, and why

Images come from **Openverse** (`api.openverse.org`), the CC-licensed index over
Flickr, Wikimedia and others. It was chosen over scraping sites directly because
it provides machine-readable license and attribution metadata and an explicit
`mature=false` filter.

**Not used, deliberately:**

| Source | Why not |
| --- | --- |
| Tumblr | Terms of service prohibit automated access, the third-party API is gated, and bulk pulls carry a large volume of adult content that cannot be reliably filtered at this scale. |
| Flickr HTML | Blocked and against terms of service. The compliant path is the official API (needs a free key), or Openverse, which already indexes Flickr's CC content. |
| Wikimedia Commons | Works, but its search is textual and returns even worse results for aesthetic queries. |
| Random web images | No license metadata, so usage rights cannot be established. |

## The source domain works

`normal_photos/` is built from ordinary daylight/indoor photography queries
(street, cafe, market, kitchen, hiking, beach, farm, bicycle …). Keyword search
is genuinely appropriate here: the population is enormous, homogeneous, and
matches what the text describes. Screening only has to remove the minority that
drift dark, and the pass rate is high. 1,000 images, 423 distinct creators,
0 corrupt files.

## The held-out set

`validation_photos/` uses *different queries* from training (picnic, barber shop,
train platform, bakery) so the split is by scene rather than by adjacent frame,
and it refuses any image whose perceptual hash is within 6 bits of a training
image.

Note that held-out sets are **sampled, not ranked**. An earlier version took the
top 20 by screen score; because that score rewards sharpness and contrast, the
result was 20 near-identical group photographs of people in formal uniforms —
useless for measuring anything, because it did not resemble the training
distribution. `balanced_order()` now round-robins across queries and shuffles
within each.

Two known limitations:

- The set is drawn from 4 of the 8 planned queries. The anonymous API allowance
  of 200 requests/day was exhausted by the source-domain build, and the cache
  had no pages for the remaining four. Re-running on a later day fills them in
  for free from cache, or immediately with an API key.
- Bakery and shop-interior imagery is over-represented (roughly 8 of 20).
  Round-robin balances *queries*, not *subjects*, and those two queries return
  repetitive shop interiors. With more queries this dilutes; otherwise drop
  some by hand.

## The target domain could not be scraped

**The requested "photos captured at night by a disposable camera" set does not
exist in CC-licensed indexed content at any useful size.** This is a scarcity
problem, not a rate-limit problem, and it is worth being precise about why.

### The aesthetic is visual, and search is textual

A disposable-camera night shot is defined by *direct flash on cheap film in
darkness* — a blown-out foreground subject a metre from the lens, a receding dark
background, colour casts and grain. None of that is in a filename or a tag.
Searching for it returns whatever the words happen to co-occur with.

Measured `result_count` from the API, all licenses, adult content excluded:

| Query | Results |
| --- | --- |
| `disposable camera night` | 17 |
| `night photo cheap camera` | 7 |
| `toy camera night` | 64 |
| `toy camera` / `lomography` / `disposable camera` | 240 (API cap) |
| `night flash photography` | 240 (API cap) |

The specific query returns **17 images**. Broader queries hit the API's
240-per-query ceiling, but they are no more on-target — `toy camera` is mostly
daylight lo-fi, and `night …` is mostly landscapes, long exposures and street
scenes that are not flash snapshots at all.

### Pixel screening is not sufficient

Retrieval was improved by screening on measured pixel statistics rather than
tags. Each image was scored for darkness (`frac_dark`), flash separation
(`flash_contrast` = bright mass over dark mass), muted colour (`sat_mean`),
detail, plus `frac_pure_black` and `frac_mid` to reject studio macro and
astronomy shots.

A calibration pass of 253 downloads kept 60. Inspecting those 60 as a contact
sheet showed the screen had captured *dark frames with bright highlights*, which
is a much broader category than the target look. The kept set contained:

- the Moon, and starfields
- macro insects on black studio backgrounds, a dead spider, dead bees, a wasp
- taxidermy specimens
- infrared trail-camera wildlife (deer, foxes, raccoons)
- fireworks, neon signage, abstract bokeh
- a camera product shot and a retail interior

Adding the stricter void-background rule removed 7 more, but the remaining 51
were still dominated by the same categories. Genuine flash snapshots of people
were roughly 5–10% of what could be retrieved.

Haar face detection was tried as a subject-presence filter and rejected: recall
was 13/60 on dark, grainy, soft film, and it produced false positives on macro
textures. It could not be used as a gate.

### Why this would hurt the model

This project is unpaired image-to-image translation: the PatchGAN discriminator
learns the *distribution* of `target/`. Fed this set, it would learn "black
background with one isolated bright object" — the astronomy and macro look —
rather than the Night Flash look. The generator would then be trained against the
wrong target. A large, confidently-wrong dataset is worse than a small honest
one, so no scraped target set is shipped.

## What will work

### Current state of `disposable_photos/`

935 images are present now, with **no `manifest.csv`**, so their provenance and
licensing are unrecorded. Measured over all 935:

| mean luminance | images | share | reads as |
|---|---:|---:|---|
| 0.00 – 0.10 | 19 | 2.0% | near-black, probably astronomy |
| **0.10 – 0.20** | **68** | **7.3%** | **genuine night flash** |
| 0.20 – 0.35 | 425 | 45.5% | dim indoor / dusk |
| 0.35 – 0.50 | 307 | 32.8% | overcast daylight |
| 0.50 – 1.00 | 116 | 12.4% | full daylight |

Mean 0.350, median 0.334. Filenames are daylight travel and outdoor stock
(`young-woman-exploring-the-grand-palace-in-bangkok.jpg`,
`scenic-desert-road-along-colorado-river.jpg`). Median long edge 579 px, 27%
under 512 px.

So this folder is **not the rejected astronomy scrape described above, but it is
not the target domain either** — it is mostly ordinary daylight photography with
a dark cast. Only 9.3% of it is in the Night Flash range. Training on it teaches
a slightly hazy, desaturated daylight look.

The reproduction snippet is in [`DATA.md`](DATA.md#check-the-target-domain-before-you-train).
Do not spend GPU hours on this folder as it stands.

### Options, in order of quality

**1. Generate the target set (recommended).** Run the app's existing
deterministic `NightFlashProcessor` over `normal_photos/`. This is unlimited,
correctly labelled by construction, carries no licensing question, and converges
far faster than scraping because the domain gap is exactly the gap you designed.
It also gives you a sanity baseline: a model trained on generated targets should
at least approach the deterministic processor, and if it does not, the bug is in
the model rather than the data.

```text
training_photos/disposable_photos/     # regenerated, plus a manifest.csv
```

**2. Real photographs.** Shoot or scan a disposable-camera roll at night.
300–500 genuine frames will beat any scraped set. Vary scene, subject, distance
and flash intensity — a roll of near-identical flash portraits teaches a very
narrow style. Then screen it before training:

```bash
python scripts/scrape_dataset.py --domain disposable --count 1000   # no-op/top-up guard
nf train \
  --source-dir training_photos/normal_photos \
  --target-dir training_photos/disposable_photos \
  --name conv-baseline --image-size 256 --batch-size 8 --epochs 30
```

**3. A free API key.** A Flickr API key enables *tag intersection*
(`toycamera` AND `night` AND `flash`), which is precisely the capability
keyword search lacks, plus deep pagination well past the 240 cap. That is the
only realistic way to approach this target domain from the web. An Openverse
token also lifts the 240/query and 200/day caps, though it does not fix the
relevance problem on its own.

## Screening thresholds are a floor, not a guarantee

`screen_disposable()` reliably removes daylight frames, blown-out frames and
macro-on-black shots. It does **not** certify the aesthetic. It was written to
support option 1 — to sanity-check that supplied real photos are dark, flash-lit
and varied — not to manufacture the target domain from search results.

Always eyeball the result before training:

```bash
python scripts/scrape_dataset.py --domain disposable --count 200   # writes .cache/sheet_disposable.jpg
open training_photos/.cache/sheet_disposable.jpg
```

## Re-running

The script is resumable. API responses are cached under `training_photos/.cache/openverse/`
and finished downloads are recorded in each domain's `manifest.csv`. Cached pages
are free — quota is spent only on pages not already on disk — so re-runs and
top-ups do not re-buy the same results. Anonymous Openverse access allows 200
requests per day, so topping up a large set takes several days unless a key is
supplied.

```bash
python scripts/scrape_dataset.py --all --max-requests 150
```
