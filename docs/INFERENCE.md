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

## Aspect ratio: no more stretching or cropping

Training uses square `N×N` crops. The earlier inference path resized every input
to a square with `ImageOps.fit`, which **centre-crops**, and then resized the
square result back to the original dimensions, which **stretches**. A 600×800
photo lost 25% of its height and came back distorted.

Both are fixed. `--fit` now controls the behaviour:

| Mode | Model input | Content | Geometry | Global context |
|---|---|---|---|---|
| `letterbox` (default) | `512×512` padded | **all kept** | exact | matches training |
| `native` | true aspect, e.g. `512×688` | **all kept** | exact | shifted scale |
| `crop` (legacy) | `512×512` | **edges lost** | stretched | matches training |

```bash
nightflash-infer --checkpoint ck.pt --input photo.jpg --output out.png
# default: letterbox
```

How it works: the image is scaled to fit inside the square, the remainder is
padded, the model runs, then the padding is cropped away **before** the final
resize. Because the crop happens first, no padding bleeds in and nothing is
stretched.

Verified on a 600×800 frame with coloured landmark dots:

| Mode | Top-left dot at y=0.15 lands at | Verdict |
|---|---|---|
| `letterbox` | y = 0.15 | correct |
| `native` | y = 0.15 | correct |
| `crop` | y = 0.03 | content destroyed |

Other options:

```bash
--fit letterbox --pad-mode reflect   # smoother border than edge replication
--fit native --image-size 512        # better context, shifted receptive field
--fit crop                           # only to reproduce old results
```

### Why `letterbox` is the default

`native` preserves more global context, but it also shifts the model's effective
receptive field. A 128px receptive field covers 25% of a 512 frame and only 16% of
an 800px-wide frame, so `native` presents the network with a distribution it was
not trained on. `letterbox` keeps the input square, matching training exactly,
and the padding is cropped away before delivery.

### Effects are rendered at output resolution

`--effects-at output` (default) un-pads and resizes to the delivered size
**first**, then applies grain/leak/vignette. This matters: rendering grain at
512px and then upscaling to 800px would make the grain 1.56× too large and soft.

```bash
--effects-at model   # cheaper; grain gets rescaled with the image
```

### Vignette and leak are now aspect-correct

The vignette used to normalise each axis independently to `[-1, 1]`, so on a
192×256 frame an equal 40px step produced **33% more falloff vertically** than
horizontally — an oval instead of a circle. Both effects now share one scale
derived from the shorter side, and the measured anisotropy is **0.000000**.

This also means the same `effectParams` vector means the same thing on a phone
photo, a 3:2 crop, and a square preview.

## Tiled inference — optional, for real detail

The whole-frame path is correct but heavily downscaled. Measured scale factor of
what the network actually sees:

| input | `--fit letterbox` | `--fit native` |
|---|---|---|
| 600×800 | 256×256, **0.32×** | 256×344, **0.43×** |
| 1920×1080 | 256×256, **0.13×** | 456×256, **0.24×** |
| 4032×3024 | 256×256, **0.06×** | 344×256, **0.085×** |

At `--image-size 256` a 12-megapixel phone photo is letterboxed to 6% scale. Tone
and colour survive; grain and halation detail do not.

Tiling fixes the scale rather than the geometry:

```bash
# whole frame (default) — nothing cropped, but heavily downscaled
nightflash-infer --checkpoint ck.pt --input big.jpg --output out.png

# tiled — model works near native resolution, crossfaded
nightflash-infer --checkpoint ck.pt --input big.jpg --output big-t512.png \
                 --tile 512 --tile-overlap 128
```

```python
s.infer("big.jpg")                     # whole frame
s.infer("big.jpg", tile=512)           # full-resolution detail
s.infer("big.jpg", tile=512, tile_overlap=256)   # smoother, more cost
```

Tiled output is written as `<stem>-t512.png` so it never overwrites the
whole-frame result.

### Choosing a tile size

**Match the tile to `--image-size`.** This is the one rule that matters. A model
trained on 256px crops has learned the feature statistics of 256px crops; feed
it a different tile size and you are outside that distribution.

| trained at | infer with | result |
|---|---|---|
| 256 crops | 256 tiles | **Correct.** Same feature statistics, smooth Hann blend. |
| 256 crops | 512 tiles | Degraded. Larger context than the model ever saw. |
| 1024 images | 256 tiles | Worst. Objects appear cropped in ways never seen in training. |

`Session.infer` warns when `tile != image_size` rather than blocking, because
running a large tile through a small-crop model is occasionally useful.

Cost on a 4032×3024 source, measured:

| tile | overlap | overlap ratio | tiles | MP processed |
|---:|---:|---:|---:|---:|
| 256 | 64 | 25% | 336 | 22.0 |
| 256 | 128 (default) | 50% | 713 | 46.7 |
| 512 | 128 | 25% | 88 | 23.1 |
| 1024 | 256 | 25% | 20 | 21.0 |

What matters is the overlap **ratio**, not its absolute size: the three 25% rows
all land near 22 MP, and going to 50% roughly doubles it to 46.7. Use a 25%
ratio for speed on a batch, 50% when the output quality matters.

Note that cost scales with the source, not with `tile`: tiling a 12MP photo
means running the network over 22-47 megapixels, which is more than five to
eleven full passes over the whole image.

The trainer prints the cost before it starts:

```text
big.jpg (1600, 1200) -> tiled: 12 tiles of 512px, 3.1 MP processed
```

### Padding outside the frame

Edge tiles must be a full `tile_size`, so something has to fill the part that
hangs off the photograph. `pad_mode` chooses what the model sees there:

| mode | behaviour | when |
|---|---|---|
| `edge` (default) | replicate the border row/column | safe; never invents a gradient, never a black bar |
| `reflect` | mirror the border | strong vertical structure, where a repeated column is visible |
| `black` | zero fill | only correct if the model was trained on padded black borders |

`reflect` silently falls back to `edge` when the pad would exceed the axis it is
mirroring, which reflection cannot do.

### Borders do not fade to black

A Hann window tapers to zero at its edges, so naively crossfading tiles fades the
frame border towards black. Two mechanisms prevent it here, so you do **not**
need to pre-pad the image with reflection yourself:

1. the window has a positive floor (1e-3), so weights never vanish;
2. edge tiles extend past the frame and are padded, so the frame border is never
   a tile border.

Verified: a 255-valued border strip survives tiled inference at 254.6/255, with
worst-case error 1.0. A fade would show near 0.

### Downscaling while tiling

`max_edge` shrinks the long edge, preserving aspect ratio, *before* tiling. Use
it to batch many images, or to match a model trained at a small `image_size`:

```python
s.infer("12mp.jpg", max_edge=1024)                  # 4032x3024 -> 1024x768
s.infer("12mp.jpg", max_edge=1024, tile=256)       # downscale AND tile
s.infer("huge/", max_edge=2048, seeds=2)
```

It never upscales: `max_edge=4096` on a 300×200 photo returns 300×200. Aspect
ratio is preserved to within 0.005 on the sizes that matter.

### What tiling changes semantically

- **`region-exposure` becomes tile-local.** Its 4×4 grid now describes a tile, not
  the photograph. Train with `--no-region-exposure` if you need global regions.
  `Session.infer` prints a note when it detects this combination.
- **Halation across tile borders** is weakened, because the highlight field is
  truncated. The overlap hides most of it.
- **Subject gate** is computed per tile, which usually helps on large subjects.

### Seam behaviour

Tiles are crossfaded with a raised-cosine (Hann) window, so seams are zero by
construction rather than blended away afterwards. Measured:

| check | result |
|---|---|
| corners round-trip | exact, all four |
| linear gradient stays linear | mean error 0.10/255 |
| checkerboard spurious step | none (step equals the original 255) |
| output size and aspect | identical to input |

Two implementation details that are easy to get wrong, and were:

1. **Every tile is exactly `tile_size`**, edge-replicated where it runs off-frame.
   Partial edge tiles have a window approaching zero at the border.
2. **The window floor must be applied to numerator and denominator identically.**
   Since the blend is `sum(w·pixel) / sum(w)`, clamping only the denominator
   scales the output — an earlier version turned all four corners black.

---

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
