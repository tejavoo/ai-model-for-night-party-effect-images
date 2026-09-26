# Architecture notes

Two generations of generator live side by side. `edgehat` is the current default;
`hybrid` and `restormer` are retained so older checkpoints keep loading and stay
comparable.

| arch | params | attention | notes |
|---|---:|---|---|
| `edgehat` | 4.02 M | local windows + one global block | current default |
| `swinir` | 4.90 M | windowed, relative position bias | comparison arm |
| `restormer` | 0.76 M | over channels | cheapest, C×C attention |
| `hybrid` | 0.76 M | pooled 16×16 tokens | original, keeps old checkpoints |

---

## `edgehat`

```text
input RGB  (+ optional person mask)
 │
 ├─ stem 3×3 → 48
 ├─ down1 → 96    stride 2
 ├─ down2 → 192   stride 2
 ├─ down3 → 384   stride 2
 ├─ down4 → 384   stride 2
 │
 ├─ L4 (384)  EdgeViTBlock ×1  local windowed attention
 │            HATBlock    ×1  overlapping cross-attention, global
 │
 ├─ L3 (192)  EdgeViTBlock ×2  local windowed attention
 │
 ├─ fuse: 384 + 192 → 192
 ├─ up2: 192 + 96  → 96
 ├─ up1: 96  + 48  → 48
 │
 ├─ head (freqsplit / perpixel / lut3d / tanh) → RGB residual
 ├─ effect head → 8 stochastic coefficients
 │
 └─ three optional extras, composed onto the RGB residual:
      subject gate      soft S(x,y), mask-guided when supplied
      region exposure   4×4 coefficients, bilinearly upsampled
      halation          thresholded highlight, dilated spread, warm tint
```

Level widths are `stem=48, s2=96, s3=192, s4=384, s5=384`. The deep stages fuse
once at `s3`, which is where L3 attention (192ch) and L4 attention (384ch) meet.

### Why these attention types

**EdgeViT local attention** runs over the `window × window` tokens of an 8×8
window at full channel width. Cost is `O(window⁴)` per window and **independent
of input resolution**, so 256px and 600×800 cost the same. The positional term is
a depthwise conv folded into the key (CPVT style) rather than a learned table, so
there is nothing to interpolate when the aspect ratio changes.

**HAT overlapping cross-attention** exists because local blocks cannot see the
whole frame. It pools to a coarse token grid, attends there, and projects back.
One block buys global lighting context without paying for full attention.

The alternative — attention over *channels* as in Restormer — produces a `C×C`
matrix that is also resolution independent, but it mixes features without any
spatial locality, so it cannot express "this region is brighter than that one".
For a photographic look that needs local tone decisions, spatial attention is the
better fit.

### The three extras

All three are photometric 3-channel terms applied to the RGB residual, and all
three are gated by a learnable scalar **initialised to zero**. A freshly built
model is therefore exactly the plain residual: the extras can only help once
training gives them a reason to switch on, which is what keeps early steps stable.

**Halation** is a physical operator, not a learned guess:

```text
luma      = Y · (0.2126, 0.7152, 0.0722)
highlight = relu(luma − threshold) / (1 − threshold)
spill     = dilated depthwise blur(highlight)      # wide receptive field, cheap
glow      = relu(spill + tonal(features)) · tint
```

Light scattering through the film base is a low-pass operation on over-threshold
luminance, so a dilated depthwise blur is a faithful and very cheap model. A
learned conv stack would be strictly worse here: more parameters, less physics,
and no reason to generalise to a highlight the training set never contained.

**Subject gate** produces a soft `S(x,y) ∈ [0,1]` from mid-level features. When
a Vision person mask is supplied it is adaptively max-pooled to the same grid and
added, so the gate is *guided* rather than guessed. Tone is then modulated as
`residual + mix · S · tanh(residual)`, letting the subject diverge from the
background without letting either region run away.

**Region exposure** pools to a `4×4` grid and applies a 1×1 convolution, so each
cell's coefficients are conditioned on that cell's own features. Flattening the
grid into one large linear layer would couple every cell to every other and cost
far more parameters for a worse result.

### Tiled inference interacts with region exposure

Region exposure conditions each cell on that cell's own features, which assumes
the grid spans the photograph. Under tiling the grid spans a *tile*, so the
semantics change silently. `Session.infer` prints a note when it detects
`tile > 0` together with `region_exposure`; either infer whole-frame or train
with `--no-region-exposure`.

---

## Output heads

The head decides what the network is even allowed to do.

| head | learns | params | character |
|---|---|---:|---|
| `tanh` | bounded delta (±`residual_scale`) | 21 k | safest with little data, but feels like a global filter |
| `perpixel` | unbounded delta, clamped on read | 42 k | free to repaint pixels, not just shift colour |
| `freqsplit` | low / mid / high bands separately | 63 k | routes grain and halation into the high band |
| `lut3d` | 33³ trilinear colour LUT **+ per-pixel delta** | 129 k | global grade *and* local detail in one head |

`freqsplit` uses a fixed 3-tap blur and its complement as band separators, so the
head stays a plain convolutional network while each frequency band still gets its
own pathway.

`lut3d` is implemented with **explicit trilinear gathers**, not `grid_sample`.
`F.grid_sample` was found returning near-centre values regardless of the grid it
was given, and Core ML support is uneven; eight gathered corners blended by
trilinear weights is exact, trivially testable, and lowers to plain indexing. An
identity LUT now reproduces the input to within `1.2e-7`.

### Composite heads

Heads can be combined with `+`. A `CompositeHead` sums its members and clamps
once, so a composite is still a bounded residual no matter how many parts it has.

| composite | params | when it earns its place |
|---|---:|---|
| `lut3d+freqsplit` | 192 k | **best default.** Global grade + local band detail; the parts are complementary |
| `lut3d+perpixel` | 171 k | global grade + unconstrained local delta; mostly doubles per-pixel work |
| `tanh+freqsplit` | 84 k | band-separated detail with everything bounded; most conservative |

Combining is only worth it when the members do genuinely different things.
`perpixel` and `freqsplit` are both unconstrained per-pixel paths, so pairing
them mostly just adds parameters without adding capability — which is why that
combination is not offered.

`lut3d` already contains a per-pixel conv, so "LUT + per-pixel" is already a
head on its own.

---

## Full-resolution inference: tiling

The whole-frame path letterboxes to the training size, so a 12-megapixel photo
reaches the network at about 0.06× scale. Geometry survives; detail does not, and
detail is precisely what the halation and high-frequency heads exist to supply.

Tiling fixes the *scale* problem. Each tile is cut from the **original** image,
processed at close to native resolution, and crossfaded back together:

```text
original frame
  → overlapping tile_size boxes, each edge-replicated where it runs off-frame
  → normal letterbox pipeline per tile  (so scale ≈ 1.0)
  → accumulate patch × Hann window
  → divide by accumulated window
  → one image, same size as the input
```

Two details that matter:

**Every tile is exactly `tile_size`.** Edge boxes deliberately extend past the
frame and are edge-replicated. A partial edge tile would have a blend window
approaching zero at the border, and since the result is `sum(w·pixel) / sum(w)`,
a near-zero weight makes the divide numerically fragile.

**The window has a positive floor, applied to numerator and denominator
identically.** The separable form squares the per-axis floor, so the 2D window is
clamped explicitly. Clamping only one side — as an earlier version did — scaled
the result by the mismatch factor and turned all four image corners black.

Verified seam-free: a linear gradient stays linear (mean error `0.10/255`), a
5000-step checkerboard shows no spurious step, and all four corners round-trip
exactly.

Cost scales with tile count:

| frame | tile 512 / overlap 128 | megapixels processed |
|---|---:|---:|
| 1920×1080 | 15 tiles | 3.9 MP |
| 4032×3024 | 88 tiles | 23.1 MP |

Two semantic caveats:

- **`region-exposure` becomes tile-local.** Its 4×4 grid describes the tile, not
  the photograph. Train without that capability if you need global regions.
- **Halation near a tile border** is computed from a truncated highlight field, so
  glow that should cross a boundary is weakened. The overlap hides most of it.

The subject gate is computed per tile, which is usually *better* for a large
subject.

---

No positional embeddings, no fixed spatial resolution, no resolution-dependent
parameters:

- normalisation is `GroupNorm` / `LayerNorm`, both per-channel
- the effect head uses `AdaptiveAvgPool2d(1)`
- the decoder upsamples with dynamic `F.interpolate` to each skip's size
- attention windows are re-partitioned per input size

Consequence: the model runs unmodified at 48×64, 96×128, or 600×800, and attention
cost is **constant** regardless of input size. Verified at every one of those.

What is *not* free is relative scale: a fixed-size receptive field covers a
smaller fraction of a larger frame. `geometry.py` handles that at the boundary by
letterboxing to the training shape, so the network never sees the shifted
distribution unless you explicitly ask with `--fit native`.

The procedural effects are aspect-corrected too. Vignette and light leak use
coordinates normalised by **half the shorter side**, so a circle stays circular
and a diagonal stays diagonal on any rectangle. Normalising each axis
independently to `[-1, 1]` produced a vignette 33% stronger vertically than
horizontally on a 3:4 frame; the measured anisotropy is now `0.000000`.

---

## Losses

Unpaired, so the objective is adversarial plus constraints:

| loss | weight | job |
|---|---|---|
| PatchGAN | 1.0 | match the target *distribution* |
| content + Sobel edge | 1.5 | stay close to the source |
| identity on targets | 0.75 | already-styled images pass through unchanged |
| batch RGB moments | 0.35 | coarse colour statistics, batch level, not paired |
| effect L2 | 0.02 | stop coefficients saturating at 0/1 |

The discriminator is the only loss that sees targets as targets. Everything else
is a leash.

These weights are fixed and are not exposed on `nf` / `Session`. `nightflash-train`
has flags for them if you need to experiment; see
[`TRAINING.md`](TRAINING.md#loss-weights-are-nightflash-train-only).

---

## Data pipeline

`src/nightflash/data.py` and `src/nightflash/augment.py`. The loader is unpaired:
each item pairs a source image with a *randomly chosen* target image, and the
network never sees them as a correspondence.

### One warp, three images

The most important property: a single set of geometric parameters is sampled per
item and applied to the **source, the target, and the person mask** alike.

```text
params = {scale, degrees, hflip, vflip}      # sampled once

source -> resize+scale -> crop -> rotate -> flip  \
target -> resize+scale -> crop -> rotate -> flip   >  same params
mask   -> resize+scale -> crop -> rotate -> flip  /   (nearest-neighbour, fill 0)
```

This is not a detail. With unpaired sampling the two images come from different
files, so nothing forces their crops to correspond. Applying different warps
would hand the discriminator two unrelated scenes and pull the content losses in
contradictory directions — a silent, plausible-looking degradation with no error.

### Guarantees

- every sample is exactly `image_size`², including a source smaller than the
  crop (resampled **up**, never grey-padded — padding would teach the model that
  a flat border is a normal input);
- the mask uses nearest-neighbour and a 0 fill, so it stays binary and a rotated
  corner cannot invent a person;
- the rotate fill colour is identical for source and target;
- target colour jitter is off unless explicitly enabled — the target carries the
  style signal, and perturbing it teaches a fuzzy look.

Keeping every sample at a fixed native crop resolution is also what lets the same
number be used for `--tile` at inference. Knob-by-knob rationale and the presets
are in [`TRAINING.md`](TRAINING.md#augmentation).

---

## Core ML boundary

The export wrapper returns the styled image plus optional effect coefficients.
Core Image/Metal remains responsible for grain, light leaks, bloom, vignette and
overlays. This boundary makes the model easy to debug and lets you compare
deterministic and stochastic outputs independently.
