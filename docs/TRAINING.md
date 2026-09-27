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

## Training modes

When the target pool is your only data — the normal case for a photographic
aesthetic — the source domain is **manufactured** by simulating a non-flash
capture of the same scenes.

| `--train-mode` | source | target | correspondence |
|---|---|---|---|
| `paired` *(default for one pool)* | `degrade(image_i)` | `image_i` | exact, same file |
| `sameset` | `degrade(image_j)`, `j ≠ i` | `image_i` | none, same directory |
| `unpaired` | `--source-dir` | `--target-dir` | none, two pools |

Omit `--train-mode` and it resolves from the arguments: one pool gives `paired`,
two distinct pools give `unpaired` with a printed notice. An explicit mode is
honoured and rejected loudly if it cannot work, so `--source-dir` with
`--train-mode paired` is an error rather than a silent fallback.

In both degradation modes the **shared geometric warp is applied first, to both
images**, and the degradation runs afterwards on the source only. Any crop, flip
or rotation is therefore common to the pair; only the simulated camera differs.

---

## Synthetic degradation

### Pipeline

Nine stages, in the order a camera pipeline would be *undone*. Resampling comes
first so the photometric stages act on clean pixels.

| # | stage | purpose |
|---|---|---|
| 1 | crop + rotate | geometric normalisation, shared with the target |
| 2 | downscale → upscale | removes grain and fine detail |
| 3 | blur | residual softening |
| 4 | contrast ↓ | **flattens the flash falloff signature** |
| 5 | black lift `x + b(1-x)` | removes crushed shadows, preserves white |
| 6 | exposure ↓ | overall gain |
| 7 | white balance ↑ | flash is neutral, ambient is tungsten |
| 8 | saturation ↑ | flash photos only *look* desaturated because of the WB mismatch |
| 9 | sensor noise | so the model doesn't assume a pristine input |

Stages 4 and 5 carry most of the domain shift. Compressing the tonal range is
what removes the falloff — the bright subject on dark ground — and makes the
output read as an evenly-lit non-flash capture rather than merely a dark one.

### Profiles

```bash
--degrade-profile subtle      # gentle; useful for the first epochs
--degrade-profile moderate    # default
--degrade-profile strong      # widest domain shift that still looks like a photo
```

| stage | subtle | moderate | strong |
|---|---|---|---|
| `resize` | 0.85–1.00 | 0.60–0.90 | 0.50–0.80 |
| `blur` | 0.0–0.6 | 0.3–1.0 | 0.4–1.4 |
| `contrast` | 0.85–0.95 | 0.70–0.90 | 0.65–0.85 |
| `black_lift` | 0.02–0.06 | 0.04–0.12 | 0.05–0.14 |
| `exposure` | 0.85–0.95 | 0.70–0.90 | 0.65–0.85 |
| `saturation` | 1.00–1.15 | 1.05–1.30 | 1.10–1.40 |
| `wb` | +2 to +8 | +4 to +14 | +6 to +20 |
| `noise` | 0.000–0.010 | 0.005–0.020 | 0.005–0.018 |

Measured on a real photograph, relative to the clean target:

| | subtle | moderate | strong |
|---|---:|---:|---:|
| high-frequency energy | 0.52 | 0.30 | 0.21 |
| tonal range `p99 - p1` | 0.76 | 0.62 | 0.58 |
| warmth `R - B` | 0.040 | 0.078 | 0.121 |

### Overriding individual bounds

```bash
nf train --target-dir POOL --degrade-contrast-min 0.3 --degrade-contrast-max 0.4
nf train --target-dir POOL --degrade-resize-min 0.4 --degrade-resize-max 0.5
```

All eight stages take `--degrade-<stage>-min` and `--degrade-<stage>-max`:
`exposure`, `black-lift`, `contrast`, `saturation`, `wb`, `resize`, `blur`,
`noise`.

For a reusable profile, or a simulated "virtual camera", write a JSON file:

```bash
nf train --target-dir POOL --degrade-config my-camera.json
```

```json
{ "resize_min": 0.35, "resize_max": 0.55, "none_prob": 0.2, "curriculum": false }
```

An explicit flag beats the file; an unspecified flag defers to it.

### Behaviour

```bash
--degrade-none-prob 0.10        # pass the target through undegraded, this often
--no-degrade-curriculum         # do not ramp across training
```

**The identity path** (`none_prob`, default 0.10) feeds the target through
undegraded. It anchors the identity case and widens the input distribution, which
is the main defence against the failure in the next section.

**The curriculum** is on by default and ramps `subtle → moderate → strong` across
training. Early epochs get an easy mapping, later epochs get robustness. It is
inert unless the epoch horizon is known, which `Session` supplies from `epochs`.

### Two limitations worth knowing

**Resampling cannot remove a wide halo.** An early draft of this documentation
claimed the downscale/upscale stage destroys halation bloom. Measurement refutes
it. Halation is wide and *low-frequency*, and a low-pass filter has nothing to
remove — blurring a glow makes it brighter by spreading the core into the ring. On
a probe, the ring-to-peak luminance ratio *rises* from 0.57 to 0.86 as blur goes
from 0 to 4.

What the pipeline achieves is attenuation, driven by the tonal compression rather
than the filtering. That is enough here — the model sees a highlight with less
glow than the target and learns the difference — but not enough if you need a
source with provably zero halation. That would require re-rendering the lighting
rather than filtering it.

**The source must stay plausible.** A first `strong` preset used blur up to 2.5
and black lift up to 0.20. The result looked foggy rather than like a photograph,
and its injected noise was glaring against the softened background. Training on
that teaches a model to expect an input distribution that never occurs at
inference, which is worse than a weaker domain shift. The ranges above are
tuned so even `strong` still reads as a believable overcast capture.

### Choosing between `paired` and `sameset`

`paired` admits a shortcut. Because the source is a deterministic function of the
target, the cheapest way to reduce the loss is to learn `degrade⁻¹` rather than
the aesthetic. The model then looks excellent on freshly-degraded images and poor
on real photographs, and the failure is invisible unless you look for it.

`sameset` removes the shortcut by construction: `image_i` and `image_j` are
unrelated photographs, so there is no pointwise relationship to invert and the
model must learn the conditional distribution. It also doubles the usable
pairings, since every image serves as both source and target.

The cost is that `sameset` has no correspondence, so no supervised reconstruction
and no real quality number. Only distribution-level metrics apply, and `best.pt`
is deliberately not written.

**The tell:** good results on freshly-degraded images but poor results on ordinary
photographs means the inverse was learned. The validation strip makes this visible
within a few epochs.

**Recommendation:** run both as parallel arms on two GPUs and compare the strips.
That is the comparison worth making, and it costs one session.

---

## Validation

```bash
--val-every 1              # render a strip every N epochs; 0 disables
--val-count 8              # images per strip
--val-select fixed         # same images every epoch (default), or random
--val-cols 3               # 3 = input|output|target, 2 drops the target
--val-split 0.15           # fraction of the pool held out
--val-holdout-dir PATH     # explicit held-out directory, overrides --val-split
--val-seed 0               # seeds the deterministic degradation and effects
--val-layout rows          # rows = one image per row; cols transposes
```

One PNG per run, in `runs/<name>/validations/`:

```
epoch-0007.png   one image per row, cells = input | output | target
latest.png       a copy, for quick viewing in a notebook
history.csv      one row per epoch: epoch,images,psnr,ssim,l1,effect
```

The header prints the epoch, the mode, the active profile and the aggregate
metrics, so the strip is readable without opening a log.

### Two determinisms, and why both are needed

**Which** images appear is set by `--val-select`. `fixed` is the default because
per-image progress is only legible if the same images recur — and because
re-showing a handful the model has memorised is exactly how you fail to notice it
memorised them. `random` answers a different question, "is the model still healthy
on *arbitrary* images", and is the right choice when you suspect overfitting.

**How** each image is degraded and seeded is pinned unconditionally. Seeds derive
from the filename via SHA-256, not from the epoch-dependent random stream, so a
strip rendered in this session matches one rendered last week. This is not an
optimisation: a validation image degraded with a fresh draw each epoch is a
different problem each time, and any apparent improvement is noise.

Python's built-in `hash()` is salted per interpreter, so it cannot be used here.

### Splitting

Validation images are drawn from the target pool by a deterministic seeded split,
by scene, and are never trained on. A pool too small to split leaves the
validation half empty rather than stealing training data.

### Checkpoint selection

```bash
--no-keep-best     # do not write checkpoints/best.pt
--keep-last 3      # retain only the newest N per-epoch checkpoints
```

`best.pt` is written whenever validation PSNR improves. This is the difference
between being able to recover a run that peaked at epoch 40 and decaying by 60,
and not being able to.

It is written in **`paired` mode only**. In `sameset`, PSNR compares two
unrelated photographs, so ranking on it would be meaningless, and emitting a
checkpoint that looks authoritative but is not would be worse than emitting none.

`--keep-last` exists because 60 epochs of a 17 MB checkpoint is a gigabyte, which
fills Kaggle's disk and makes every snapshot slow.

### Weight averaging

```bash
--ema-decay 0.999
```

Maintains a shadow copy of the generator weights, checkpointed and resumed with
the run. Averaging over the trajectory is standard for adversarial training and
reduces frame-to-frame flicker, which matters for a live camera preview. Off by
default.

---

## Training controls

### Augmentation

The loader samples one geometric warp per item and applies it **identically to
the source, the target and the mask**. That last part is what keeps an unpaired
dataset coherent: the two images come from different files, but both are
cropped, scaled, rotated and flipped the same way, so the pair still describes a
consistent crop position. If they diverged, the discriminator would see two
different scenes and the content losses would pull in contradictory directions.

Pick a preset:

```bash
--augment default        # scale 0.8-1.25, rot +/-12deg, hflip + vflip
--augment conservative   # scale 0.9-1.15, rot +/-6deg,  hflip + vflip
--augment strong         # wider scale, rot +/-15deg, + colour jitter + 2 holes
--augment none           # deterministic centre crop, no jitter
```

Or in Python, for full control of the bounds:

```python
from nightflash.augment import AugmentConfig

s = Session(..., augment=AugmentConfig(
    scale_min=0.85, scale_max=1.2,     # how much of the scene fits the crop
    rotate_degrees=8.0,               # cameras are rarely level
    hflip=0.5, vflip=0.5,             # doubles the effective dataset
    brightness=0.0, contrast=0.0,     # OFF by default; see below
    saturation=0.0, temperature=0.0,
    apply_colour_to_target=False,
    holes=0, hole_size=0.12, hole_count_prob=0.0,
))
```

The full rationale for each knob is in the `augment.py` module docstring. The
short version:

| knob | default | why |
|---|---|---|
| `scale_min/max` | 0.8 / 1.25 | **The one that matters most.** A 256px crop of a 12MP night photo is a close-up; the same crop of a 3MP wide shot is a whole room. Without jitter the model silently learns only the first case. |
| `rotate_degrees` | 12 | Stops the model learning that a horizon is always horizontal. Kept small: past ~15° you are teaching it to accept tilted *content*. |
| `hflip`, `vflip` | 0.5 each | Both are safe because neither domain has an inherent up direction. Diagonal flip is deliberately excluded — it would mirror real text and faces. |
| colour jitter | **off** | See the trap below. |
| `holes` (cutout) | **off** | Standard for classification because it forces context use. Less useful for *appearance* transfer, and it can teach the model to hallucinate texture over occluders — a bad trait for a camera filter. |

#### The colour-jitter trap

**Colour jitter is off by default, and that is deliberate.** Your target domain
*is* a colour look. Perturbing target colour teaches the discriminator that the
style is fuzzy, and it can wash out exactly the grade you are trying to learn.
It is available behind `apply_colour_to_target=True` for when you have already
memorised the target set, and it is included in the `strong` preset so you can
measure what it does.

Source-only colour jitter is safe and mildly useful, because the source domain
genuinely varies in white balance.

#### Consistency guarantees

These are enforced and covered by tests, because violating any of them silently
degrades training rather than raising:

- every sample is exactly `image_size` × `image_size`, including a source
  smaller than the crop (it is resampled up, never grey-padded — padding would
  teach the model that a flat border is a normal input);
- the mask is resampled with nearest-neighbour and filled with 0 at rotated
  corners, so it stays binary and never invents a person;
- the rotate fill colour is identical for source and target;
- target colour jitter is off unless explicitly enabled.

Keeping the crop at a fixed native resolution is also what lets you use the same
number for `--tile` at inference.

### Loss weights are `nightflash-train` only

The subsections below tune the loss weights, and those flags live on the legacy
standalone trainer:

```bash
nightflash-train --source-dir ... --target-dir ... --adversarial-weight 1.5
```

`nf train` and `Session` do **not** expose them, because the recommended path is
the `Session` object and its fixed, tested loss balance. If you need to change a
weight, either use `nightflash-train` or pass the config through
`ModelConfig` — but be aware you are leaving the tested defaults, so re-verify
the previews afterwards.

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

## The objective, and where the texture penalty belongs

The generator emits a deterministic residual which a seeded procedural stage then
turns into a stochastic image. Writing `x_det` for the pre-effect output and
`x_styled` for the post-effect one, the generator loss is:

```text
L = L_adv + 1.5 * L_content + 0.35 * L_moment + 0.75 * L_identity + 0.02 * L_effect
    + paired_weight * L_paired        (paired and sameset only)
```

`L_content` is a convex combination of an L1 half and a Sobel edge half, and the
two halves are evaluated against **different images**:

```text
L_content = 0.65 * L1(x_det,    source)      # texture statistic -> pre-effect
          + 0.35 * edge(x_styled, source)   # structure statistic -> final image
```

### Why this matters

Evaluating both halves on `x_styled` — as an earlier version did — charges the
output for the stochastic texture, because L1 is texture sensitive. Measured on a
trained model:

| quantity | value |
|---|---:|
| `content(x_det, source)` | 0.02271 |
| `content(x_styled, source)` | 0.04565 |
| **penalty the effects incur** | **+0.02294** |
| `‖x_det − source‖₁` | 0.01233 |

The robust claim is the **ratio**, not the absolute values, which are
run-specific: the effects are charged **1.86×** more than the entire learned
residual is charged for performing the whole style transfer. The adversarial
term at weight 1.0 asks for grain; the content term at weight 1.5 penalises it by
nearly double the residual's own contribution. The content term wins, and the
model learns to suppress the grain and halation the effect head exists to
produce — the classic failure where an L1 penalty on a stochastic target
averages the stochastic component out.

The fix keeps each half where it belongs. L1 is a texture statistic and must be
evaluated where the texture is not stochastic. The Sobel response is a structure
statistic, is indifferent to fine texture, and *should* see the final image so
that broken geometry is actually caught.

`L_paired` is evaluated on `x_det` for the same reason: an L1 term pointed at a
stochastic **target** would average the target's own grain out of both sides.

### Consequence for comparability

Grain and halation become visibly stronger after this change, which is the
intent — but checkpoints trained under the old balance are not comparable to new
ones. `LOSS_VERSION` was incremented to 2, is written into `config.json` and every
checkpoint, and resuming a checkpoint with a different version prints a warning
rather than silently mixing two objectives.

```bash
# start a new run directory for a clean comparison
nf train --target-dir POOL --name after-loss-fix --epochs 60
```

## Recommended training schedule

With the degradation modes, the schedule is driven by the validation strip rather
than by a fixed epoch count:

1. **20–30 epochs**, `paired`, `--degrade-profile moderate`. Read the strip. You
   are looking for grain and halation to appear at all.
2. **30–60 epochs.** The curriculum ramps to `strong` on its own. Watch
   `history.csv`; if PSNR peaks and then falls, `best.pt` already holds the good
   weights.
3. Repeat steps 1–2 with `--train-mode sameset` on the second GPU and compare the
   strips.
4. Raise `--image-size 512 --batch-size 4` only once 256px looks right.
5. Export only after held-out evaluation.

Do not assume more epochs improve quality. Unpaired GANs can overfit the target
domain and damage content preservation, which is precisely why `best.pt` exists
rather than only `latest.pt`.

Two pre-flight checks are worth more than any hyperparameter:

- **Measure the target domain** before training. A PatchGAN learns whatever is in
  that folder, and a wrong folder fails silently. See
  [`DATA.md`](DATA.md#check-the-target-domain-before-you-train).
- **Read the validation strip**, not the loss curve. The discriminator loss is not
  a quality signal, and it will fall while the output gets worse.

## Progress output in notebooks

`tqdm` draws its bar with carriage returns on **stderr**. That works in a
terminal but not in a notebook, where it produces two problems:

1. The bar is often invisible, because the kernel frontend does not redraw
   stderr `\r` output.
2. When the trainer is driven through `subprocess.Popen(..., text=True)`, Python's
   *universal newlines* translation rewrites every `\r` into `\n`. Each redraw of
   the bar becomes its own printed line, flooding the cell with hundreds of
   near-identical lines.

`train.py` handles this at the source. `--progress auto` (the default) picks:

| Environment | Behaviour |
|---|---|
| Real TTY, not a notebook | live `tqdm` bar |
| Notebook, or piped output | one clean newline-terminated line per update |

Force it either way:

```bash
--progress plain     # always one line per update (notebook-safe)
--progress tqdm      # always a live bar
--progress none      # silent, for scripted runs
```

Plain output looks like this:

```text
epoch 3/50 started (125 steps, log every 25)
  [ 20.0%] epoch 3/50 step 25/125 eta=1m12s  loss_d=0.5310 loss_g=1.3765 adv=0.9213
  [ 40.0%] epoch 3/50 step 50/125 eta=48s    loss_d=0.5050 loss_g=1.2898 adv=0.8762
epoch 3/50 complete in 3m01s  loss_d=0.5182 loss_g=1.3310
```

`--log-every` controls the update cadence for both styles.

### Driving the trainer from a notebook

Use the helper rather than raw `Popen`. It reads the pipe in **binary** mode, so
carriage returns are never rewritten, and it separates real lines from in-place
redraws:

```python
from nightflash.notebook import train, show_previews, last_checkpoint

train([
    "--source-dir", f"{ROOT}/training_photos/normal_photos",
    "--target-dir", f"{ROOT}/training_photos/disposable_photos",
    "--output-dir", f"{ROOT}/runs/hybrid-warm",
    "--image-size", "256", "--batch-size", "16", "--epochs", "50",
    "--num-workers", "2", "--preview-every", "25", "--log-every", "25",
    "--device", "cuda", "--amp",
    "--transformer-blocks", "1",
], root="/content", log=f"{ROOT}/runs/hybrid-warm/train.log")

show_previews(f"{ROOT}/runs/hybrid-warm", n=6)
```

If you must use `Popen` yourself, the two rules are:

```python
# 1. binary mode, so universal newlines cannot split redraws
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

# 2. split redraws from lines yourself
from nightflash.streaming import iter_output
for kind, text in iter_output(proc.stdout):
    if kind == "line":
        print(text)                   # safe to print
    else:
        status_widget.update(text)    # overwrite in place
```

A `text=True` pipe plus a plain `for line in process.stdout` is the broken
combination. `tests/test_progress_streaming.py` asserts the difference.

---

## Resuming and warm starting

`--epochs` is an **absolute target**, not a count of extra epochs. Resuming with
`--epochs` at or below the checkpoint's epoch trains nothing; the trainer now
exits with an explicit error instead of reporting false success. To move from a
trained conv model to a hybrid one, use `scripts/warm_start.py`.

Both are covered in [`WARM_START.md`](WARM_START.md).

---

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
