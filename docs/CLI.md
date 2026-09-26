# CLI and Session

Two equivalent surfaces. The `Session` object for notebooks, the `nf` command for
terminals. Both drive the same code.

## The idea

A run owns its own directory, and every run remembers where it got to:

```text
runs/edgehat_freqsplit_night/
  config.json
  checkpoints/  latest.pt  epoch-0001.pt  epoch-0002.pt
  previews/     step-0000250.png  epoch-0001.png
  inferences/   photo-seed0.png
  evaluations/  report.json
  logs/         train.log
  warm_start_report.json
```

Because the folder is named after `arch` and `head`, switching architectures
never overwrites another run, and old checkpoints stay where they are.

---

## The briefing

Run this before anything else. It is generated from the live config and the live
filesystem, so it cannot drift from what the code will actually do.

```bash
nf brief
```

```python
from nightflash import Session
from nightflash.briefing import brief

s = Session(arch="edgehat", head="freqsplit", name="night", ...)
print(brief(s, mode="plan"))
```

Seven sections:

1. **What this model is** — the architecture and head you selected, plus the
   full menu of what else exists
2. **Capabilities** — halation, subject gate, region exposure, each on or off
3. **What will be trained** — parameter count, resolution, batch, device, optimiser
4. **Where everything goes** — every path, absolute
5. **What will happen** — the step-by-step flow for *this* mode and *this* state
6. **Current state** — resume epoch, global step, checkpoint size, previews
7. **Decision** — numbered options, each with the exact call to run

Section 7 is the point. It never offers a "continue" that would train zero
epochs:

```text
[!] The default target (4) is at or before the resume point (4).
    Continuing with the default would train zero epochs, so
    s.train() will refuse. Pick a positive extra-epoch count:

  [1] CONTINUE  resume from epoch 4, add 50 epochs
      s.train(extra_epochs=50)
```

---

## Modes

| mode | what it does | writes |
|---|---|---|
| `plan` | print what *would* happen | nothing |
| `train` | resume if a checkpoint exists, else start fresh | checkpoints, previews |
| `infer` | load and run, skipping training entirely | inferences only |

```python
s.plan("plan")      # no side effects at all
s.train()           # resume, or start
s.train(50)         # resume and add 50 epochs
s.infer("p.jpg", seeds=3)
```

`train()` **refuses** when the target epoch is at or before the resume point,
rather than silently training nothing:

```text
NO-OP GUARD - already at epoch 4; target 4 would train zero epochs.
             Ask for more epochs.
```

The CLI returns exit code `2` for that case, so it is scriptable.

---

## `--epochs` is absolute

This trips up everyone once. The trainer computes:

```python
start_epoch = checkpoint["epoch"] + 1
for epoch in range(start_epoch, args.epochs):
```

So `--epochs 40` on a checkpoint at epoch 40 runs `range(41, 40)` — an empty
loop. The `Session` API exists partly to make that impossible:

```python
s.target_epoch(extra=50)   # last_epoch + 1 + 50
```

`nf train --extra-epochs 50` does the same arithmetic for you.

---

## Reset and archive

```python
s.reset()                # moves the run to <name>_archive_1, clears in place
s.reset(archive=False)   # deletes outright
```

Reset **archives by default**. A 40-epoch baseline is not something to lose to a
stray call, and archives accumulate as `_archive_1`, `_archive_2`, so you can
delete them deliberately later.

---

## CLI reference

```bash
nf brief      [--extra-epochs N]     # the full briefing (default command)
nf state                                # machine-readable state JSON
nf config                               # resolved config JSON
nf train      [--extra-epochs N] [--yes]
nf infer      --input PATH [--seeds N] [--comparison]
              [--tile N] [--tile-overlap N] [--pad-mode edge|reflect|black]
              [--max-edge N]
nf reset      [--delete]
nf snapshot   [--note TEXT] [--strict] # publish a Kaggle Dataset version
nf resume     [--required]             # restore an attached snapshot
```

`nf state`, `nf config`, `nf reset`, `nf snapshot` and `nf resume` take no data
paths — they read the run directory only.

`--tile 0` (default) processes the whole frame. `--tile 512` processes
overlapping 512px tiles and crossfades them, so the model works near native
resolution. Outputs are written as `<stem>-t512.png` so both are kept.

`--max-edge N` downscales the long edge to at most N pixels, **preserving aspect
ratio**, before anything else. It never upscales. Combined with `--tile` it
gives you downscale-then-tile, which is usually what you want for batching.

`--pad-mode` chooses what an edge tile sees outside the frame: `edge` (default,
replicate), `reflect` (mirror), `black` (zero fill).

Every command takes the same run flags, so they all address the same run:

```bash
--arch edgehat|swinir|restormer|hybrid
--head tanh|perpixel|freqsplit|lut3d
--head lut3d+freqsplit|lut3d+perpixel|tanh+freqsplit
--name night
--runs-root runs
--image-size 256
--batch-size 8
--epochs 50
--device auto
--no-amp
--num-workers 2
--preview-every 250
--log-every 25
--effects-strength 0.5
--halation / --no-halation
--subject-gate / --no-subject-gate
--region-exposure / --no-region-exposure
--bottleneck-channels N
--attention-heads N
--region-grid N
--augment default|conservative|strong|none
--snapshot-every N
--snapshot-dataset SLUG
```

Training additionally takes `--source-dir`, `--target-dir`, `--val-dir`,
`--mask-dir`.

`--snapshot-every N` publishes a Kaggle Dataset version every N epochs. It is
off by default (`0`) and never fails training — a snapshot error prints a
warning and the run continues, because losing an hour of training to a network
error would be a bad trade.

Examples:

```bash
# inspect before doing anything
nf brief --arch edgehat --head freqsplit --name night

# train 50 epochs
nf train --source-dir data/source --target-dir data/target \
         --arch edgehat --head freqsplit --name night --epochs 50

# add 30 more
nf train --source-dir data/source --target-dir data/target --extra-epochs 30 --yes

# compare arms
nf brief --arch swinir --head perpixel --name swin-compare
nf brief --arch edgehat --head lut3d --name lut-compare
nf brief --arch edgehat --head lut3d+freqsplit --name composite-compare
```

---

## Tiled inference

```bash
# whole frame (default)
nf infer --input big.jpg --output out.png

# tiled, near-native resolution
nf infer --input big.jpg --output big-t512.png --tile 512 --tile-overlap 128
```

```python
s.infer("big.jpg")            # whole frame
s.infer("big.jpg", tile=512)  # tiled
```

Match `--tile` to the model's `image_size`. Cost scales with the tile count: a
4032×3024 frame needs 88 tiles at `--tile 512` and 336 at `--tile 256`. Tiled
results are named `<stem>-t<tile>.png` so they never overwrite the whole-frame
output.

One semantic caveat: tiled inference makes `region-exposure` **tile-local**.

---

## Progress in notebooks

`tqdm` draws with `\r` on stderr, which a notebook does not redraw, and Python's
universal-newlines translation turns every redraw into a separate printed line
when read through `Popen(text=True)`. `Session` avoids both:

- the trainer detects a kernel and emits newline-terminated lines
- `nightflash.notebook.train` reads the pipe in **binary** and separates real
   lines from in-place redraws

`tests/test_progress_streaming.py` asserts the difference: 20 redraws become 20
printed lines under `text=True`, and 19 redraws + 1 line under binary.

---

## Programmatic use

```python
from nightflash import Session

s = Session(
    arch="edgehat", head="freqsplit", name="night",
    source_dir="data/source", target_dir="data/target",
    runs_root="runs", image_size=256, batch_size=8, epochs=50,
)

s.state()          # RunState dataclass
s.plan()           # dict: what would happen
brief(s)           # the full printed report
s.train()          # resume or start
s.infer("x.jpg")   # inference only
s.reset()          # archive and clear
```

`Session` never prompts. If you want the decision menu, print `brief(s)`; if you
are scripting, read `s.plan()` and branch on `would_train`.
