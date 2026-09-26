# Kaggle

Same six cells as Colab. Only the data plumbing differs, because Kaggle wipes
`/kaggle/working` between sessions and you attach data as a Dataset rather than
mounting a drive.

Notebook: [`notebook/nightflash_kaggle.ipynb`](../notebook/nightflash_kaggle.ipynb)

---

## Why Kaggle differs from Colab

| | Colab | Kaggle |
|---|---|---|
| persistent storage | Google Drive | **none** — `/kaggle/working` is wiped |
| resume | write to Drive, works | **Save Version as Dataset**, re-attach next session |
| GPU | T4 / L4 / A100 | T4×2 or P100 |
| hours | ~12 h per session | 30 h/week |
| data | Drive mount | attached Dataset under `/kaggle/input` |

The wiped working directory is the only thing that changes your workflow.
Everything else, including the briefing and the no-op guard, behaves identically.

---

## Setup

**1. Upload the dataset.** Create a Kaggle Dataset containing:

```text
training_photos/
  normal_photos/
  disposable_photos/
  validation_photos/
```

**2. Open a notebook** and set **Accelerator → GPU T4 x2**.

**3. Put the repo in place.** Either clone it in an early cell, or upload the
repo as a Dataset and unzip it into `/kaggle/working/nightflash-edge`.

```python
!git clone --depth 1 https://github.com/you/nightflash-edge /kaggle/working/nightflash-edge
```

**4. Attach the dataset** in the right-hand panel. The notebook finds it
automatically and tells you what it saw if it cannot.

---

## The T4×2 tip

A 4M-parameter model at 256px will not saturate a T4. On a **T4×2** kernel the
second GPU is better spent running a second experiment than sitting idle.

Duplicate the notebook, give the copy a different `name`, and run them at the
same time:

```python
# notebook A
s = Session(arch="edgehat", head="freqsplit", name="a", ...)

# notebook B
s = Session(arch="swinir",  head="perpixel",   name="b", ...)
```

Different `name` means different run directories, so they cannot collide. That
turns "which head is better" into a same-wall-clock experiment.

---

## Resuming across sessions

`/kaggle/working` does not survive. Snapshot the run out, attach it next time,
restore it in.

**End of a session:**

```python
from nightflash.kaggle import snapshot
snapshot(s.run_dir)
```

This publishes the run directory as a new version of a Kaggle Dataset. First call
creates the dataset; later calls add versions. Requires `~/.kaggle/kaggle.json`
or `KAGGLE_USERNAME` / `KAGGLE_KEY`.

**Start of the next session**, after attaching that dataset in the panel:

```python
from nightflash.kaggle import restore
restore(s.run_dir)          # archives anything already there, then copies back
s.train(extra_epochs=40)    # resumes where the last session stopped
```

`restore` archives the existing directory first, so you cannot accidentally
overwrite a resumed run with a stale copy.

---

## Environment report

Cell 1 prints everything worth knowing:

```python
from nightflash.kaggle import environment_report
environment_report()
```

```python
{
  'kaggle': True,
  'torch': '2.x.x',
  'cuda': True,
  'gpus': ['Tesla T4', 'Tesla T4'],
  'recommended_device': 'cuda',
  'working_dir': '/kaggle/working',
  'datasets': ['nightflash-photos'],
}
```

`recommended_device("auto")` is what the configuration cell uses, so the notebook
picks `cuda` on Kaggle and `mps` on a Mac without editing anything.

---

## Typical session

```python
# 3. configure
s = Session(arch="edgehat", head="freqsplit", name="night",
            source_dir=SOURCE_DIR, target_dir=TARGET_DIR,
            runs_root=Path("/kaggle/working/runs"),
            image_size=256, batch_size=8, epochs=50,
            device=recommended_device("auto"))

# 4. read the briefing
print(brief(s, mode="plan"))

# 5. train
s.train(extra_epochs=20)          # session budget is ~12 h, not 50 epochs

# 6. inspect
show_previews(s.run_dir, n=4)
snapshot(s.run_dir)                # so the next session can continue
```

Note `epochs=50` with `extra_epochs=20` in the same breath: `epochs` is the
absolute target, `extra_epochs` is what this session should attempt. The
briefing prints both, so the mismatch is visible before you spend GPU hours.

---

## Choosing a head on Kaggle

A T4×2 kernel runs two experiments in parallel, so use that to compare heads
rather than running them one after another. Duplicate the notebook, change
`name` and `head`:

```python
# notebook A
s = Session(arch="edgehat", head="freqsplit",      name="a", ...)

# notebook B
s = Session(arch="edgehat", head="lut3d+freqsplit", name="b", ...)
```

Different `name` means different run directories, so they cannot collide. Both
write previews to their own folder.

Head parameter counts (generator total in brackets):

| head | head params | generator total |
|---|---:|---:|
| `freqsplit` | 63 k | 4.02 M |
| `lut3d` | 129 k | 4.08 M |
| `lut3d+freqsplit` | 192 k | 4.15 M |

---

## Tiled inference on Kaggle

Kaggle images are often large, and a 256px model would see a 12MP frame at 0.06×
scale. Tiling fixes that, at a compute cost:

```python
# whole frame
s.infer(photo_path)

# tiled, near-native resolution
s.infer(photo_path, tile=512, tile_overlap=128)
```

| frame | tile 512 / overlap 128 | tiles | MP processed |
|---|---:|---:|---:|
| 1920×1080 | 512 | 15 | 3.9 |
| 4032×3024 | 512 | 88 | 23.1 |

On a T4 this is fine for a handful of photos. Do not run tiled inference over a
whole dataset expecting it to be cheap — a 500-image dataset at 88 tiles each is
44,000 forward passes. Use the whole-frame path for bulk work and tiling for
final images.

Caveat: tiled inference makes `region-exposure` **tile-local**. If you need
global regions, train with `--no-region-exposure`.

---

## No TPU

TPU needs `tf.distribute` semantics that do not exist in this codebase, and a 4M
model is a poor fit for a TPU regardless. Use the GPUs.

If you later want TPU support it is a real piece of work — the training loop,
the loss reductions, and the data pipeline all need to become distribution-aware.
It is not a flag.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: No attached dataset contains 'normal_photos'` | Dataset not attached. Use the right-hand panel, then re-run cell 2. |
| Everything gone after reconnecting | Expected — `/kaggle/working` is wiped. `snapshot()` before ending. |
| `SnapshotInfo` fails on auth | Needs `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`. |
| `CUDA out of memory` | Lower `--batch-size`, or set `num_workers=0`. |
| Only one GPU visible | Accelerator is not set to T4×2, or the kernel is CPU-only. |
| Training restarts from epoch 0 | `restore()` was not called, or the session had no snapshot. Check `s.state().status`. |
