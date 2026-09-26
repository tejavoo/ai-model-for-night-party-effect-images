"""Held-out validation: one strip image per run, plus a metric history.

Why a strip and not a number
----------------------------
A loss curve tells you the objective is going down. It does not tell you whether
the output still looks like a photograph. The strip is the thing you actually
look at, and it is cheap enough to render every epoch.

Determinism
-----------
Two independent things are pinned, because without both, epoch 7 and epoch 8 are
not comparable:

* **which** images appear, when ``val_select="fixed"`` (the default);
* **how** each one is degraded and how the stochastic effects are seeded, always.

The second is unconditional. A validation image degraded with a fresh random
draw each epoch would be a different problem every time, so any apparent
improvement would be noise. Seeds are derived from the filename with
:func:`nightflash.degrade.stable_seed`, which is stable across processes, so a
strip rendered in a Kaggle session matches one rendered last week.

Output
------
``runs/<name>/validations/epoch-0007.png``   the strip
``runs/<name>/validations/latest.png``       a copy, for quick viewing
``runs/<name>/validations/history.csv``      one row per epoch
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from .degrade import stable_seed
from .losses import global_ssim, psnr
from .utils import ensure_dir, tensor_to_pil

LABEL_HEIGHT = 20
HEADER_HEIGHT = 26
CELL_PADDING = 4


@dataclass
class ValidationMetrics:
    """Aggregate scores for one validation pass."""

    images: int = 0
    psnr: float = 0.0
    ssim: float = 0.0
    l1: float = 0.0
    effect: float = 0.0

    def as_row(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "images": self.images,
            "psnr": round(self.psnr, 4),
            "ssim": round(self.ssim, 4),
            "l1": round(self.l1, 5),
            "effect": round(self.effect, 5),
        }


def _metric_key(metrics: ValidationMetrics) -> float:
    """Higher is better. PSNR is the ranking signal for ``best.pt``.

    Only meaningful in ``paired`` mode, where prediction and target are the same
    scene. In ``sameset`` there is no correspondence and PSNR is not a quality
    measure, so ``best.pt`` is not written.
    """

    return metrics.psnr


def render_strip(
    rows: Sequence[dict[str, Any]],
    *,
    columns: int = 3,
    layout: str = "rows",
    title: str = "",
    metrics: ValidationMetrics | None = None,
) -> Image.Image:
    """Compose one strip image from per-image panels.

    ``rows`` is a sequence of dicts with keys ``index``, ``filename``,
    ``source``, ``output``, ``target``, each a CHW float tensor in [0, 1].
    ``columns`` is 2 (source | output) or 3 (source | output | target).

    ``layout="rows"`` gives one image per row, so a given image's source and
    output are adjacent and the comparison needs no eye travel. ``layout="cols"``
    transposes it, one image per column, for wide strips.
    """

    if not rows:
        return Image.new("RGB", (64, 32), "white")

    keys = ("source", "output", "target")[:columns]
    count = len(rows)
    first = tensor_to_pil(rows[0][keys[0]])
    cell_w, cell_h = first.size
    pad = CELL_PADDING

    if layout == "rows":
        # one image per row: the cells within a row are source|output|target
        grid_cols = columns
        per_row_label = True
    else:
        # one image per column: the cells down a column are source|output|target
        grid_cols = count
        per_row_label = False

    body_w = grid_cols * cell_w + (grid_cols + 1) * pad
    row_label_h = LABEL_HEIGHT if per_row_label else 0
    body_h = columns * cell_h + (columns + 1) * pad + count * row_label_h

    head_text = title
    if metrics is not None and metrics.images:
        head_text += (
            f"   n={metrics.images}  PSNR {metrics.psnr:.2f}dB"
            f"  SSIM {metrics.ssim:.4f}  L1 {metrics.l1:.4f}"
            f"  effect {metrics.effect:.3f}"
        )
    head_text += f"   [cell order: {' | '.join(keys)}]"
    header = HEADER_HEIGHT if head_text else 0

    # Widen the canvas if the header is longer than the image grid, so the
    # metrics line is never clipped on a narrow strip.
    if header:
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        needed = int(measure.textlength(head_text)) + 16
        body_w = max(body_w, needed)

    canvas = Image.new("RGB", (body_w, header + body_h), "white")
    draw = ImageDraw.Draw(canvas)
    if header:
        draw.text((6, 7), head_text, fill="black")

    if per_row_label:
        y = header
        for i, row in enumerate(rows):
            draw.text((pad, y + 5), f"#{row['index']}  {row['filename']}", fill="black")
            y += LABEL_HEIGHT
            for j, key in enumerate(keys):
                canvas.paste(
                    tensor_to_pil(row[key]),
                    (pad + j * (cell_w + pad), y + pad),
                )
            y += cell_h + pad
    else:
        for j, key in enumerate(keys):
            draw.text((pad, header + 5), key, fill="black")
        x = pad
        for i, row in enumerate(rows):
            draw.text((x + 2, header + 5), f"#{row['index']} {row['filename']}", fill="black")
            y = header + LABEL_HEIGHT
            for key in keys:
                canvas.paste(tensor_to_pil(row[key]), (x, y + pad))
                y += cell_h + pad
            x += cell_w + pad

    return canvas


def column_key_legend(columns: int) -> str:
    names = ("source", "output", "target")
    return "  |  ".join(names[:columns])


def evaluate_batch(
    batch: dict[str, Any], styled: torch.Tensor, deterministic: torch.Tensor
) -> ValidationMetrics:
    """Score a validation batch against its ground-truth targets."""

    target = batch["target"]
    n = target.shape[0]
    psnrs, ssims, l1s = [], [], []
    for i in range(n):
        pred = deterministic[i : i + 1].clamp(0.0, 1.0)
        truth = target[i : i + 1]
        psnrs.append(psnr(pred, truth))
        ssims.append(float(global_ssim(pred, truth)))
        l1s.append(float((pred - truth).abs().mean()))
    effect = 0.0
    if "degrade_params" in batch and batch["degrade_params"] is not None:
        effect = float(batch["degrade_params"].abs().mean())
    return ValidationMetrics(
        images=n,
        psnr=float(np.mean(psnrs)) if psnrs else 0.0,
        ssim=float(np.mean(ssims)) if ssims else 0.0,
        l1=float(np.mean(l1s)) if l1s else 0.0,
        effect=effect,
    )


def combine_metrics(parts: list[ValidationMetrics]) -> ValidationMetrics:
    """Image-weighted average across batches.

    Weighted by image count, not by taking the mean of batch means, which would
    over-weight a short final batch.
    """

    parts = [p for p in parts if p is not None and p.images > 0]
    if not parts:
        return ValidationMetrics()
    weights = np.array([p.images for p in parts], dtype=np.float64)
    total = float(weights.sum())
    weights = weights / total
    return ValidationMetrics(
        images=int(total),
        psnr=float(np.average([p.psnr for p in parts], weights=weights)),
        ssim=float(np.average([p.ssim for p in parts], weights=weights)),
        l1=float(np.average([p.l1 for p in parts], weights=weights)),
        effect=float(np.average([p.effect for p in parts], weights=weights)),
    )


HISTORY_COLUMNS = ("epoch", "images", "psnr", "ssim", "l1", "effect")


def append_history(path: Path, row: dict[str, Any]) -> None:
    """Append one row, creating the header on first write."""

    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_COLUMNS))
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HISTORY_COLUMNS})


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def best_epoch(path: Path) -> tuple[int, float] | None:
    """Highest-PSNR epoch recorded so far, or ``None``."""

    rows = read_history(path)
    scored = [r for r in rows if r.get("psnr") not in (None, "", "0", "0.0")]
    if not scored:
        return None
    winner = max(scored, key=lambda r: float(r["psnr"]))
    return int(winner["epoch"]), float(winner["psnr"])
