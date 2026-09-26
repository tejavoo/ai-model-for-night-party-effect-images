from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_ADVERSARIAL_LOSS = nn.BCEWithLogitsLoss()


def adversarial_loss(prediction: Tensor, target_is_real: bool) -> Tensor:
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return _ADVERSARIAL_LOSS(prediction, target)


def _sobel_filters(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    return kernel_x, kernel_y


def edge_loss(prediction: Tensor, target: Tensor, *, reduction: str = "mean") -> Tensor:
    """Sobel gradient L1. With ``reduction="none"`` returns one value per image."""

    kernel_x, kernel_y = _sobel_filters(prediction.device, prediction.dtype)
    channels = prediction.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    pred_x = F.conv2d(prediction, kernel_x, padding=1, groups=channels)
    pred_y = F.conv2d(prediction, kernel_y, padding=1, groups=channels)
    target_x = F.conv2d(target, kernel_x, padding=1, groups=channels)
    target_y = F.conv2d(target, kernel_y, padding=1, groups=channels)
    if reduction == "none":
        # per-image, then per-pixel inside the image
        return ((pred_x - target_x).abs() + (pred_y - target_y).abs()).mean(
            dim=(1, 2, 3)
        )
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def masked_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    *,
    reduction: str = "mean",
) -> Tensor:
    difference = (prediction - target).abs().mean(dim=1, keepdim=True)
    if reduction == "none":
        if mask is None:
            return difference.mean(dim=(1, 2, 3))
        mask = mask.clamp(0.0, 1.0)
        return (difference * mask).sum(dim=(1, 2, 3)) / (
            mask.sum(dim=(1, 2, 3)) + 1e-6
        )
    if mask is None:
        return difference.mean()
    mask = mask.clamp(0.0, 1.0)
    return (difference * mask).sum() / (mask.sum() * difference.shape[1] + 1e-6)


def content_loss(
    prediction: Tensor,
    source: Tensor,
    mask: Tensor | None = None,
    *,
    texture_reference: Tensor | None = None,
) -> Tensor:
    """Content term that is deliberately weaker than the adversarial term.

    The two halves are evaluated against **different images**, and this matters
    more than it looks.

    The L1 half compares against ``texture_reference`` (defaulting to
    ``prediction``'s caller-supplied pre-effect image). L1 is texture
    sensitive: measured on a trained model, the stochastic grain and halation
    incur a content penalty roughly twice the size of the entire learned
    residual's own deviation from the source. With the adversarial term at
    weight 1.0 and the content term at 1.5, that penalty simply wins, and the
    model learns to suppress the very grain and halation the effect head exists
    to produce. This is the classic "MSE averages out stochastic texture"
    failure, and it is why the L1 term must not see the post-effect image.

    The edge half compares ``prediction`` against ``source``. Structure *should*
    be verified on the final image, because that is the only way to confirm the
    grain and halation did not break geometry. Sobel responds to edges, not to
    fine texture, so it is indifferent to the effect.

    Pass ``texture_reference=deterministic`` and ``prediction=styled`` to get
    the intended split.
    """

    l1_target = prediction if texture_reference is None else texture_reference
    return 0.65 * masked_l1(l1_target, source, mask) + 0.35 * edge_loss(prediction, source)


def moment_loss(prediction: Tensor, target_batch: Tensor) -> Tensor:
    """Match coarse color/exposure statistics without requiring aligned pairs."""

    pred_mean = prediction.mean(dim=(0, 2, 3))
    target_mean = target_batch.mean(dim=(0, 2, 3))
    pred_std = prediction.std(dim=(0, 2, 3), unbiased=False)
    target_std = target_batch.std(dim=(0, 2, 3), unbiased=False)
    return F.l1_loss(pred_mean, target_mean) + 0.5 * F.l1_loss(pred_std, target_std)


def identity_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(prediction, target) + 0.25 * edge_loss(prediction, target)


def effect_regularization(effect_params: Tensor) -> Tensor:
    """Discourage an early generator from producing extreme effect values."""

    return effect_params.square().mean()


def paired_reconstruction(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Supervised reconstruction against a known corresponding target.

    Only available in ``paired`` and ``sameset`` training modes, where the
    target is a genuine photograph of the same scene. This is the only term in
    the objective that receives direct supervision of the target appearance,
    so it anchors appearance while the adversarial term handles texture.

    Evaluated on the **deterministic** output, never the styled one. Pointing an
    L1 term at a stochastic target is the same mistake
    :func:`content_loss` avoids: it would average the grain out of both sides
    and pull the model toward a smooth, grain-free result.

    ``reduction="none"`` returns a per-image tensor, which ``sameset`` needs: in
    that mode only some images in a batch have a corresponding target, so the
    term has to be masked per item rather than applied to the whole batch.
    """

    l1 = masked_l1(prediction, target, mask, reduction=reduction)
    edge = edge_loss(prediction, target, reduction=reduction)
    total = l1 + 0.5 * edge
    return total if reduction == "none" else total.mean()


def psnr(prediction: Tensor, target: Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in decibels.

    Only meaningful in ``paired`` mode, where prediction and target are the
    same scene. Reported for model selection, never used as a training loss.
    """

    mse = F.mse_loss(prediction.clamp(0.0, 1.0), target).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10((data_range**2) / mse))


#: Bumped whenever the objective changes in a way that makes older checkpoints
#: non-comparable. Stored in ``config.json`` and checked on resume.
LOSS_VERSION = 2


def global_ssim(prediction: Tensor, target: Tensor) -> Tensor:
    """Small dependency-free SSIM approximation for evaluation reports."""

    pred_mean = prediction.mean(dim=(2, 3), keepdim=True)
    target_mean = target.mean(dim=(2, 3), keepdim=True)
    pred_var = prediction.var(dim=(2, 3), keepdim=True, unbiased=False)
    target_var = target.var(dim=(2, 3), keepdim=True, unbiased=False)
    covariance = ((prediction - pred_mean) * (target - target_mean)).mean(
        dim=(2, 3), keepdim=True
    )
    numerator = (2 * pred_mean * target_mean + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (pred_mean.square() + target_mean.square() + 0.01**2) * (
        pred_var + target_var + 0.03**2
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()
