from __future__ import annotations

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


def edge_loss(prediction: Tensor, target: Tensor) -> Tensor:
    kernel_x, kernel_y = _sobel_filters(prediction.device, prediction.dtype)
    channels = prediction.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    pred_x = F.conv2d(prediction, kernel_x, padding=1, groups=channels)
    pred_y = F.conv2d(prediction, kernel_y, padding=1, groups=channels)
    target_x = F.conv2d(target, kernel_x, padding=1, groups=channels)
    target_y = F.conv2d(target, kernel_y, padding=1, groups=channels)
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def masked_l1(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    difference = (prediction - target).abs().mean(dim=1, keepdim=True)
    if mask is None:
        return difference.mean()
    mask = mask.clamp(0.0, 1.0)
    return (difference * mask).sum() / (mask.sum() * difference.shape[1] + 1e-6)


def content_loss(prediction: Tensor, source: Tensor, mask: Tensor | None = None) -> Tensor:
    """Content term that is deliberately weaker than the adversarial term."""

    return 0.65 * masked_l1(prediction, source, mask) + 0.35 * edge_loss(prediction, source)


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
