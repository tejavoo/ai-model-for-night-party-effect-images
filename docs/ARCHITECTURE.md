# Architecture notes

## Generator

The default generator is a hybrid image-to-image network:

```text
input RGB or RGB+mask
→ shallow convolution
→ three stride-2 depthwise-separable stages
→ low-resolution MobileViT/EdgeViT-style bottleneck
→ bilinear upsampling
→ three decoder stages with skip connections
→ bounded RGB residual
→ eight effect coefficients
```

The residual is bounded by `residual_scale`, currently `0.35`, to prevent early training from destroying the source.

The default configuration is deliberately compact:

```text
base channels:       48
bottleneck channels: 256
transformer blocks:  1
token grid:          16x16
```

This is roughly a sub-million-parameter generator. If the pilot underfits, increase the channel counts gradually, for example `--base-channels 64 --bottleneck-channels 384`, and compare latency before scaling further.

## Stochastic effects

The effect head predicts coefficients, not a full random image.

The Python renderer applies:

- Grain
- Light leak
- Exposure offset
- Saturation
- Vignette
- Leak color tint

This separation is intentional:

- The model learns what style strength is appropriate.
- The renderer controls randomness and reproducibility.
- The app can move the renderer to Metal.
- Core ML does not need a random operator.

## Losses

The unpaired training objective combines:

```text
PatchGAN adversarial loss
content L1 and edge loss
identity loss on target images
batch-level RGB moment matching
effect coefficient regularization
```

There is no paired pixel target.

## Core ML boundary

The export wrapper returns:

```text
styled image
optional effect coefficients
```

Core Image/Metal remains responsible for:

```text
grain
light leaks
bloom
vignette
overlays
```

This boundary makes the model easier to debug and lets you compare deterministic and stochastic outputs independently.
