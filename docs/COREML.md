# Core ML export and iOS integration

## Install the exporter

Run this on macOS:

```bash
cd /Users/tejavu/Documents/opencode-experiments/disposable-cam-model-train
python3 -m pip install -e '.[coreml]'
```

`coremltools` is not required for training. It is only needed to export the model.

## Export a deterministic model

For the simplest integration, export only the styled image:

```bash
nightflash-export-coreml \
  --checkpoint runs/hybrid-512/checkpoints/latest.pt \
  --output runs/hybrid-512/NightFlashEdge.mlpackage \
  --image-size 512 \
  --half \
  --deterministic-only
```

This produces a model with:

```text
inputImage  -> styledImage
```

The stochastic grain/leak renderer can remain in Swift/Metal or Core Image.

## Export the effect coefficients too

```bash
nightflash-export-coreml \
  --checkpoint runs/hybrid-512/checkpoints/latest.pt \
  --output runs/hybrid-512/NightFlashEdgeEffects.mlpackage \
  --image-size 512 \
  --half
```

Outputs:

```text
styledImage
 effectParams[8]
```

The eight values are:

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

The values are sigmoid outputs, generally in the range `0...1`.

## Export a `.mlmodel` file

`.mlpackage` is recommended for ML Program models:

```bash
nightflash-export-coreml \
  --checkpoint runs/hybrid-512/checkpoints/latest.pt \
  --output runs/hybrid-512/NightFlashEdge.mlmodel \
  --image-size 512 \
  --convert-to mlmodel
```

Use `.mlmodel` when a legacy single-file Neural Network package is required. Test it with the exact Xcode/iOS version used by the app.

## Tensor I/O for deterministic validation

ImageType preprocessing is convenient in Xcode, but coremltools' Python image prediction path can make scale handling confusing. For numerical PyTorch/Core ML comparison, export tensor I/O:

```bash
nightflash-export-coreml \
  --checkpoint runs/hybrid-512/checkpoints/latest.pt \
  --output runs/hybrid-512/NightFlashEdgeTensor.mlpackage \
  --image-size 512 \
  --half \
  --input-type tensor
```

The contract becomes:

```text
inputTensor:  [1,3,H,W] float32 normalized to [0,1]
styledImage:  [1,3,H,W] float32
```

For mask-aware models, the input is `[1,4,H,W]`. This mode is usually easiest to compare directly against PyTorch. The app must construct the normalized pixel buffer/MultiArray itself.

Use image I/O when you want Vision/Core ML image convenience; use tensor I/O when exact normalization and numerical validation matter most.

## Mask-aware export

If the checkpoint was trained with `--mask-dir`:

```bash
nightflash-export-coreml \
  --checkpoint runs/mask-aware/checkpoints/latest.pt \
  --output runs/mask-aware/NightFlashEdgeMask.mlpackage \
  --image-size 512 \
  --half \
  --mask-input
```

The input is a four-channel tensor:

```text
RGB[3] + person mask[1]
```

The model input is normalized to `[0, 1]`. The mask is resized with nearest-neighbor interpolation.

## Expected model contract

### Deterministic export

```text
inputImage:  [1,3,H,W] image, scale 1/255
styledImage: [1,3,H,W] image output
```

### Effect export

```text
inputImage:   [1,3,H,W]
styledImage:  [1,3,H,W]
effectParams: [1,8]
```

### Mask export

Mask exports use tensor I/O:

```text
inputTensor:  [1,4,H,W]
styledImage:  [1,3,H,W]
effectParams: [1,8]
```

Always inspect the generated model in Xcode after conversion. Names and feature types can vary slightly between `coremltools` versions.

## Add the package to the iOS project

1. Copy the `.mlpackage` directory into the app project.
2. Add it to the app target's **Copy Bundle Resources** phase.
3. Confirm the deployment target is compatible with the exported model.
4. Load it from the main bundle:

```swift
guard let modelURL = Bundle.main.url(
    forResource: "NightFlashEdge",
    withExtension: "mlpackage"
) else {
    fatalError("Missing NightFlashEdge.mlpackage")
}

let model = try MLModel(contentsOf: modelURL)
```

Depending on how the package is added to Xcode, the resource lookup can differ. Verify the actual bundle path with a small debug print.

## Swift prediction outline

For a model with an image input:

```swift
let provider = try MLDictionaryFeatureProvider(dictionary: [
    "inputImage": MLFeatureValue(image: inputImage)
])

let output = try model.prediction(from: provider)

if let styledImage = output.featureValue(for: "styledImage")?.imageValue {
    // Continue with Core Image / Metal finishing.
}

if let effectArray = output.featureValue(for: "effectParams")?.multiArrayValue {
    // Read the eight coefficients and apply seeded effects in Swift/Metal.
}
```

If coremltools exports the styled output as a multi-array rather than an image, convert the `MLMultiArray` to a `CGImage` using a known scale and color layout. Prefer the image-typed output where supported.

## Applying stochastic effects on iOS

Do not rely on a random operator inside the Core ML model. Generate the random field in Swift or Metal so that you control:

- Seed reproducibility
- Temporal smoothing
- Effect strength
- Device consistency

For each frame or shot:

```text
seed
→ deterministic noise texture
→ grain amplitude from effectParams[0]
→ low-resolution leak basis from effectParams[1,5,6,7]
→ exposure/saturation from effectParams[2,3]
→ vignette from effectParams[4]
```

Use the same equations as the Python reference implementation in:

```text
src/nightflash/effects.py
```

For video, sample a seed per shot and smoothly interpolate effect coefficients over time. Do not independently sample a new grain pattern every frame.

## Recommended app integration

Keep the existing pipeline architecture:

```text
AVCapture frame
→ resize/normalize
→ Vision person mask
→ NightFlashEdge Core ML model
→ Swift/Metal stochastic effects
→ existing Core Image finishing
→ result/save/share
```

The current project files to adapt are:

```text
/Users/tejavu/Documents/opencode-experiments/disposable-cam/Services/NightFlashModel.swift
/Users/tejavu/Documents/opencode-experiments/disposable-cam/Services/NightFlashProcessor.swift
```

The current `NightFlashModel.swift` adapter expects a 512×512 image-buffer model. The new exporter uses a fixed 512×512 image input by default. You can either:

1. Keep the old adapter and use a deterministic export, or
2. Add a new adapter for the new model and keep the old model as a fallback.

Do not silently replace the old model until the new output has passed visual and performance tests.

## Core ML troubleshooting

### Conversion fails on attention

Try one of these:

- Export a convolution-only checkpoint trained with `--no-transformer`
- Use `mlmodel` conversion instead of `mlprogram`
- Remove `--half`
- Pin the coremltools version
- Use a fixed input size
- Simplify unsupported operations before integrating

### Model exports but is slow

- Benchmark the UNet/model separately from Core Image
- Use float16 only after validating color accuracy
- Avoid unnecessary dynamic operations
- Use GPU compute units on Mac
- Do not assume Neural Engine is faster for a large model
- Keep high-resolution effects in Core Image/Metal

### Model output differs from PyTorch

Check:

- RGB/BGR order
- Input scale `0...1` versus `0...255`
- Image orientation
- Color profile
- Output clamping
- Resize algorithm
- Effect seed
- Whether the stochastic renderer was included in the test

## Validate the exported model numerically

For tensor I/O, compare PyTorch and Core ML directly:

```bash
nightflash-validate-coreml \
  --checkpoint runs/hybrid-512/checkpoints/latest.pt \
  --coreml runs/hybrid-512/NightFlashEdgeTensor.mlpackage \
  --input data/validation/source/example.jpg \
  --image-size 512 \
  --output-dir runs/hybrid-512/coreml-validation
```

Inspect:

```text
mean_absolute_error
max_absolute_error
rmse
```

A small float16 error is expected. A large error usually indicates one of:

- Wrong input scale
- RGB/BGR mismatch
- Wrong image orientation
- Output type conversion
- Different preprocessing
- Incorrect checkpoint/config

## Performance benchmark

Record separately:

```text
model load time
first inference
steady-state p50
steady-state p95
memory
energy
camera frame rate
```

Test on both:

```text
M3 Ultra Mac
target iPhone
```

The M3 Ultra result for a diffusion model does not predict iPhone performance.
