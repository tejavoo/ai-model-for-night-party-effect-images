#!/usr/bin/env bash
#
# Package the repository for upload to Google Colab.
#
# Produces dist/nightflash-edge-colab.zip containing only the code, docs and
# the Colab notebook. Datasets, runs, checkpoints and git history are excluded
# because they are large and are handled separately (Drive / clone).
#
# Usage:
#   ./scripts/make_colab_bundle.sh
#   python scripts/make_colab_bundle.sh          # works on macOS and Linux
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="$ROOT/dist"
ZIP_PATH="$OUT_DIR/nightflash-edge-colab.zip"

if ! command -v zip >/dev/null 2>&1; then
  echo "error: 'zip' not found. Install it (macOS: brew install zip)." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$ZIP_PATH"

# Patterns are relative to the repo root.
zip -q -r "$ZIP_PATH" . \
  -x '.git/*' \
  -x '.github/*' \
  -x '.venv/*' \
  -x 'venv/*' \
  -x 'dist/*' \
  -x 'training_photos/*' \
  -x 'training_photos.zip' \
  -x 'data/*' \
  -x 'datasets/*' \
  -x 'runs/*' \
  -x 'outputs/*' \
  -x 'checkpoints/*' \
  -x '__pycache__/*' \
  -x '*.egg-info/*' \
  -x '*.pyc' \
  -x '*.pt' \
  -x '*.pth' \
  -x '*.safetensors' \
  -x '*.mlpackage/*' \
  -x '*.mlmodel' \
  -x '.pytest_cache/*' \
  -x '.DS_Store' \
  -x '*/.DS_Store' \
  -x '*_Downloaded_from_google_drive*' \
  -x '*.ipynb_checkpoints*'

SIZE="$(du -h "$ZIP_PATH" | cut -f1 | tr -d ' ')"
echo "created $ZIP_PATH ($SIZE)"
echo
echo "Contains both notebooks:"
echo "  colab/NightFlash_Colab.ipynb        -> Google Colab"
echo "  notebook/nightflash_kaggle.ipynb    -> Kaggle"
echo
echo "Next:"
echo "  Colab  : upload this zip to /content/, attach training_photos/ on Drive,"
echo "           then open colab/NightFlash_Colab.ipynb"
echo "  Kaggle : clone or unzip into /kaggle/working/nightflash-edge, attach"
echo "           training_photos/ as a Dataset, then open"
echo "           notebook/nightflash_kaggle.ipynb"
