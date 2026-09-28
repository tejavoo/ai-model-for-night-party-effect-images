#!/usr/bin/env bash
#
# Package the repository for upload to Kaggle.
#
# Produces dist/nightflash-edge-kaggle.zip containing the code, docs and
# the Kaggle notebook. Datasets, runs, checkpoints and git history are excluded
# because they are large and are handled separately (Kaggle Datasets).
#
# Usage:
#   ./scripts/make_kaggle_bundle.sh
#   bash scripts/make_kaggle_bundle.sh             # works on macOS and Linux
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="$ROOT/dist"
ZIP_PATH="$OUT_DIR/nightflash-edge-kaggle.zip"

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
echo "Contains:"
echo "  notebook/nightflash_kaggle.ipynb   -> Kaggle (7 cells)"
echo "  docs/QuickStartKaggle.md           -> full Kaggle quickstart"
echo "  docs/KAGGLE.md                     -> Kaggle setup and troubleshooting"
echo "  src/                               -> all source code"
echo "  scripts/                           -> utility scripts"
echo
echo "Next:"
echo "  Kaggle : upload this zip as a Dataset, attach it in the Add Data panel."
echo "           Kaggle auto-extracts it into /kaggle/input/. No unzip needed."
echo "           Then open notebook/nightflash_kaggle.ipynb"
