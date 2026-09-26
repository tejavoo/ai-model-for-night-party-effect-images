#!/usr/bin/env bash
#
# Build the LaTeX documents in this folder to PDF.
#
# Requires tectonic, which is a single self-contained binary that fetches only
# the packages a document actually uses. It is far lighter than a full TeX Live
# installation:
#
#   brew install tectonic
#
# Usage:
#   ./build.sh                  # build everything
#   ./build.sh technical-report # build one document
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v tectonic >/dev/null 2>&1; then
  cat >&2 <<'MSG'
error: 'tectonic' not found. Install it with:

  brew install tectonic

On Linux, download a release binary from
https://github.com/tectonic-typesetting/tectonic/releases
MSG
  exit 1
fi

DOCS=("$@")
if [ ${#DOCS[@]} -eq 0 ]; then
  DOCS=(technical-report paper)
fi

for name in "${DOCS[@]}"; do
  src="$name.tex"
  if [ ! -f "$src" ]; then
    echo "skip: $src not found" >&2
    continue
  fi
  printf 'building %-22s ' "$src"
  # --keep-logs leaves the .log for debugging a failed build.
  if tectonic -X compile "$src" --outfmt pdf --keep-logs >/dev/null 2>"$name.build.log"; then
    size="$(du -h "$name.pdf" | cut -f1 | tr -d ' ')"
    echo "ok -> $name.pdf ($size)"
  else
    echo "FAILED"
    echo "  see $name.build.log and $name.log" >&2
    exit 1
  fi
done

echo
echo "built:"
for name in "${DOCS[@]}"; do
  [ -f "$name.pdf" ] && echo "  $HERE/$name.pdf"
done
