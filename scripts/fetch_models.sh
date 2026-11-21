#!/bin/bash
# Download the local speech models (Kokoro TTS) into models/. faster-whisper fetches its own
# model from Hugging Face on first use. Safe to re-run; existing files are kept.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  if [ ! -s "$ROOT/models/$f" ]; then
    echo "fetching $f"
    curl -L --progress-bar -o "$ROOT/models/$f" "$BASE/$f"
  else
    echo "have $f"
  fi
done
echo "done"
