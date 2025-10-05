#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_DIR="$ROOT_DIR/vendor/whisper.cpp"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "whisper.cpp already present at $TARGET_DIR"
else
  mkdir -p "$ROOT_DIR/vendor"
  git clone https://github.com/ggerganov/whisper.cpp.git "$TARGET_DIR"
fi

(cd "$TARGET_DIR" && make METAL=1)
