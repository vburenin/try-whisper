#!/usr/bin/env bash
# Bootstrap the entire project: clone/build whisper.cpp, create venv, install deps, download models.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.10}
VENV_DIR="$ROOT_DIR/.venv"
MODELS="base.en-q5_0"
DOWNLOAD_ALL=false

usage() {
  cat <<USAGE
Usage: $0 [--python PATH] [--models name[,name...]] [--all-models]

Options:
  --python PATH        Python interpreter to use (default: python3.10)
  --models LIST        Comma-separated list of models to download via scripts/download_model.py.
                       Use suffix "-q5_0" to quantize (e.g. base.en-q5_0). Defaults to base.en-q5_0.
  --all-models         Download and quantize the full whisper.cpp model suite (tiny..large-v3).
  -h, --help           Show this help message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      shift
      PYTHON_BIN=${1:?"--python requires a value"}
      ;;
    --models)
      shift
      MODELS=${1:?"--models requires a value"}
      ;;
    --all-models)
      DOWNLOAD_ALL=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift || true
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' not found." >&2
  exit 1
fi

echo "[1/5] Ensuring whisper.cpp backend is built"
"$ROOT_DIR/scripts/setup_whisper_cpp.sh"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[2/5] Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[2/5] Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[3/5] Upgrading pip"
pip install --upgrade pip >/dev/null

echo "[4/5] Installing project dependencies"
pip install -e .[dev]

declare -a MODEL_LIST
if $DOWNLOAD_ALL; then
  MODEL_LIST=(tiny tiny.en base base.en small small.en medium medium.en large large-v3)
else
  IFS=',' read -r -a MODEL_LIST <<<"$MODELS"
fi

if [[ ${#MODEL_LIST[@]} -gt 0 ]]; then
  echo "[5/5] Downloading models: ${MODEL_LIST[*]}"
  for model in "${MODEL_LIST[@]}"; do
    if [[ "$model" == *"-q5_0" ]]; then
      base_model=${model%-q5_0}
      python "$ROOT_DIR/scripts/download_model.py" "$base_model" --quantize q5_0
    elif [[ "$model" == *"-q4_0" || "$model" == *"-q4_1" || "$model" == *"-q5_1" || "$model" == *"-q8_0" ]]; then
      base_model=${model%-*}
      quant="${model##*-}"
      python "$ROOT_DIR/scripts/download_model.py" "$base_model" --quantize "$quant"
    else
      python "$ROOT_DIR/scripts/download_model.py" "$model"
    fi
  done
else
  echo "[5/5] No models requested"
fi

echo "\nSetup complete!"
echo "Activate the environment with: source $VENV_DIR/bin/activate"
echo "Run the CLI with: python -m try_whisper.cli --model base.en-q5_0"
