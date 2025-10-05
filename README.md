# try-whisper

Real-time speech recognition CLI for macOS that streams microphone audio through [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp) with Metal acceleration.

## Prerequisites

- macOS with Xcode Command Line Tools installed (`xcode-select --install`)
- Python 3.10+
- Homebrew packages: `brew install cmake portaudio`
- Create and activate a virtual environment and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e .[dev]
  ```

## Quick start

```bash
./scripts/setup_project.sh --all-models  # optional --all-models downloads every ggml variant
source .venv/bin/activate
python -m try_whisper.cli --model base.en-q5_0 --final-only
```

The setup script clones/builds `whisper.cpp`, creates a Python virtual environment, installs dependencies, and downloads the requested models (default: `base.en` with `q5_0` quantization). Use `./scripts/setup_project.sh --help` for customization.

## Manual native backend build

```bash
./scripts/setup_whisper_cpp.sh
```

This produces `vendor/whisper.cpp/build/src/libwhisper.dylib` and tools such as `quantize`.

## Download a model

Use the helper script to grab a model and optional quantized variant:

```bash
./scripts/download_model.py base.en --quantize q5_0
```

The files are stored in `models/` by default, e.g. `models/ggml-base.en-q5_0.bin`.

## Run the CLI

```bash
python -m try_whisper.cli --help
python -m try_whisper.cli --model base.en-q5_0
```

The CLI prints timestamped segments as they are produced. Use `Ctrl+C` to stop.

### Final-only mode

Add `--final-only` to print only the final transcription for each utterance (VAD must be enabled; the flag turns it on automatically):

```bash
python -m try_whisper.cli --model large-v3-q5_0 --final-only --no-timestamps
```

### Offline testing with a WAV file

```bash
python -m try_whisper.cli run --wav path/to/sample.wav --realtime-playback
```

The WAV file must be 16-bit mono at 16 kHz.

## Testing

```bash
pytest
```

This currently exercises the ring buffer utilities.

## Project layout

```
src/try_whisper/      # Python package (audio capture, transcriber, CLI)
vendor/whisper.cpp/   # whisper.cpp source tree (cloned via setup script)
models/               # Downloaded ggml models (ignored by git)
scripts/              # Utility scripts (native setup, model download)
```

## Repository hygiene

- Large model binaries and the `vendor/whisper.cpp` checkout are ignored by git; CI/bootstrap scripts recreate them.
- Use `scripts/setup_whisper_cpp.sh` and `scripts/download_model.py` on fresh clones.
