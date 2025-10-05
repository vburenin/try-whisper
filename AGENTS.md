# Project Overview

This workspace hosts a local, real-time speech recognition CLI built on top of `whisper.cpp` with Metal acceleration for Apple Silicon (macOS). The Python package lives under `src/try_whisper`, and the native whisper backend is vendored in `vendor/whisper.cpp`.

## Key Components

- `vendor/whisper.cpp/`: upstream whisper.cpp repository cloned as a submodule-like vendor; already built with `make METAL=1` producing `build/src/libwhisper.dylib` and helper binaries such as `quantize`.
- `src/try_whisper/`: Python package implementing audio capture (`audio.py`), a ring buffer (`ring_buffer.py`), ctypes bindings to whisper.cpp (`whisper_cpp.py`), the streaming transcription pipeline with optional WebRTC VAD (`transcriber.py`), and the Typer-based CLI (`cli.py`).
- `scripts/download_model.py`: HTTPS + curl-backup downloader for ggml models with optional quantization.
- `scripts/setup_project.sh`: single-entry bootstrap (builds whisper.cpp, creates `.venv`, installs deps, downloads models).
- `tests/`: currently contains `test_ring_buffer.py` covering buffer wraparound/tail behavior.
- `README.md`: setup instructions, including dependency installation, building the Metal backend, downloading models, and running the CLI.

## Runtime Expectations

- Requires Python 3.10+, `sounddevice`, `webrtcvad`, `numpy`, `typer`, and `rich`. Development extras include `pytest`.
- Assumes ggml model files (e.g., `models/ggml-base.en-q5_0.bin`) are downloaded via the helper script.
- The CLI defaults to using the Metal-enabled `libwhisper.dylib` at `vendor/whisper.cpp/build/src/libwhisper.dylib`.
- Real-time transcription reads 16 kHz mono 16-bit PCM from the system microphone, chunks it (default 500 ms), optionally gates through WebRTC VAD, and feeds the rolling window into whisper.cpp for low-latency decoding.

## Usage Summary

1. (Once per machine) Build the native backend:
   ```bash
   make -C vendor/whisper.cpp METAL=1
   ```
2. Install Python dependencies inside a virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev]
   ```
3. Download a model (example with quantization):
   ```bash
   ./scripts/download_model.py base.en --quantize q5_0
   ```
4. Run the CLI:
   ```bash
   python -m try_whisper.cli run --model base.en-q5_0
   ```
   Use `--wav <file>` for offline processing.

## Known Gaps / Next Steps

- No automated audio integration tests; only the ring buffer utilities are covered.
- Need to ensure virtualenv is created before running `pytest` or the CLI (dependencies not yet installed).
- Consider wrapping the ctypes bindings with higher-level error handling or adopting an official Python binding if upstream adds one.
- Potential enhancements: partial transcript UI improvements, logging/metrics export, packaging for distribution.
