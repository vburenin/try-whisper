#!/usr/bin/env python3
"""Download ggml Whisper models and optional quantization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
MODEL_NAMES = {
    "tiny": "ggml-tiny.bin",
    "tiny.en": "ggml-tiny.en.bin",
    "base": "ggml-base.bin",
    "base.en": "ggml-base.en.bin",
    "small": "ggml-small.bin",
    "small.en": "ggml-small.en.bin",
    "medium": "ggml-medium.bin",
    "medium.en": "ggml-medium.en.bin",
    "large": "ggml-large-v3.bin",
}
QUANT_SUFFIX = "ggml-{}-{}.bin"


class DownloadError(Exception):
    pass


def stream_to_file(response: requests.Response, dest: Path) -> None:
    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    block_size = 1 << 14
    with dest.open("wb") as out:
        for chunk in response.iter_content(block_size):
            if not chunk:
                continue
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = 100 * downloaded / total
                print(f"\rDownloading {dest.name}: {percent:5.1f}%", end="", flush=True)
    if total:
        print(f"\rDownloading {dest.name}: done".ljust(40))
    else:
        print(f"Downloaded {dest.name} ({downloaded} bytes)")


def download_with_requests(url: str, dest: Path, *, verify: bool = True) -> None:
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True, verify=verify) as response:
            response.raise_for_status()
            stream_to_file(response, dest)
    except requests.exceptions.SSLError as exc:
        raise DownloadError("SSL certificate verification failed") from exc
    except requests.RequestException as exc:
        raise DownloadError(str(exc)) from exc


def download_with_curl(url: str, dest: Path) -> None:
    import subprocess

    print("Falling back to curl...")
    cmd = ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5", "-o", str(dest), url]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise DownloadError("curl executable not found; install curl or fix SSL certificates") from exc
    except subprocess.CalledProcessError as exc:
        raise DownloadError(f"curl failed with exit code {exc.returncode}") from exc


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_with_requests(url, dest)
    except DownloadError as first_error:
        try:
            download_with_curl(url, dest)
        except DownloadError as second_error:
            raise SystemExit(f"Failed to download {url}: {second_error}") from first_error


def run_quantize(model: Path, quantize: str, output: Path) -> None:
    quant_binary = Path("vendor/whisper.cpp/build/bin/quantize").resolve()
    if not quant_binary.is_file():
        raise SystemExit("quantize binary not found. Build whisper.cpp with `make METAL=1` first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    cmd = [str(quant_binary), str(model), str(output), quantize]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_NAMES.keys(), help="Model to download")
    parser.add_argument(
        "--quantize",
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
        help="Optional quantization to produce.",
    )
    parser.add_argument(
        "--dest",
        default="models",
        type=Path,
        help="Destination directory for models.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args(argv)

    dest_dir: Path = args.dest.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = MODEL_NAMES[args.model]
    model_path = dest_dir / filename

    if model_path.exists() and not args.force:
        print(f"Model {model_path} already exists. Skipping download.")
    else:
        url = f"{MODEL_BASE_URL}/{filename}?download=1"
        print(f"Fetching {url}")
        download(url, model_path)

    if args.quantize:
        stem = filename.removeprefix("ggml-").removesuffix(".bin")
        quant_name = QUANT_SUFFIX.format(stem, args.quantize)
        quant_path = dest_dir / quant_name
        if quant_path.exists() and not args.force:
            print(f"Quantized file {quant_path} already exists. Skipping.")
        else:
            run_quantize(model_path, args.quantize, quant_path)
            print(f"Saved quantized model to {quant_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
