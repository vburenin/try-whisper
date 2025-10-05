"""Command-line interface for try-whisper."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import RuntimeConfig
from .transcriber import StreamingTranscriber

app = typer.Typer(add_completion=False)
console = Console()


def format_timestamp(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def resolve_model(model: str, model_dir: Path) -> Path:
    candidate = Path(model)
    if candidate.is_file():
        return candidate
    if candidate.suffix == ".bin":
        candidate = (model_dir / candidate).resolve()
        if candidate.is_file():
            return candidate
    if not candidate.suffix:
        guess = model_dir / f"ggml-{candidate.name}.bin"
        if guess.is_file():
            return guess.resolve()
        guess_q = model_dir / f"ggml-{candidate.name}-q5_0.bin"
        if guess_q.is_file():
            return guess_q.resolve()
    raise typer.BadParameter(f"Could not resolve model '{model}'.")


@app.command()
def run(
    model: str = typer.Option(
        "base.en-q5_0",
        help="Model file name or path (e.g. ggml-base.en-q5_0.bin)",
    ),
    model_dir: Path = typer.Option(Path("models"), help="Directory containing ggml models."),
    library: Path = typer.Option(
        Path("vendor/whisper.cpp/build/src/libwhisper.dylib"),
        help="Path to libwhisper shared library.",
    ),
    chunk_ms: int = typer.Option(500, help="Audio chunk size in milliseconds."),
    window_ms: int = typer.Option(5000, help="Sliding window length in milliseconds."),
    keep_ms: int = typer.Option(250, help="Duration to keep from previous window (ms)."),
    threads: Optional[int] = typer.Option(None, help="Number of inference threads."),
    beam_size: Optional[int] = typer.Option(None, help="Beam width (>=2 enables beam search)."),
    translate: bool = typer.Option(False, help="Enable translation to English."),
    language: str = typer.Option("en", help="Language hint or 'auto'."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    no_context: bool = typer.Option(True, help="Disable cross-chunk context."),
    no_timestamps: bool = typer.Option(False, help="Suppress timestamps in output."),
    vad: bool = typer.Option(True, help="Enable WebRTC voice activity detection."),
    vad_aggressiveness: int = typer.Option(2, min=0, max=3, help="VAD aggressiveness (0-3)."),
    vad_speech_ratio: float = typer.Option(0.3, help="Ratio of frames flagged as speech."),
    vad_silence_grace_ms: int = typer.Option(600, help="Grace period before considering silence."),
    device: Optional[int] = typer.Option(None, help="Input device index (sounddevice)."),
    final_only: bool = typer.Option(
        False,
        help="Print only finalised segments (requires VAD).",
    ),
    wav: Optional[Path] = typer.Option(None, help="Optional WAV file for offline testing."),
    realtime_playback: bool = typer.Option(False, help="When using --wav, sleep to emulate realtime."),
) -> None:
    """Run real-time speech recognition."""

    model_dir = model_dir.expanduser().resolve()
    resolved_model = resolve_model(model, model_dir)
    resolved_library = library.expanduser().resolve()
    if not resolved_library.is_file():
        raise typer.BadParameter(f"Shared library not found at {resolved_library}")

    if final_only and not vad:
        console.print("[yellow]Enabling VAD because --final-only requires it.[/]")
        vad = True

    num_threads = threads or max(1, min(16, (os.cpu_count() or 4)))

    runtime = RuntimeConfig(
        model_path=resolved_model,
        library_path=resolved_library,
        sample_rate=16000,
        chunk_ms=chunk_ms,
        window_ms=window_ms,
        keep_ms=keep_ms,
        threads=num_threads,
        beam_size=beam_size,
        temperature=temperature,
        translate=translate,
        language=language,
        no_context=no_context,
        no_timestamps=no_timestamps,
        use_vad=vad,
        vad_aggressiveness=vad_aggressiveness,
        vad_speech_ratio=vad_speech_ratio,
        vad_silence_grace_ms=vad_silence_grace_ms,
        device_index=device,
        final_only=final_only,
    )

    console.print(
        f"[bold green]Loading[/] model: {runtime.model_path} | lib: {runtime.library_path}"
    )

    transcriber = StreamingTranscriber(runtime)

    def emit_segment(segment, text: str, is_update: bool) -> None:
        stamp = format_timestamp(segment.start_ms)
        prefix = "~" if is_update else "➜"
        content = text if runtime.no_timestamps else f"[{stamp}] {text}"
        console.print(f"{prefix} {content}")

    last_metric = time.perf_counter()

    def emit_metrics(stats) -> None:
        nonlocal last_metric
        now = time.perf_counter()
        if now - last_metric >= 2.0:
            console.log(
                "processing={:.1f} ms rtf={:.2f} window={} samples".format(
                    stats.get("processing_ms", 0.0),
                    stats.get("rtf", 0.0),
                    int(stats.get("window_samples", 0.0)),
                )
            )
            last_metric = now

    def handle_exit(signum, frame):
        console.print("\n[bold yellow]Stopping...[/]")
        transcriber.close()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_exit)

    try:
        if wav:
            console.print(f"Running against file {wav} (realtime={realtime_playback})")
            transcriber.run_wav(wav, emit_segment, emit_metrics=emit_metrics, realtime=realtime_playback)
        else:
            console.print("Capturing microphone input. Press Ctrl+C to stop.")
            transcriber.run_microphone(emit_segment, emit_metrics=emit_metrics)
    finally:
        transcriber.flush(emit_segment)
        transcriber.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
