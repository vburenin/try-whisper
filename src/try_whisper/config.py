"""Configuration helpers for the CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 500
DEFAULT_WINDOW_MS = 5000
DEFAULT_KEEP_MS = 250


@dataclass(slots=True)
class RuntimeConfig:
    model_path: Path
    library_path: Path
    sample_rate: int = DEFAULT_SAMPLE_RATE
    chunk_ms: int = DEFAULT_CHUNK_MS
    window_ms: int = DEFAULT_WINDOW_MS
    keep_ms: int = DEFAULT_KEEP_MS
    threads: int = max(1, os.cpu_count() or 1)
    beam_size: int | None = None
    temperature: float = 0.0
    translate: bool = False
    language: str = "en"
    no_context: bool = True
    no_timestamps: bool = False
    print_special: bool = False
    use_vad: bool = True
    vad_aggressiveness: int = 2
    vad_speech_ratio: float = 0.3
    vad_silence_grace_ms: int = 600
    device_index: int | None = None
    flash_attn: bool = True
    use_gpu: bool = True
    final_only: bool = False

    def __post_init__(self) -> None:
        if self.chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if self.window_ms < self.chunk_ms:
            raise ValueError("window_ms must be >= chunk_ms")
        if self.keep_ms > self.window_ms:
            raise ValueError("keep_ms must be <= window_ms")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.model_path = self.model_path.expanduser().resolve()
        self.library_path = self.library_path.expanduser().resolve()

    @property
    def step_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_ms / 1000)

    @property
    def keep_samples(self) -> int:
        return int(self.sample_rate * self.keep_ms / 1000)
