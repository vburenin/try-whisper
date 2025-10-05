"""Audio capture utilities."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import sounddevice as sd


@dataclass(slots=True)
class AudioChunk:
    data: np.ndarray  # int16 mono samples
    sample_rate: int


class MicrophoneStream:
    """Collects audio frames from the default input device."""

    def __init__(
        self,
        sample_rate: int,
        frame_duration_ms: int = 20,
        device: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        if self.frame_samples <= 0:
            raise ValueError("frame_duration_ms yields non-positive frame size")
        self._device = device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
        self._stream: Optional[sd.InputStream] = None
        self._closed = threading.Event()

    def __enter__(self) -> "MicrophoneStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._closed.clear()
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        self._closed.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield successive int16 frames."""
        if self._stream is None:
            self.start()
        while not self._closed.is_set():
            frame = self._queue.get()
            if frame.size == 0:
                continue
            yield frame

    def read(self, num_samples: int) -> np.ndarray:
        """Blocking read for a specific number of samples."""
        if num_samples <= 0:
            return np.empty(0, dtype=np.int16)
        collected: list[np.ndarray] = []
        total = 0
        for frame in self.frames():
            collected.append(frame)
            total += frame.size
            if total >= num_samples:
                break
        if not collected:
            return np.empty(0, dtype=np.int16)
        audio = np.concatenate(collected)
        if audio.size > num_samples:
            audio = audio[:num_samples]
        return audio

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
        if status:
            # Drop frames on overflow underrun to keep stream responsive
            pass
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # drop frame if queue is congested
            self._queue.get_nowait()
            self._queue.put_nowait(indata[:, 0].copy())


def int16_to_float32(samples: np.ndarray) -> np.ndarray:
    return np.asarray(samples, dtype=np.float32) / 32768.0


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)

