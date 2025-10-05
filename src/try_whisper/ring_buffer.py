"""Lightweight ring buffer for audio samples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class RingBufferSnapshot:
    """Immutable view over the buffer contents."""

    data: np.ndarray

    def as_array(self) -> np.ndarray:
        return self.data


class RingBuffer:
    """A fixed-size ring buffer storing float32 audio samples."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._length = 0
        self._start = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def length(self) -> int:
        return self._length

    def clear(self) -> None:
        self._length = 0
        self._start = 0

    def extend(self, samples: Iterable[float] | np.ndarray) -> None:
        if isinstance(samples, np.ndarray):
            data = np.ascontiguousarray(samples, dtype=np.float32)
        else:
            data = np.fromiter(samples, dtype=np.float32)
        if data.size == 0:
            return
        if data.size >= self._capacity:
            self._buffer[:] = data[-self._capacity :]
            self._start = 0
            self._length = self._capacity
            return
        end = (self._start + self._length) % self._capacity
        first_chunk = min(data.size, self._capacity - end)
        self._buffer[end : end + first_chunk] = data[:first_chunk]
        remaining = data.size - first_chunk
        if remaining:
            self._buffer[0:remaining] = data[first_chunk:]
        if self._length == self._capacity:
            self._start = (self._start + data.size) % self._capacity
        else:
            new_length = self._length + data.size
            if new_length > self._capacity:
                overflow = new_length - self._capacity
                self._start = (self._start + overflow) % self._capacity
                new_length = self._capacity
            self._length = new_length

    def snapshot(self) -> RingBufferSnapshot:
        if self._length == 0:
            return RingBufferSnapshot(np.empty(0, dtype=np.float32))
        end = (self._start + self._length) % self._capacity
        if end > self._start:
            data = self._buffer[self._start:end].copy()
        else:
            data = np.concatenate((self._buffer[self._start:], self._buffer[:end])).astype(
                np.float32
            )
        return RingBufferSnapshot(data)

    def tail(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0 or self._length == 0:
            return np.empty(0, dtype=np.float32)
        n = min(n_samples, self._length)
        start = (self._start + self._length - n) % self._capacity
        end = (start + n) % self._capacity
        if end > start:
            return self._buffer[start:end].copy()
        return np.concatenate((self._buffer[start:], self._buffer[:end])).astype(np.float32)

