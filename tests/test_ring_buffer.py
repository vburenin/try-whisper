from __future__ import annotations

import numpy as np

from try_whisper.ring_buffer import RingBuffer


def test_ring_buffer_wraparound() -> None:
    buf = RingBuffer(5)
    buf.extend(np.array([1, 2, 3], dtype=np.float32))
    assert buf.snapshot().as_array().tolist() == [1.0, 2.0, 3.0]
    buf.extend(np.array([4, 5, 6], dtype=np.float32))
    assert buf.snapshot().as_array().tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_ring_buffer_tail() -> None:
    buf = RingBuffer(4)
    buf.extend(np.array([1, 2, 3, 4], dtype=np.float32))
    tail = buf.tail(2)
    assert tail.tolist() == [3.0, 4.0]
    buf.extend(np.array([5, 6], dtype=np.float32))
    tail = buf.tail(3)
    assert tail.tolist() == [4.0, 5.0, 6.0]

