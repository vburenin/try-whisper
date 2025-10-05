"""Streaming transcription pipeline."""
from __future__ import annotations

import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import webrtcvad

from .audio import MicrophoneStream, int16_to_float32
from .config import RuntimeConfig
from .ring_buffer import RingBuffer
from .whisper_cpp import Segment, WhisperContext

WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1

SegmentCallback = Callable[[Segment, str, bool], None]
MetricsCallback = Callable[[Dict[str, float]], None]


class VadGate:
    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int,
        min_speech_ratio: float,
        frame_ms: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.aggressiveness = int(max(0, min(3, aggressiveness)))
        self.min_speech_ratio = min(1.0, max(0.0, min_speech_ratio))
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        if self.frame_samples <= 0:
            raise ValueError("frame_ms leads to zero frame size")
        self._vad = webrtcvad.Vad(self.aggressiveness)

    def is_speech(self, chunk: np.ndarray) -> bool:
        if chunk.size < self.frame_samples:
            return False
        pcm_bytes = chunk.tobytes()
        frame_bytes = self.frame_samples * 2
        total = 0
        speech_frames = 0
        for offset in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes):
            frame = pcm_bytes[offset : offset + frame_bytes]
            total += 1
            if self._vad.is_speech(frame, self.sample_rate):
                speech_frames += 1
        if total == 0:
            return False
        return (speech_frames / total) >= self.min_speech_ratio


class StreamingTranscriber:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.context = WhisperContext(
            config.model_path,
            config.library_path,
            use_gpu=config.use_gpu,
            flash_attn=config.flash_attn,
        )
        self.buffer = RingBuffer(config.window_samples)
        self.prompt_tokens: list[int] = []
        self.history: "OrderedDict[tuple[int, int], Segment]" = OrderedDict()
        self.history_limit = 256
        self.vad: Optional[VadGate] = None
        if config.use_vad:
            self.vad = VadGate(
                sample_rate=config.sample_rate,
                aggressiveness=config.vad_aggressiveness,
                min_speech_ratio=config.vad_speech_ratio,
            )
        self._in_speech = False
        self._silence_ms = 0
        self._final_only = bool(config.final_only and self.vad is not None)
        self._pending_segments: "OrderedDict[tuple[int, int], Segment]" = OrderedDict()

    def close(self) -> None:
        self.context.close()

    def flush(self, emit_segment: SegmentCallback) -> None:
        if self._final_only:
            self._flush_pending(emit_segment)

    def run_microphone(
        self,
        emit_segment: SegmentCallback,
        *,
        emit_metrics: Optional[MetricsCallback] = None,
    ) -> None:
        with MicrophoneStream(
            sample_rate=self.config.sample_rate,
            frame_duration_ms=20,
            device=self.config.device_index,
        ) as stream:
            try:
                while True:
                    chunk = stream.read(self.config.step_samples)
                    if chunk.size == 0:
                        continue
                    self._process_chunk(chunk, emit_segment, emit_metrics)
            except KeyboardInterrupt:
                if self._final_only:
                    self._flush_pending(emit_segment)
                return

    def run_wav(
        self,
        wav_path: Path,
        emit_segment: SegmentCallback,
        *,
        emit_metrics: Optional[MetricsCallback] = None,
        realtime: bool = False,
    ) -> None:
        samples = self._read_wav(wav_path)
        chunk_size = self.config.step_samples
        start = 0
        while start < samples.size:
            chunk = samples[start : start + chunk_size]
            if chunk.size == 0:
                break
            self._process_chunk(chunk, emit_segment, emit_metrics)
            if realtime:
                time.sleep(self.config.chunk_ms / 1000.0)
            start += chunk_size
        if self._final_only:
            self._flush_pending(emit_segment)

    def _process_chunk(
        self,
        chunk: np.ndarray,
        emit_segment: SegmentCallback,
        emit_metrics: Optional[MetricsCallback],
    ) -> None:
        if self.vad is not None:
            speech = self.vad.is_speech(chunk)
            if speech:
                self._in_speech = True
                self._silence_ms = 0
            else:
                if self._in_speech:
                    self._silence_ms += self.config.chunk_ms
                    if self._silence_ms <= self.config.vad_silence_grace_ms:
                        speech = True
                    else:
                        self._in_speech = False
                        self._silence_ms = 0
                        self.prompt_tokens.clear()
                        if self._final_only:
                            self._flush_pending(emit_segment)
                if not speech:
                    return
        else:
            self._in_speech = True
        float_chunk = int16_to_float32(chunk)
        self.buffer.extend(float_chunk)
        snapshot = self.buffer.snapshot().as_array()
        if snapshot.size == 0:
            return
        strategy = (
            WHISPER_SAMPLING_GREEDY
            if not self.config.beam_size or self.config.beam_size <= 1
            else WHISPER_SAMPLING_BEAM_SEARCH
        )
        t_start = time.perf_counter()
        segments, tokens = self.context.transcribe(
            snapshot,
            n_threads=self.config.threads,
            strategy=strategy,
            beam_size=self.config.beam_size,
            translate=self.config.translate,
            no_context=self.config.no_context,
            no_timestamps=self.config.no_timestamps,
            language=self.config.language,
            temperature=self.config.temperature,
            prompt_tokens=self.prompt_tokens,
            max_tokens=0,
            audio_ctx=0,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if not self.config.no_context:
            self.prompt_tokens = tokens
        if self._final_only:
            self._update_pending_segments(segments)
        else:
            self._emit_streaming_segments(segments, emit_segment)
        if emit_metrics:
            emit_metrics(
                {
                    "processing_ms": elapsed_ms,
                    "rtf": elapsed_ms / max(1.0, self.config.chunk_ms),
                    "window_samples": float(snapshot.size),
                }
            )

    def _emit_streaming_segments(self, segments: Iterable[Segment], emit: SegmentCallback) -> None:
        for segment in segments:
            if not segment.text:
                continue
            key = (segment.start_ms, segment.end_ms)
            prev = self.history.get(key)
            text_to_emit = segment.text
            is_update = prev is not None
            if prev is not None:
                if segment.text == prev.text:
                    continue
                if segment.text.startswith(prev.text):
                    suffix = segment.text[len(prev.text) :].lstrip()
                    if not suffix:
                        continue
                    text_to_emit = suffix
                else:
                    text_to_emit = segment.text
            if prev is None:
                is_update = False
            emit(segment, text_to_emit, is_update)
            self.history[key] = segment
            self.history.move_to_end(key)
            if len(self.history) > self.history_limit:
                self.history.popitem(last=False)

    def _update_pending_segments(self, segments: Iterable[Segment]) -> None:
        for segment in segments:
            if not segment.text:
                continue
            key = (segment.start_ms, segment.end_ms)
            self._pending_segments[key] = segment

    def _flush_pending(self, emit: SegmentCallback) -> None:
        if not self._pending_segments:
            return
        for segment in self._pending_segments.values():
            emit(segment, segment.text, False)
            key = (segment.start_ms, segment.end_ms)
            self.history[key] = segment
            self.history.move_to_end(key)
            if len(self.history) > self.history_limit:
                self.history.popitem(last=False)
        self._pending_segments.clear()

    def _read_wav(self, path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError("Only mono WAV files are supported")
            if wf.getsampwidth() != 2:
                raise ValueError("Expected 16-bit PCM WAV")
            if wf.getframerate() != self.config.sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: {wf.getframerate()} != {self.config.sample_rate}"
                )
            pcm = wf.readframes(wf.getnframes())
            return np.frombuffer(pcm, dtype=np.int16)
