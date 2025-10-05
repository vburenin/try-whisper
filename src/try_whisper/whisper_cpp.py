"""Minimal ctypes wrapper around libwhisper."""
from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_void_p,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


LIB_NAME = "libwhisper.dylib"


class WhisperError(RuntimeError):
    pass


class WhisperAhead(ctypes.Structure):
    _fields_ = [
        ("n_text_layer", c_int),
        ("n_head", c_int),
    ]


class WhisperAheads(ctypes.Structure):
    _fields_ = [
        ("n_heads", c_size_t),
        ("heads", POINTER(WhisperAhead)),
    ]


class WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("flash_attn", c_bool),
        ("gpu_device", c_int),
        ("dtw_token_timestamps", c_bool),
        ("dtw_aheads_preset", c_int),
        ("dtw_n_top", c_int),
        ("dtw_aheads", WhisperAheads),
        ("dtw_mem_size", c_size_t),
    ]


class WhisperVadParams(ctypes.Structure):
    _fields_ = [
        ("threshold", c_float),
        ("min_speech_duration_ms", c_int),
        ("min_silence_duration_ms", c_int),
        ("max_speech_duration_s", c_float),
        ("speech_pad_ms", c_int),
        ("samples_overlap", c_float),
    ]


class WhisperGrammarElement(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("value", ctypes.c_uint32),
    ]


class WhisperGreedyParams(ctypes.Structure):
    _fields_ = [("best_of", c_int)]


class WhisperBeamSearchParams(ctypes.Structure):
    _fields_ = [
        ("beam_size", c_int),
        ("patience", c_float),
    ]


class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", c_int),
        ("n_threads", c_int),
        ("n_max_text_ctx", c_int),
        ("offset_ms", c_int),
        ("duration_ms", c_int),
        ("translate", c_bool),
        ("no_context", c_bool),
        ("no_timestamps", c_bool),
        ("single_segment", c_bool),
        ("print_special", c_bool),
        ("print_progress", c_bool),
        ("print_realtime", c_bool),
        ("print_timestamps", c_bool),
        ("token_timestamps", c_bool),
        ("thold_pt", c_float),
        ("thold_ptsum", c_float),
        ("max_len", c_int),
        ("split_on_word", c_bool),
        ("max_tokens", c_int),
        ("debug_mode", c_bool),
        ("audio_ctx", c_int),
        ("tdrz_enable", c_bool),
        ("suppress_regex", c_char_p),
        ("initial_prompt", c_char_p),
        ("prompt_tokens", POINTER(c_int32)),
        ("prompt_n_tokens", c_int),
        ("language", c_char_p),
        ("detect_language", c_bool),
        ("suppress_blank", c_bool),
        ("suppress_nst", c_bool),
        ("temperature", c_float),
        ("max_initial_ts", c_float),
        ("length_penalty", c_float),
        ("temperature_inc", c_float),
        ("entropy_thold", c_float),
        ("logprob_thold", c_float),
        ("no_speech_thold", c_float),
        ("greedy", WhisperGreedyParams),
        ("beam_search", WhisperBeamSearchParams),
        ("new_segment_callback", c_void_p),
        ("new_segment_callback_user_data", c_void_p),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("encoder_begin_callback", c_void_p),
        ("encoder_begin_callback_user_data", c_void_p),
        ("abort_callback", c_void_p),
        ("abort_callback_user_data", c_void_p),
        ("logits_filter_callback", c_void_p),
        ("logits_filter_callback_user_data", c_void_p),
        ("grammar_rules", POINTER(POINTER(WhisperGrammarElement))),
        ("n_grammar_rules", c_size_t),
        ("i_start_rule", c_size_t),
        ("grammar_penalty", c_float),
        ("vad", c_bool),
        ("vad_model_path", c_char_p),
        ("vad_params", WhisperVadParams),
    ]


class WhisperLib:
    """Wrapper around the whisper.cpp shared library."""

    def __init__(self, library_path: Path) -> None:
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._configure()

    def _configure(self) -> None:
        lib = self.lib
        lib.whisper_context_default_params.restype = WhisperContextParams
        lib.whisper_context_default_params.argtypes = []

        lib.whisper_full_default_params.restype = WhisperFullParams
        lib.whisper_full_default_params.argtypes = [c_int]

        lib.whisper_init_from_file_with_params.restype = c_void_p
        lib.whisper_init_from_file_with_params.argtypes = [c_char_p, WhisperContextParams]

        lib.whisper_free.restype = None
        lib.whisper_free.argtypes = [c_void_p]

        lib.whisper_full.restype = c_int
        lib.whisper_full.argtypes = [c_void_p, WhisperFullParams, POINTER(c_float), c_int]

        lib.whisper_full_n_segments.restype = c_int
        lib.whisper_full_n_segments.argtypes = [c_void_p]

        lib.whisper_full_get_segment_text.restype = c_char_p
        lib.whisper_full_get_segment_text.argtypes = [c_void_p, c_int]

        lib.whisper_full_get_segment_t0.restype = c_int64
        lib.whisper_full_get_segment_t0.argtypes = [c_void_p, c_int]

        lib.whisper_full_get_segment_t1.restype = c_int64
        lib.whisper_full_get_segment_t1.argtypes = [c_void_p, c_int]

        lib.whisper_full_n_tokens.restype = c_int
        lib.whisper_full_n_tokens.argtypes = [c_void_p, c_int]

        lib.whisper_full_get_token_id.restype = c_int
        lib.whisper_full_get_token_id.argtypes = [c_void_p, c_int, c_int]

        lib.whisper_token_to_str.restype = c_char_p
        lib.whisper_token_to_str.argtypes = [c_void_p, c_int]

        lib.whisper_n_text_ctx.restype = c_int
        lib.whisper_n_text_ctx.argtypes = [c_void_p]


@dataclass(slots=True)
class Segment:
    index: int
    start_ms: int
    end_ms: int
    text: str


class WhisperContext:
    def __init__(
        self,
        model_path: Path,
        library_path: Path,
        *,
        use_gpu: bool = True,
        flash_attn: bool = True,
    ) -> None:
        self.model_path = model_path
        self._lib = WhisperLib(library_path)
        self._language_bytes: bytes | None = None
        params = self._lib.lib.whisper_context_default_params()
        params.use_gpu = use_gpu
        params.flash_attn = flash_attn
        ctx = self._lib.lib.whisper_init_from_file_with_params(
            str(model_path).encode("utf-8"),
            params,
        )
        if not ctx:
            raise WhisperError(f"Failed to initialize whisper context from {model_path}")
        self._ctx = ctx
        self._n_text_ctx = self._lib.lib.whisper_n_text_ctx(self._ctx)

    @property
    def lib(self) -> WhisperLib:
        return self._lib

    def __enter__(self) -> "WhisperContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.lib.whisper_free(self._ctx)
            self._ctx = None  # type: ignore

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        n_threads: int,
        strategy: int,
        beam_size: int | None,
        translate: bool,
        no_context: bool,
        no_timestamps: bool,
        language: str | None,
        temperature: float,
        prompt_tokens: Sequence[int] | None = None,
        max_tokens: int = 0,
        audio_ctx: int = 0,
    ) -> tuple[list[Segment], list[int]]:
        if getattr(self, "_ctx", None) is None:
            raise WhisperError("Context already closed")
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        params = self._lib.lib.whisper_full_default_params(strategy)
        params.n_threads = int(max(1, n_threads))
        params.translate = translate
        params.no_context = no_context
        params.no_timestamps = no_timestamps
        params.temperature = float(temperature)
        params.audio_ctx = int(audio_ctx)
        params.max_tokens = int(max(0, max_tokens))
        params.print_progress = False
        params.print_realtime = False
        params.print_timestamps = not no_timestamps
        params.single_segment = False
        params.beam_search.beam_size = beam_size or params.beam_search.beam_size
        if language and language not in {"auto", ""}:
            lang_bytes = language.encode("utf-8")
            self._language_bytes = lang_bytes
            params.language = lang_bytes
            params.detect_language = False
        else:
            params.language = None
            params.detect_language = True
        token_buffer = None
        if prompt_tokens and not no_context:
            token_limit = max(0, self._n_text_ctx // 2)
            cropped = list(prompt_tokens)[-token_limit:] if token_limit else []
            if cropped:
                token_buffer = (c_int32 * len(cropped))(*cropped)
                params.prompt_tokens = ctypes.cast(token_buffer, POINTER(c_int32))
                params.prompt_n_tokens = len(cropped)
            else:
                params.prompt_tokens = ctypes.cast(None, POINTER(c_int32))
                params.prompt_n_tokens = 0
        else:
            params.prompt_tokens = ctypes.cast(None, POINTER(c_int32))
            params.prompt_n_tokens = 0
        result = self._lib.lib.whisper_full(
            self._ctx,
            params,
            samples.ctypes.data_as(POINTER(c_float)),
            samples.size,
        )
        if result != 0:
            raise WhisperError(f"whisper_full returned error code {result}")
        segments = self._collect_segments()
        tokens = self._collect_tokens(no_context)
        return segments, tokens

    def _collect_segments(self) -> list[Segment]:
        n_segments = self._lib.lib.whisper_full_n_segments(self._ctx)
        out: list[Segment] = []
        for i in range(n_segments):
            text_ptr = self._lib.lib.whisper_full_get_segment_text(self._ctx, i)
            text = text_ptr.decode("utf-8") if text_ptr else ""
            t0 = self._lib.lib.whisper_full_get_segment_t0(self._ctx, i)
            t1 = self._lib.lib.whisper_full_get_segment_t1(self._ctx, i)
            out.append(Segment(index=i, start_ms=int(t0), end_ms=int(t1), text=text.strip()))
        return out

    def _collect_tokens(self, no_context: bool) -> list[int]:
        if no_context:
            return []
        prompt_limit = max(0, self._n_text_ctx // 2)
        tokens: list[int] = []
        n_segments = self._lib.lib.whisper_full_n_segments(self._ctx)
        for i in range(n_segments):
            token_count = self._lib.lib.whisper_full_n_tokens(self._ctx, i)
            for j in range(token_count):
                tokens.append(self._lib.lib.whisper_full_get_token_id(self._ctx, i, j))
        if len(tokens) > prompt_limit:
            tokens = tokens[-prompt_limit:]
        return tokens


