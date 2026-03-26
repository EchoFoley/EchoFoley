from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

try:
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sf = None  # type: ignore

from .sounding_events import SoundingEvent
from .utils.audioLLMs.main import AudioLLMOnsetOffsetDetector

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]
EventLike = Union[SoundingEvent, Mapping[str, object], MutableMapping[str, object]]

EPS = 1e-12

__all__ = ["evaluate_temporal_control"]


def evaluate_temporal_control(
    audio: AudioLike,
    events: Sequence[EventLike],
    *,
    sample_rate: Optional[int] = None,
    detector: Optional[AudioLLMOnsetOffsetDetector] = None,
    min_duration_ms: float = 1.0,
) -> float:

    waveform, sr = _resolve_audio(audio, sample_rate)
    if waveform.size == 0:
        return math.nan

    waveform = _ensure_mono_float32(waveform)
    detector = detector or AudioLLMOnsetOffsetDetector()

    ious = []
    for raw_event in events:
        event_dict = _event_to_mapping(raw_event)
        try:
            gt_start = float(event_dict["start_timestamp"])
            gt_end = float(event_dict["end_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue

        if gt_end - gt_start < min_duration_ms:
            continue

        pred_interval = detector.detect_event_interval(waveform, sr, event_dict)
        if pred_interval is None:
            continue

        pred_start, pred_end = pred_interval
        if pred_end - pred_start < min_duration_ms:
            continue

        iou = _interval_iou((gt_start, gt_end), (pred_start, pred_end))
        if not math.isnan(iou):
            ious.append(iou)

    if not ious:
        return math.nan

    return float(np.mean(ious))


def _resolve_audio(audio: AudioLike, sample_rate: Optional[int]) -> Tuple[np.ndarray, int]:
    if isinstance(audio, tuple):
        waveform, sr = audio
        if sample_rate and sr != sample_rate:
            raise ValueError(
                f"Provided sample_rate ({sample_rate}) does not match audio tuple rate ({sr})."
            )
        return _to_numpy_array(waveform), int(sr)

    if isinstance(audio, str):
        if sf is None:
            raise ImportError("soundfile is required to load audio from file paths.")
        waveform, sr = cast(Tuple[np.ndarray, int], sf.read(audio))
        return _to_numpy_array(waveform), int(sr)

    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate must be provided when audio is a numpy array.")
        return _to_numpy_array(audio), int(sample_rate)

    raise TypeError(f"Unsupported audio input type: {type(audio)!r}")


def _ensure_mono_float32(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 2:
        if waveform.shape[0] < waveform.shape[1]:
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = np.mean(waveform, axis=1)

    waveform = waveform.astype(np.float32, copy=False)
    if np.issubdtype(waveform.dtype, np.integer):
        max_val = np.iinfo(waveform.dtype).max
        waveform = waveform / max_val

    return waveform


def _event_to_mapping(event: EventLike) -> MutableMapping[str, object]:
    if isinstance(event, MutableMapping):
        return event
    if isinstance(event, Mapping):
        return dict(event)
    if isinstance(event, SoundingEvent):
        return event.to_dict()
    if is_dataclass(event):
        return cast(MutableMapping[str, object], asdict(event))
    raise TypeError(f"Unsupported event type: {type(event)!r}")


def _interval_iou(interval_a: Tuple[float, float], interval_b: Tuple[float, float]) -> float:
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    if end_a <= start_a or end_b <= start_b:
        return math.nan

    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)

    if union <= 0:
        return math.nan

    return intersection / (union + EPS)


def _to_numpy_array(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return array

