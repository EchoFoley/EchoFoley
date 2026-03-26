from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

try:  # soundfile offers fast, dependency-light IO; optional for callers providing np.ndarray
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sf = None  # type: ignore

from .sounding_events import SoundingEvent


AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]
EventLike = Union[SoundingEvent, Mapping[str, object], MutableMapping[str, object]]

DEFAULT_VOLUME_KEYS: Tuple[str, ...] = (
    "volume",
    "volume_db",
    "loudness",
    "loudness_db",
    "depth",
    "depth_db",
)

EPS = 1e-12

__all__ = ["evaluate_depth_control"]


def evaluate_depth_control(
    audio: AudioLike,
    events: Sequence[EventLike],
    *,
    sample_rate: Optional[int] = None,
    target_volume_keys: Iterable[str] = DEFAULT_VOLUME_KEYS,
    loudness_floor_db: float = -80.0,
) -> float:

    waveform, sr = _resolve_audio(audio, sample_rate)
    if waveform.size == 0:
        return math.nan

    if waveform.ndim == 2:
        # Handle both (channels, samples) and (samples, channels) layouts
        if waveform.shape[0] < waveform.shape[1]:
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = np.mean(waveform, axis=1)
    waveform = waveform.astype(np.float32, copy=False)

    if np.issubdtype(waveform.dtype, np.integer):
        max_val = np.iinfo(waveform.dtype).max
        waveform = waveform / max_val

    loudness_errors = []
    for raw_event in events:
        event_dict = _event_to_mapping(raw_event)
        try:
            start_ms = float(event_dict["start_timestamp"])
            end_ms = float(event_dict["end_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue

        loudness_gt = _extract_ground_truth_loudness(event_dict, target_volume_keys)
        if loudness_gt is None:
            continue

        start_sample = max(0, int(round(start_ms * sr / 1000.0)))
        end_sample = min(len(waveform), int(round(end_ms * sr / 1000.0)))
        if end_sample <= start_sample:
            continue

        segment = waveform[start_sample:end_sample]
        loudness_gen = _compute_loudness_db(segment, floor_db=loudness_floor_db)
        loudness_errors.append(abs(loudness_gen - loudness_gt))

    if not loudness_errors:
        return math.nan

    return float(np.mean(loudness_errors))


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


def _extract_ground_truth_loudness(
    event: Mapping[str, object],
    candidate_keys: Iterable[str],
) -> Optional[float]:
    # Direct lookup by ordered candidate keys
    for key in candidate_keys:
        if key in event and event[key] is not None:
            return _safe_float(event[key])
        attributes = event.get("attributes")
        if isinstance(attributes, Mapping) and key in attributes and attributes[key] is not None:
            return _safe_float(attributes[key])

    # Special case for structured volume annotation inside "volume"
    raw_volume = event.get("volume")
    if isinstance(raw_volume, Mapping):
        for key in candidate_keys:
            if key in raw_volume and raw_volume[key] is not None:
                return _safe_float(raw_volume[key])

    return _safe_float(raw_volume) if raw_volume is not None else None


def _compute_loudness_db(segment: np.ndarray, floor_db: float) -> float:
    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        return floor_db

    rms = float(np.sqrt(np.mean(np.square(segment))) + EPS)
    loudness_db = 20.0 * np.log10(rms + EPS)
    if floor_db is not None:
        loudness_db = max(loudness_db, float(floor_db))
    return loudness_db


def _to_numpy_array(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return array


def _safe_float(value: object) -> Optional[float]:
    try:
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return float(stripped)
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

