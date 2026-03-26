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
from .utils.CLAP.main import CLAPSimilarityScorer

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]
EventLike = Union[SoundingEvent, Mapping[str, object], MutableMapping[str, object]]

__all__ = ["evaluate_timbre_control"]


def evaluate_timbre_control(
    audio: AudioLike,
    events: Sequence[EventLike],
    *,
    sample_rate: Optional[int] = None,
    scorer: Optional[CLAPSimilarityScorer] = None,
) -> float:

    waveform, sr = _resolve_audio(audio, sample_rate)
    if waveform.size == 0:
        return math.nan

    waveform = _ensure_mono_float32(waveform)
    scorer = scorer or CLAPSimilarityScorer()

    similarities = []
    for raw_event in events:
        event_dict = _event_to_mapping(raw_event)
        try:
            start_ms = float(event_dict["start_timestamp"])
            end_ms = float(event_dict["end_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue

        if end_ms <= start_ms:
            continue

        description = str(event_dict.get("description", "")).strip()
        attributes = event_dict.get("attributes")
        if not description and isinstance(attributes, Mapping):
            description = str(attributes.get("description", "")).strip()
        if not description:
            continue

        start_sample = max(0, int(round(start_ms * sr / 1000.0)))
        end_sample = min(len(waveform), int(round(end_ms * sr / 1000.0)))
        if end_sample <= start_sample:
            continue

        segment = waveform[start_sample:end_sample]
        similarity = scorer.compute_similarity(segment, sr, description)
        if similarity is None or math.isnan(similarity):
            continue
        similarities.append(float(similarity))

    if not similarities:
        return math.nan

    return float(np.mean(similarities))


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


def _to_numpy_array(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return array

