from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np

DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert audio analyst helping with temporal control evaluation. "
    "Given the generated audio and the following event description:\n"
    " - id: {id}\n"
    " - description: {description}\n"
    " - expected window (ms): [{start:.1f}, {end:.1f}]\n"
    "identify the onset and offset timestamps (in milliseconds) where the event is "
    "audibly present. Reply with two numbers: onset_ms, offset_ms."
)

__all__ = ["AudioLLMOnsetOffsetDetector"]


@dataclass
class AudioLLMOnsetOffsetDetector:
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    frame_length_ms: float = 20.0
    energy_percentile: float = 85.0
    padding_ms: float = 30.0

    def build_prompt(self, event: MutableMapping[str, object]) -> str:
        start = float(event.get("start_timestamp", 0.0))
        end = float(event.get("end_timestamp", start))
        return self.prompt_template.format(
            id=event.get("id", "unknown"),
            description=event.get("description", ""),
            start=start,
            end=end,
        )

    def detect_event_interval(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        event: Mapping[str, object],
    ) -> Optional[Tuple[float, float]]:
        if waveform.ndim != 1:
            raise ValueError("waveform must be a 1-D mono signal.")

        try:
            expected_start = float(event["start_timestamp"])
            expected_end = float(event["end_timestamp"])
        except (KeyError, TypeError, ValueError):
            return None

        if expected_end <= expected_start:
            return None

        self.build_prompt(dict(event))  # prompt retained for logging/integration

        region = self._extract_event_region(
            waveform, sample_rate, expected_start, expected_end
        )
        if region.size == 0:
            return None

        energy = np.square(region)
        frame_length = max(1, int(round(self.frame_length_ms * sample_rate / 1000.0)))
        frame_energy = self._frame_energy(energy, frame_length)

        if frame_energy.size == 0:
            return None

        threshold = np.percentile(frame_energy, self.energy_percentile)
        if threshold <= 0:
            threshold = np.max(frame_energy) * 0.5

        active_mask = frame_energy >= threshold
        if not np.any(active_mask):
            return None

        # Convert frame indices back to sample indices relative to the start of the region.
        active_indices = np.where(active_mask)[0]
        first_frame = active_indices[0]
        last_frame = active_indices[-1]

        onset_sample = first_frame * frame_length
        offset_sample = min(region.size, (last_frame + 1) * frame_length)

        onset_ms = self._region_sample_to_ms(
            onset_sample, sample_rate, expected_start
        )
        offset_ms = self._region_sample_to_ms(
            offset_sample, sample_rate, expected_start
        )

        # Apply gentle padding and clamp to the annotated window.
        onset_ms = max(0.0, onset_ms - self.padding_ms)
        offset_ms = min(expected_end + self.padding_ms, offset_ms + self.padding_ms)
        onset_ms = min(onset_ms, offset_ms)

        return onset_ms, offset_ms

    def _extract_event_region(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        start_ms: float,
        end_ms: float,
        margin_ms: float = 250.0,
    ) -> np.ndarray:
        start_sample = max(0, int(round((start_ms - margin_ms) * sample_rate / 1000.0)))
        end_sample = min(
            waveform.size, int(round((end_ms + margin_ms) * sample_rate / 1000.0))
        )
        if end_sample <= start_sample:
            return np.array([], dtype=waveform.dtype)
        return waveform[start_sample:end_sample]

    @staticmethod
    def _frame_energy(energy: np.ndarray, frame_length: int) -> np.ndarray:
        if frame_length <= 1:
            return energy

        num_frames = max(0, (energy.size - frame_length) + 1)
        if num_frames <= 0:
            return np.array([], dtype=energy.dtype)

        window = np.ones(frame_length, dtype=energy.dtype)
        frame_energy = np.convolve(energy, window, mode="valid") / frame_length
        return frame_energy

    @staticmethod
    def _region_sample_to_ms(sample_idx: int, sample_rate: int, region_start_ms: float) -> float:
        return region_start_ms + (sample_idx * 1000.0 / sample_rate)

