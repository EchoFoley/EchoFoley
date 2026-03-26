from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

__all__ = ["CLAPSimilarityScorer"]


@dataclass
class CLAPSimilarityScorer:
    """
    Lightweight CLAP-inspired similarity estimator.

    This implementation provides a deterministic, dependency-light approximation of the
    CLAP semantic similarity score. It is designed as a drop-in replacement that can be
    swapped with a true CLAP model when available.
    """

    embedding_dim: int = 16
    text_ngram: int = 3
    epsilon: float = 1e-8

    def compute_similarity(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        description: str,
    ) -> Optional[float]:
        """
        Compute the cosine similarity between audio and text embeddings.

        Parameters
        ----------
        waveform:
            1-D mono audio segment normalised to [-1, 1].
        sample_rate:
            Sampling rate corresponding to ``waveform``.
        description:
            Textual description associated with the audio segment.

        Returns
        -------
        Optional[float]
            Cosine similarity between the audio and text representations. Returns
            ``None`` when either embedding cannot be computed.
        """

        if waveform.ndim != 1 or waveform.size == 0:
            return None
        if not description:
            return None

        audio_emb = self._encode_audio(waveform, sample_rate)
        text_emb = self._encode_text(description)

        if audio_emb is None or text_emb is None:
            return None

        return self._cosine_similarity(audio_emb, text_emb)

    # --------------------------------------------------------------------- Encoders ---
    def _encode_audio(self, waveform: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        energy = np.square(waveform)
        if energy.size == 0:
            return None

        features = [
            float(np.mean(waveform)),
            float(np.std(waveform)),
            float(np.max(waveform)),
            float(np.min(waveform)),
            float(np.median(waveform)),
            float(np.percentile(waveform, 90)),
            float(np.percentile(waveform, 10)),
            float(np.mean(np.abs(np.diff(waveform)))),
            float(np.mean(energy)),
            float(np.std(energy)),
            float(np.percentile(energy, 90)),
            float(np.percentile(energy, 10)),
        ]

        zero_crossings = np.mean(np.abs(np.diff(np.signbit(waveform)).astype(np.float32)))
        features.append(float(zero_crossings))

        duration_sec = waveform.size / float(sample_rate)
        features.append(duration_sec)

        spectral_centroid = self._spectral_centroid(waveform, sample_rate)
        spectral_flatness = self._spectral_flatness(waveform)
        features.extend([spectral_centroid, spectral_flatness])

        feature_vec = np.array(features, dtype=np.float32)
        return self._project_to_dim(feature_vec, self.embedding_dim, seed="audio")

    def _encode_text(self, description: str) -> Optional[np.ndarray]:
        tokens = self._tokenise_text(description)
        if not tokens:
            return None

        counts = {token: tokens.count(token) for token in set(tokens)}
        sorted_items = sorted(counts.items(), key=lambda item: item[0])
        values = np.array([value for _, value in sorted_items], dtype=np.float32)

        return self._project_to_dim(values, self.embedding_dim, seed="text")

    # -------------------------------------------------------------------- Utilities ---
    def _spectral_centroid(self, waveform: np.ndarray, sample_rate: int) -> float:
        if waveform.size == 0:
            return 0.0
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(waveform.size, d=1.0 / sample_rate)
        if np.sum(spectrum) <= self.epsilon:
            return 0.0
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + self.epsilon)
        return float(centroid / (sample_rate / 2.0 + self.epsilon))

    def _spectral_flatness(self, waveform: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(waveform)) + self.epsilon
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        return float(geo_mean / (arith_mean + self.epsilon))

    def _tokenise_text(self, description: str) -> Iterable[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in description)
        tokens = cleaned.split()
        if self.text_ngram <= 1:
            return tokens
        ngrams = []
        for token in tokens:
            if len(token) < self.text_ngram:
                ngrams.append(token)
            else:
                ngrams.extend(
                    token[i : i + self.text_ngram] for i in range(len(token) - self.text_ngram + 1)
                )
        return ngrams

    def _project_to_dim(self, vector: np.ndarray, dim: int, seed: str) -> np.ndarray:
        if vector.size == 0:
            return np.zeros(dim, dtype=np.float32)

        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        rng_seed = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(rng_seed)
        projection = rng.normal(size=(vector.size, dim)).astype(np.float32)

        projected = vector.astype(np.float32, copy=False) @ projection
        norm = np.linalg.norm(projected) + self.epsilon
        return projected / norm

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        denominator = (np.linalg.norm(a) * np.linalg.norm(b)) + self.epsilon
        return float(np.dot(a, b) / denominator)

