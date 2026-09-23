"""
Base class for spike encoders.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseSpikeEncoder(ABC):
    """
    Abstract base class for all spike encoding methods.
    All encoders receive dense continuous-valued embeddings and return
    binary spike trains of shape (time_steps, n_features).
    """

    def __init__(self, time_steps: int = 50, **kwargs):
        self.time_steps = time_steps
        self.name = "base"
        self._params: dict = {"time_steps": time_steps}
        self._global_min = None
        self._global_max = None

    @abstractmethod
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode a batch of embeddings into spike trains.

        Args:
            embeddings: ndarray of shape (batch_size, embedding_dim)

        Returns:
            spike_trains: ndarray of shape (batch_size, time_steps, embedding_dim)
                         dtype float32, values in {0, 1}
        """
        ...

    def encode_single(self, embedding: np.ndarray) -> np.ndarray:
        """Encode a single embedding vector."""
        return self.encode(embedding[np.newaxis])[0]

    def get_params(self) -> dict:
        """Return encoder hyperparameters."""
        return dict(self._params)

    def decode(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Reconstruct an estimate of the (per-sample min-max normalized,
        [0, 1]-scaled) embedding this encoder would have produced these
        spikes from.

        Default implementation: mean firing rate over time. This is a
        valid decoder for RATE-based codes (e.g. Poisson rate coding,
        binary threshold coding), where information is carried in *how
        often* a feature spikes.

        Encoders whose information lives in spike *timing* rather than
        spike *rate* (latency coding, temporal contrast coding) MUST
        override this method — mean-pooling over time destroys timing
        information and produces a degenerate (near-constant) decode.

        Args:
            spike_trains: (batch, time_steps, features) binary spikes

        Returns:
            decoded: (batch, features) float32 estimate in [0, 1]
        """
        return spike_trains.mean(axis=1)

    def firing_rate(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Compute average firing rate per feature dimension.

        Args:
            spike_trains: (batch, time_steps, features)

        Returns:
            rates: (batch, features) — mean spikes per time step
        """
        return spike_trains.mean(axis=1)

    def spike_count(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Count total spikes per sample.

        Args:
            spike_trains: (batch, time_steps, features)

        Returns:
            counts: (batch,)
        """
        return spike_trains.sum(axis=(1, 2))

    def sparsity(self, spike_trains: np.ndarray) -> float:
        """
        Fraction of zero entries in spike trains (higher = more sparse).
        """
        return float(1.0 - spike_trains.mean())

    def fit_normalize(self, embeddings: np.ndarray) -> None:
        """
        Compute per-feature (dataset-level) min/max from a reference set
        (the training split) and store them for reuse by _normalize.

        [FIX, 2026-09]: the original _normalize computed min/max PER SAMPLE
        (axis=-1, across the 768 embedding dimensions of a single sample).
        We confirmed empirically that pretrained transformer embeddings
        have a small number of "rogue" outlier dimensions with anomalously
        large magnitude, present in almost every sample -- for this
        project's CaseHOLD training embeddings, one dimension (205) is the
        per-sample max for 75.4% of samples, and another (308) is the
        per-sample min for 75.1% of samples. Per-sample min-max normalization
        is therefore effectively re-scaled by the SAME two dimensions for
        most samples, squashing all other (actually informative) dimensions
        into a narrow, nearly sample-invariant band. This destroys
        information for VALUE-sensitive encodings (latency, temporal, which
        map absolute normalized value to spike time/bin) while leaving
        RANK-sensitive encodings (binary_threshold's percentile cut is
        invariant to monotonic per-sample rescaling) comparatively unaffected.
        Call this once on the training split before encoding any split, so
        all splits are normalized consistently and without test-set leakage.
        """
        mins = embeddings.min(axis=0, keepdims=True)
        maxs = embeddings.max(axis=0, keepdims=True)
        self._global_min = mins
        self._global_max = maxs

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize to [0, 1]. Uses global (fit_normalize) per-feature stats
        if available; otherwise falls back to the original per-sample
        min-max behaviour for backward compatibility with any caller that
        does not call fit_normalize first.
        """
        if self._global_min is not None and self._global_max is not None:
            denom = self._global_max - self._global_min
            denom = np.where(denom == 0, 1.0, denom)
            out = (x - self._global_min) / denom
            return np.clip(out, 0.0, 1.0)
        # Fallback: original per-sample min-max (pre-fix behaviour)
        mins = x.min(axis=-1, keepdims=True)
        maxs = x.max(axis=-1, keepdims=True)
        denom = maxs - mins
        denom = np.where(denom == 0, 1.0, denom)
        return (x - mins) / denom