"""
Latency Coding.

High-activation features fire early; low-activation features fire late
or not at all. Encodes information in the *timing* of the first spike.

Reference:
    Thorpe, S., Delorme, A., & Van Rullen, R. (2001). Spike-based
    strategies for rapid processing. Neural Networks.
"""

import numpy as np
from .base import BaseSpikeEncoder


class LatencyEncoder(BaseSpikeEncoder):
    """
    Latency (Time-to-First-Spike) Coding.

    The first spike time t_i for feature i is:
        t_i = T · exp(-τ · v_i)   for v_i ∈ [0, 1]

    A high value fires at t ≈ 0; a near-zero value fires near t ≈ T.
    Values below a threshold produce no spike.
    """

    def __init__(
        self,
        time_steps: int = 50,
        tau: float = 5.0,
        normalize: bool = True,
        threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(time_steps=time_steps, **kwargs)
        self.name = "latency"
        self.tau = tau
        self.normalize = normalize
        self.threshold = threshold
        self._params.update({"tau": tau, "normalize": normalize, "threshold": threshold})

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (batch, features)

        Returns:
            spikes: (batch, time_steps, features)  — at most one spike per feature
        """
        x = self._normalize(embeddings.astype(np.float32))

        batch, features = x.shape
        spikes = np.zeros((batch, self.time_steps, features), dtype=np.float32)

        # Compute first spike time: t = T * exp(-tau * v)
        # High v → small t (early fire); low v → large t (late fire)
        spike_times = (self.time_steps * np.exp(-self.tau * x)).astype(int)
        spike_times = np.clip(spike_times, 0, self.time_steps - 1)

        # Mask features below threshold (they produce no spike)
        no_spike_mask = x < self.threshold

        # Vectorized: use advanced indexing to set spikes
        valid_mask = ~no_spike_mask
        batch_indices, feature_indices = np.where(valid_mask)
        time_indices = spike_times[batch_indices, feature_indices]
        spikes[batch_indices, time_indices, feature_indices] = 1.0

        return spikes

    def decode(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Time-to-first-spike decode.

        Latency coding puts information in *when* a feature fires, not
        how often (each feature fires at most once). Mean-pooling over
        time (the base-class default) would collapse every fired
        feature to the same ~1/T value regardless of its original
        magnitude — this override inverts the encoding formula
        (t = T * exp(-tau * v)) using the observed spike time instead.

        Args:
            spike_trains: (batch, time_steps, features)

        Returns:
            decoded: (batch, features) float32 estimate in [0, 1];
                     features with no spike decode to 0.0 (lowest value)
        """
        batch, T, features = spike_trains.shape
        has_spike = spike_trains.max(axis=1) > 0  # (batch, features)

        # First (only) spike time per feature; argmax finds the first
        # occurrence of the max value (1.0) along the time axis.
        spike_time = np.argmax(spike_trains, axis=1).astype(np.float32)

        t_norm = np.clip(spike_time / max(T - 1, 1), 1e-6, 1.0)
        v_hat = -np.log(t_norm) / self.tau
        v_hat = np.clip(v_hat, 0.0, 1.0)
        v_hat = np.where(has_spike, v_hat, 0.0)
        return v_hat.astype(np.float32)
