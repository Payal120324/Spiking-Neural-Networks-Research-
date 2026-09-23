"""
Temporal Contrast Coding.

Converts embedding values into sequences where spikes mark
significant changes (transitions) across quantization levels.

Reference:
    Auge, D. et al. (2021). A survey of encoding techniques for
    signal processing in spiking neural networks. Neural Processing Letters.
"""

import numpy as np
from .base import BaseSpikeEncoder


class TemporalEncoder(BaseSpikeEncoder):
    """
    Temporal Contrast (Step-Forward) Coding.

    The embedding is quantized into n_levels discrete bins.
    At each time step the encoder advances through the quantization
    ladder; a spike is emitted whenever the current level changes
    relative to the previous step.

    For static embeddings (no temporal dimension), we simulate a
    temporal sequence by sweeping threshold levels over time steps:
        s_{t,i} = 1  if  ⌊v_i · n_levels⌋ == t   (for t < n_levels)
    i.e. feature i fires exactly once at the time step corresponding
    to its quantization bin.
    """

    def __init__(
        self,
        time_steps: int = 50,
        n_levels: int = 10,
        **kwargs,
    ):
        super().__init__(time_steps=time_steps, **kwargs)
        self.name = "temporal"
        self.n_levels = n_levels
        self._params.update({"n_levels": n_levels})

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (batch, features)

        Returns:
            spikes: (batch, time_steps, features)
        """
        x = self._normalize(embeddings.astype(np.float32))  # [0, 1]

        batch, features = x.shape
        spikes = np.zeros((batch, self.time_steps, features), dtype=np.float32)

        # Map value → bin index in [0, n_levels-1]
        bins = np.floor(x * self.n_levels).astype(int)
        bins = np.clip(bins, 0, self.n_levels - 1)

        # Map bin index → time step (stretch n_levels onto time_steps)
        time_indices = (bins * self.time_steps // self.n_levels).astype(int)
        time_indices = np.clip(time_indices, 0, self.time_steps - 1)

        # Vectorized: use advanced indexing to set spikes
        batch_indices, feature_indices = np.indices((batch, features))
        spikes[batch_indices, time_indices, feature_indices] = 1.0

        return spikes

    def decode(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Bin-index decode.

        Every feature fires exactly once at the time step corresponding
        to its quantization bin, so the mean firing rate over time
        (the base-class default) is a CONSTANT 1/T for every feature of
        every sample — zero variance, which is exactly why downstream
        Spearman/Kendall correlations were coming out NaN. This override
        recovers the quantization bin from the spike's time index and
        maps it back to an estimated normalized value.

        Args:
            spike_trains: (batch, time_steps, features)

        Returns:
            decoded: (batch, features) float32 estimate in [0, 1]
        """
        batch, T, features = spike_trains.shape
        spike_time = np.argmax(spike_trains, axis=1).astype(np.float32)  # (batch, features)
        bin_idx = spike_time * self.n_levels / T
        v_hat = bin_idx / self.n_levels
        return np.clip(v_hat, 0.0, 1.0).astype(np.float32)
