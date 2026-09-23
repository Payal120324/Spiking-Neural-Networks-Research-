"""
Binary Threshold Encoding.

The simplest spike encoding: each feature fires throughout all time steps
if its value exceeds a threshold, otherwise it remains silent.
Threshold is computed as the Nth percentile of the embedding values.

Reference:
    Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit
    recognition using spike-timing-dependent plasticity. Frontiers in
    computational neuroscience.
"""

import numpy as np
from .base import BaseSpikeEncoder


class BinaryThresholdEncoder(BaseSpikeEncoder):
    """
    Binary Threshold Coding.

    s_{t,i} = 1   if  v_i > threshold
    s_{t,i} = 0   otherwise

    The threshold can be computed using multiple strategies:
    - percentile: Nth percentile of embedding values
    - adaptive: mean + alpha * std (adaptive threshold)
    - constant: fixed threshold value
    """

    def __init__(
        self,
        time_steps: int = 50,
        percentile: float = 50.0,
        constant_threshold: float = None,
        adaptive_alpha: float = None,
        **kwargs,
    ):
        super().__init__(time_steps=time_steps, **kwargs)
        self.name = "binary_threshold"
        self.percentile = percentile
        self.constant_threshold = constant_threshold
        self.adaptive_alpha = adaptive_alpha
        self._params.update(
            {
                "percentile": percentile, 
                "constant_threshold": constant_threshold,
                "adaptive_alpha": adaptive_alpha
            }
        )

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (batch, features)

        Returns:
            spikes: (batch, time_steps, features)
        """
        x = self._normalize(embeddings.astype(np.float32))

        if self.constant_threshold is not None:
            threshold = float(self.constant_threshold)
        elif self.adaptive_alpha is not None:
            # Adaptive threshold: mean + alpha * std
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True)
            threshold = mean + self.adaptive_alpha * std
        else:
            # Per-sample percentile threshold
            threshold = np.percentile(x, self.percentile, axis=-1, keepdims=True)

        # Binary mask: 1 if above threshold
        active = (x > threshold).astype(np.float32)  # (batch, features)

        # Repeat across all time steps
        spikes = np.repeat(active[:, np.newaxis, :], self.time_steps, axis=1)
        return spikes
