"""
Poisson Rate Coding.

Maps each embedding value to a firing rate and draws spike times
from a Poisson process. Higher activation → higher spike rate.
"""

import numpy as np
from .base import BaseSpikeEncoder


class PoissonRateEncoder(BaseSpikeEncoder):
    """
    Poisson Rate Coding.

    Each feature value v ∈ [0, 1] is treated as a firing probability
    per time step. Spikes are drawn independently at each step:

        s_t ~ Bernoulli(r(v))   where r(v) = v · (max_rate / T)

    Reference:
        Maass, W. (1997). Networks of spiking neurons: the third generation
        of neural network models. Neural networks.
    """

    def __init__(self, time_steps: int = 50, max_rate: float = 100.0, **kwargs):
        super().__init__(time_steps=time_steps, **kwargs)
        self.name = "poisson_rate"
        self.max_rate = max_rate
        self._params.update({"max_rate": max_rate})

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (batch, features)  float32, any range

        Returns:
            spikes: (batch, time_steps, features)  float32 ∈ {0,1}
        """
        x = self._normalize(embeddings.astype(np.float32))   # → [0, 1]

        # firing probability per step = x * (max_rate / (1000/dt))
        # With T time steps, p_fire = x * max_rate / (1000 / T)
        # Simplified: p_fire = x  (rate is proportional to x)
        p_fire = x  # shape (batch, features)

        rng = np.random.default_rng()
        uniform = rng.random(
            (p_fire.shape[0], self.time_steps, p_fire.shape[1]),
            dtype=np.float32,
        )

        # spike if uniform < firing probability
        spikes = (uniform < p_fire[:, np.newaxis, :]).astype(np.float32)
        return spikes
