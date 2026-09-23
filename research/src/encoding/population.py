"""
Population Coding.

Each scalar feature is represented by a population of n_neurons Gaussian
receptive fields (RFs) tiled across the input range. A feature value
activates neurons proportional to the proximity of their RF center,
then Poisson-spikes each neuron at its activation rate.

Reference:
    Bohte, S. M., Kok, J. N., & La Poutré, H. (2002). Error-backpropagation
    in temporally encoded networks of spiking neurons. Neurocomputing.
"""

import numpy as np
from .base import BaseSpikeEncoder


class PopulationEncoder(BaseSpikeEncoder):
    """
    Gaussian Population Coding.

    n_neurons Gaussian receptive fields are tiled uniformly over [0, 1].
    Output spike train shape is (batch, time_steps, features × n_neurons).
    """

    def __init__(
        self,
        time_steps: int = 50,
        n_neurons: int = 10,
        sigma: float = 0.5,
        **kwargs,
    ):
        super().__init__(time_steps=time_steps, **kwargs)
        self.name = "population"
        self.n_neurons = n_neurons
        self.sigma = sigma
        # Centers evenly spaced in [0, 1]
        self._centers = np.linspace(0.0, 1.0, n_neurons, dtype=np.float32)
        self._params.update({"n_neurons": n_neurons, "sigma": sigma})

    @property
    def output_dim_factor(self) -> int:
        """Each input feature maps to n_neurons output neurons."""
        return self.n_neurons

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (batch, features)

        Returns:
            spikes: (batch, time_steps, features * n_neurons)
        """
        x = self._normalize(embeddings.astype(np.float32))  # (batch, features)

        batch, features = x.shape

        # Gaussian activations: (batch, features, n_neurons)
        # activation[b, f, n] = exp(-0.5 * ((x[b,f] - center[n]) / sigma)^2)
        diff = x[:, :, np.newaxis] - self._centers[np.newaxis, np.newaxis, :]
        activations = np.exp(-0.5 * (diff / self.sigma) ** 2).astype(np.float32)

        # Reshape to (batch, features*n_neurons)
        pop_activations = activations.reshape(batch, features * self.n_neurons)

        # Poisson spike generation
        rng = np.random.default_rng()
        uniform = rng.random(
            (batch, self.time_steps, features * self.n_neurons),
            dtype=np.float32,
        )
        spikes = (uniform < pop_activations[:, np.newaxis, :]).astype(np.float32)
        return spikes

    def decode(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Population-vector decode.

        Reconstructs each original feature's normalized value as the
        activation-weighted average of its n_neurons Gaussian receptive
        field centers (standard population-vector decoding), rather than
        the base class's naive block-average which ignores the Gaussian
        receptive-field structure entirely.

        Args:
            spike_trains: (batch, time_steps, features * n_neurons)

        Returns:
            decoded: (batch, features) float32 estimate in [0, 1]
        """
        batch, T, expanded = spike_trains.shape
        features = expanded // self.n_neurons

        rates = spike_trains.mean(axis=1).reshape(batch, features, self.n_neurons)
        weight_sum = rates.sum(axis=2) + 1e-9
        centers = self._centers.reshape(1, 1, self.n_neurons)
        v_hat = (rates * centers).sum(axis=2) / weight_sum
        return v_hat.astype(np.float32)
