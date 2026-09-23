from .base import BaseSpikeEncoder
from .poisson import PoissonRateEncoder
from .latency import LatencyEncoder
from .temporal import TemporalEncoder
from .population import PopulationEncoder
from .binary import BinaryThresholdEncoder

ENCODERS = {
    "poisson_rate": PoissonRateEncoder,
    "latency": LatencyEncoder,
    "temporal": TemporalEncoder,
    "population": PopulationEncoder,
    "binary_threshold": BinaryThresholdEncoder,
}

__all__ = [
    "BaseSpikeEncoder",
    "PoissonRateEncoder",
    "LatencyEncoder",
    "TemporalEncoder",
    "PopulationEncoder",
    "BinaryThresholdEncoder",
    "ENCODERS",
]
