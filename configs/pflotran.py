from dataclasses import dataclass
from .common import ExperimentConfig


@dataclass(frozen=True)
class PFLOTRANConfig(ExperimentConfig):
    history: int = 10
    forecast_horizon: int = 10
    rollout_steps: int = 65
    channels: int = 2
    laplacian_modes: int = 128
    tc_wavelet_bands: tuple[int, ...] = (5, 7)


CONFIG = PFLOTRANConfig()
