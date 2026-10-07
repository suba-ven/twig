from dataclasses import dataclass
from .common import ExperimentConfig


@dataclass(frozen=True)
class PFLOTRANConfig(ExperimentConfig):
    history: int = 10
    forecast_horizon: int = 10
    rollout_steps: int = 60
    channels: int = 2
    laplacian_modes: int = 128
    tc_wavelet_bands: tuple[int, ...] = (5,)
    epochs: int = 100
    target_parameters: int = 1_000_000
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    min_learning_rate: float = 1e-6
    split_seed: int = 0


CONFIG = PFLOTRANConfig()
