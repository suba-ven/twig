from dataclasses import dataclass
from .common import ExperimentConfig


@dataclass(frozen=True)
class SIDiffusionConfig(ExperimentConfig):
    history: int = 14
    forecast_horizon: int = 14
    rollout_steps: int = 100
    channels: int = 1
    target_parameters: int = 70_224
    tc_wavelet_bands: tuple[int, ...] = (5,)
    tc_seeds: tuple[int, ...] = (42, 43, 45)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5


CONFIG = SIDiffusionConfig()
