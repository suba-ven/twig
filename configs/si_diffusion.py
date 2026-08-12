from dataclasses import dataclass
from .common import ExperimentConfig


@dataclass(frozen=True)
class SIDiffusionConfig(ExperimentConfig):
    history: int = 14
    forecast_horizon: int = 14
    rollout_steps: int = 100
    channels: int = 1
    target_parameters: int = 70_224
    tc_wavelet_bands: tuple[int, ...] = (2, 6)


CONFIG = SIDiffusionConfig()
