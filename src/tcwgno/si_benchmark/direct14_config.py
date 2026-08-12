from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Direct14Config:
    """Configuration for the direct 14-input / 14-output SI experiment."""

    history: int = 14
    forecast_horizon: int = 14
    rollout_steps: int = 100

    # Training settings. The original reference uses Adam and RMSE.
    epochs: int = 30
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5

    # Scenario-level 76/12/12 split: 19 / 3 / 3 full trajectories.
    n_train_scenarios: int = 19
    n_val_scenarios: int = 3
    n_test_scenarios: int = 3
    n_scenarios: int = 25
    full_timesteps: int = 364
    n_nodes: int = 400
    n_raw_channels: int = 2

    # Architecture and SA temporal-scale settings.
    wavelet_scales: int = 4
    spectral_pos_modes: int = 16
    coordinate_frequencies: int = 6
    scale_anchor_weight: float = 1e-4
    alpha_min: float = 0.15
    alpha_max: float = 0.85
    min_cumulative_lag: float = 0.35
    max_lag_fraction: float = 0.65

    @property
    def all_scenario_count(self) -> int:
        return (
            self.n_train_scenarios
            + self.n_val_scenarios
            + self.n_test_scenarios
        )


@dataclass(frozen=True)
class Direct14Paths:
    data: Path
    edges: Path
    output: Path
    models_3d_dir: Path
    coordinate_cache: Path
    coordinate_csv: Path
