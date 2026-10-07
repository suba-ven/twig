from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("../data/airfoil/raw")
    artifact_dir: Path = Path("../data/airfoil/artifacts")
    output_dir: Path = Path("results/airfoil")
    history: int = 20
    forecast_horizon: int = 20
    channels: int = 4
    n_modes: int = 128
    target_parameters: int = 10_000_000
    epochs: int = 100
    runs_per_model: int = 3
    # Equal to the training length so every run completes all requested epochs.
    early_stopping_patience: int = 100
    windows_per_train_trajectory: int = 2
    windows_per_eval_trajectory: int = 4
    batch_size: int = 8
    # AdamW LR schedule: warm up from 1e-5 to 1e-4, then cosine decay to 1e-5.
    learning_rate: float = 1.0e-4
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_learning_rate: float = 2.0e-5
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    rollout_steps: int = 180
    seed: int = 2026


CONFIG = ExperimentConfig()

# Balanced by expected operator/message-passing cost, not simply model count.
SUITE_A = (
    "sa_twig_h10_k5",
    "plain_graph_wno_h10",
    "gat_h10",
    "gps_transformer_h10",
    "rnn_gnn_fusion_h10",
)
SUITE_B = (
    "sa_twig_h10_k7",
    "graph_fno_h10",
    "meshgraphnet_h10",
    "gatv2_h10",
    "rnn_h10",
)
ALL_MODELS = SUITE_A + SUITE_B
TC_MODELS = ("sa_twig_h10_k5", "sa_twig_h10_k7")
