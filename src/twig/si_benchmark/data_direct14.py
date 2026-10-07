from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .direct14_config import Direct14Config, Direct14Paths


COORDINATE_URL = (
    "https://raw.githubusercontent.com/"
    "Jostarndt/Synthetic_Datasets_for_Temporal_Graphs/"
    "31e95bb2bda7bc9223a2f246b970e96a3d09f657/"
    "pde_solver/SI_diffusion/germany_centers.csv"
)


class SIDiffusionDirectDataset(Dataset):
    """Scenario-safe windows for a direct H-input / F-output forecast."""

    def __init__(
        self,
        sequence: torch.Tensor,
        starts: Iterable[int],
        history: int,
        forecast_horizon: int,
    ):
        self.sequence = sequence
        self.starts = tuple(int(start) for start in starts)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        context = self.sequence[
            start : start + self.history
        ].unsqueeze(-1)
        future = self.sequence[
            start + self.history :
            start + self.history + self.forecast_horizon
        ].unsqueeze(-1)
        return context, future


@dataclass(frozen=True)
class PreparedDirect14Data:
    infected_full: torch.Tensor
    infected_scenarios: torch.Tensor
    train_dataset: SIDiffusionDirectDataset
    val_dataset: SIDiffusionDirectDataset
    test_dataset: SIDiffusionDirectDataset
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    node_xy: torch.Tensor
    fixed_pos_features: torch.Tensor
    train_scenario_ids: tuple[int, ...]
    val_scenario_ids: tuple[int, ...]
    test_scenario_ids: tuple[int, ...]
    rollout_scenario_ids: tuple[int, ...]
    available_timesteps: int

    def effective_rollout_steps(self, requested_steps: int, history: int) -> int:
        """Return the forecast length supported by each packaged trajectory."""
        return min(int(requested_steps), self.available_timesteps - int(history))


def _load_edges(path: Path, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        edge_table = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        edge_table = torch.load(path, map_location="cpu")

    if edge_table.ndim != 2:
        raise ValueError(
            f"Expected a rank-2 edge table, got {tuple(edge_table.shape)}."
        )
    if edge_table.shape[0] != 3:
        edge_table = edge_table.T.contiguous()
    if edge_table.shape[0] != 3:
        raise ValueError(
            f"Expected edge table shape (3, E), got {tuple(edge_table.shape)}."
        )

    src = edge_table[0].long()
    dst = edge_table[1].long()
    distance = edge_table[2].float()

    if not torch.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError("Edge distances must be finite and positive.")
    if (
        src.min() < 0
        or dst.min() < 0
        or src.max() >= n_nodes
        or dst.max() >= n_nodes
    ):
        raise ValueError("Edge indices are outside the graph node range.")

    return (
        torch.stack([src, dst], dim=0).contiguous(),
        distance.reciprocal().contiguous(),
    )


def _load_coordinates(
    paths: Direct14Paths,
    cfg: Direct14Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    if paths.coordinate_cache.is_file():
        try:
            payload = torch.load(
                paths.coordinate_cache,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(paths.coordinate_cache, map_location="cpu")

        xy = torch.as_tensor(payload["xy"], dtype=torch.float32)
        xy_raw = torch.as_tensor(
            payload.get("xy_raw", xy),
            dtype=torch.float32,
        )
        return xy, xy_raw

    if not paths.coordinate_csv.is_file():
        paths.coordinate_csv.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COORDINATE_URL, paths.coordinate_csv)

    table = pd.read_csv(paths.coordinate_csv)
    required_columns = {"NUTS_CODE", "x", "y"}

    if not required_columns.issubset(table.columns):
        raise ValueError("Unexpected NUTS-3 coordinate file columns.")
    if len(table) != cfg.n_nodes:
        raise ValueError(
            f"Expected {cfg.n_nodes} coordinates, found {len(table)}."
        )

    xy_raw = torch.tensor(
        table[["x", "y"]].to_numpy(dtype=np.float32)
    )
    xy = (
        xy_raw - xy_raw.mean(dim=0, keepdim=True)
    ) / xy_raw.std(dim=0, keepdim=True).clamp_min(1e-6)

    torch.save({"xy": xy, "xy_raw": xy_raw}, paths.coordinate_cache)
    return xy, xy_raw


def _laplacian_features(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    xy: torch.Tensor,
    cfg: Direct14Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency = torch.zeros(
        (cfg.n_nodes, cfg.n_nodes),
        dtype=torch.float32,
    )
    adjacency.index_put_(
        (edge_index[0], edge_index[1]),
        edge_weight,
        accumulate=True,
    )
    adjacency = 0.5 * (adjacency + adjacency.T)
    adjacency.fill_diagonal_(0.0)

    degree = adjacency.sum(dim=1).clamp_min(1e-12)
    degree_inverse_sqrt = torch.diag(degree.rsqrt())
    laplacian = (
        torch.eye(cfg.n_nodes)
        - degree_inverse_sqrt @ adjacency @ degree_inverse_sqrt
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    modes = eigenvectors[:, :64].contiguous()
    mode_values = eigenvalues[:64].contiguous()

    xy_standard = (
        xy - xy.mean(dim=0, keepdim=True)
    ) / xy.std(dim=0, keepdim=True).clamp_min(1e-6)

    frequencies = torch.logspace(
        0.0,
        math.log10(8.0),
        cfg.coordinate_frequencies,
    )
    scaled = (
        xy_standard.unsqueeze(-1)
        * frequencies.view(1, 1, -1)
        * (2.0 * math.pi)
    )
    coordinate_features = torch.cat(
        [
            xy_standard,
            torch.sin(scaled).reshape(cfg.n_nodes, -1),
            torch.cos(scaled).reshape(cfg.n_nodes, -1),
        ],
        dim=-1,
    )

    n_pos = min(cfg.spectral_pos_modes, cfg.n_nodes - 1)
    spectral_features = eigenvectors[:, 1 : n_pos + 1]
    spectral_features = (
        spectral_features
        - spectral_features.mean(dim=0, keepdim=True)
    ) / spectral_features.std(dim=0, keepdim=True).clamp_min(1e-6)

    fixed_pos = torch.cat(
        [coordinate_features, spectral_features],
        dim=-1,
    ).contiguous()

    return modes, mode_values, fixed_pos


def _starts_for_scenarios(
    scenario_ids: Iterable[int],
    cfg: Direct14Config,
) -> list[int]:
    starts: list[int] = []
    latest_local_start = (
        cfg.full_timesteps
        - cfg.history
        - cfg.forecast_horizon
    )

    if latest_local_start < 0:
        raise ValueError(
            "The history plus direct forecast horizon exceeds the "
            "available trajectory length."
        )

    for scenario_id in scenario_ids:
        base = int(scenario_id) * cfg.full_timesteps
        starts.extend(
            range(base, base + latest_local_start + 1)
        )

    return starts


def prepare_direct14_data(
    paths: Direct14Paths,
    cfg: Direct14Config,
) -> tuple[PreparedDirect14Data, torch.Tensor, torch.Tensor]:
    if not paths.data.is_file():
        raise FileNotFoundError(f"Missing SI data file: {paths.data}")
    if not paths.edges.is_file():
        raise FileNotFoundError(f"Missing graph edge file: {paths.edges}")

    if cfg.all_scenario_count != cfg.n_scenarios:
        raise ValueError(
            "Scenario split must cover all scenarios exactly: "
            f"{cfg.n_train_scenarios} + {cfg.n_val_scenarios} + "
            f"{cfg.n_test_scenarios} != {cfg.n_scenarios}."
        )

    if paths.data.suffix.lower() == ".npy":
        raw = torch.from_numpy(
            np.load(paths.data).astype(np.float32)
        ).contiguous()
    else:
        try:
            payload = torch.load(
                paths.data,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(paths.data, map_location="cpu")
        raw = payload.get("data") if isinstance(payload, dict) else payload
        raw = torch.as_tensor(raw, dtype=torch.float32).contiguous()

    full_shape = (
        cfg.n_scenarios * cfg.full_timesteps,
        cfg.n_nodes,
        cfg.n_raw_channels,
    )
    first_100_shape = (
        cfg.n_scenarios,
        100,
        cfg.n_nodes,
        cfg.n_raw_channels,
    )
    if tuple(raw.shape) == full_shape:
        scenarios = raw.reshape(
            cfg.n_scenarios,
            cfg.full_timesteps,
            cfg.n_nodes,
            cfg.n_raw_channels,
        )
    elif tuple(raw.shape) == first_100_shape:
        scenarios = raw
    else:
        raise ValueError(
            f"Expected full SI data shape {full_shape} or packaged "
            f"first-100 shape {first_100_shape}, got {tuple(raw.shape)}."
        )

    available_timesteps = int(scenarios.shape[1])
    if cfg.history + cfg.forecast_horizon > available_timesteps:
        raise ValueError("History plus forecast horizon exceeds the packaged trajectory length.")

    raw = scenarios.reshape(
        cfg.n_scenarios * available_timesteps,
        cfg.n_nodes,
        cfg.n_raw_channels,
    ).contiguous()

    # Column 1 is the infected state. Keep it on its raw data scale.
    infected = raw[:, :, 1].contiguous()

    data_cfg = cfg if available_timesteps == cfg.full_timesteps else replace(
        cfg, full_timesteps=available_timesteps
    )

    train_scenario_ids = tuple(range(0, cfg.n_train_scenarios))
    val_start = cfg.n_train_scenarios
    val_scenario_ids = tuple(
        range(val_start, val_start + cfg.n_val_scenarios)
    )
    test_start = val_start + cfg.n_val_scenarios
    test_scenario_ids = tuple(
        range(test_start, test_start + cfg.n_test_scenarios)
    )

    edge_index, edge_weight = _load_edges(paths.edges, cfg.n_nodes)
    xy, _ = _load_coordinates(paths, cfg)
    modes, mode_values, fixed_pos = _laplacian_features(
        edge_index,
        edge_weight,
        xy,
        cfg,
    )

    prepared = PreparedDirect14Data(
        infected_full=infected,
        infected_scenarios=infected.reshape(
            cfg.n_scenarios,
            available_timesteps,
            cfg.n_nodes,
            1,
        ).contiguous(),
        train_dataset=SIDiffusionDirectDataset(
            infected,
            _starts_for_scenarios(train_scenario_ids, data_cfg),
            cfg.history,
            cfg.forecast_horizon,
        ),
        val_dataset=SIDiffusionDirectDataset(
            infected,
            _starts_for_scenarios(val_scenario_ids, data_cfg),
            cfg.history,
            cfg.forecast_horizon,
        ),
        test_dataset=SIDiffusionDirectDataset(
            infected,
            _starts_for_scenarios(test_scenario_ids, data_cfg),
            cfg.history,
            cfg.forecast_horizon,
        ),
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_xy=xy,
        fixed_pos_features=fixed_pos,
        train_scenario_ids=train_scenario_ids,
        val_scenario_ids=val_scenario_ids,
        test_scenario_ids=test_scenario_ids,
        rollout_scenario_ids=test_scenario_ids,
        available_timesteps=available_timesteps,
    )

    return prepared, modes, mode_values
