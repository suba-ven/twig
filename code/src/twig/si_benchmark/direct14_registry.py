from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .direct14_config import Direct14Config
from .direct14_models import (
    DirectHorizonTWIG,
    DirectMeshGraphNetBaseline,
    HistoryMatchedEncoder,
    TWIGArchitecture,
)
from .graph import GraphStatic


DIRECT14_TCW_BASE = TWIGArchitecture(
    width=34,
    depth=8,
    temporal_bands=4,
    input_channels=1,
    output_channels=14,
    wavelet_scales=4,
    residual_output=False,
)


@dataclass(frozen=True)
class DirectModelSpec:
    name: str
    factory: Callable[[], nn.Module]
    regularizer: Callable[[nn.Module], torch.Tensor] | None = None
    metadata: dict | None = None


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def import_graph_wno_block(models_3d_dir: Path):
    """Return the vendored paper implementation.

    The argument remains for configuration/checkpoint compatibility.
    """
    from .graph_wno_reference import GraphWNOBlock3D
    return GraphWNOBlock3D


def _nearest_width(
    factory: Callable[[int], nn.Module],
    target: int,
    candidates: range,
) -> tuple[int, int]:
    options: list[tuple[int, int, int]] = []

    for width in candidates:
        trial = factory(width)
        parameters = count_parameters(trial)
        options.append(
            (
                abs(parameters - target),
                int(width),
                int(parameters),
            )
        )
        del trial

    _, width, parameters = min(options)
    return width, parameters


def direct_sa_name(n_bands: int) -> str:
    return f"SA-TWIG-D14-K{int(n_bands)}"


def direct_mesh_name() -> str:
    return "MeshGraphNet-D14"


def build_direct14_model_specs(
    cfg: Direct14Config,
    graph: GraphStatic,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    graph_wno_block,
    device: torch.device,
    sa_band_counts: tuple[int, ...] = (2, 3, 4, 5),
) -> tuple[dict[str, DirectModelSpec], dict[str, int | float]]:
    """Build direct H=14/F=14 models and match MeshGraphNet to SA K=4."""

    if cfg.history != 14 or cfg.forecast_horizon != 14:
        raise ValueError(
            "This direct experiment is defined for history=14 and "
            "forecast_horizon=14."
        )

    normalized_bands = tuple(int(value) for value in sa_band_counts)
    if not normalized_bands:
        raise ValueError("sa_band_counts must contain at least one value.")
    if any(value < 1 for value in normalized_bands):
        raise ValueError("All temporal-band counts must be positive.")

    def build_sa_model(n_bands: int) -> nn.Module:
        architecture = TWIGArchitecture(
            width=DIRECT14_TCW_BASE.width,
            depth=DIRECT14_TCW_BASE.depth,
            temporal_bands=n_bands,
            input_channels=DIRECT14_TCW_BASE.input_channels,
            output_channels=cfg.forecast_horizon,
            wavelet_scales=DIRECT14_TCW_BASE.wavelet_scales,
            residual_output=False,
        )
        encoder = HistoryMatchedEncoder(
            channels=architecture.input_channels,
            history=cfg.history,
            n_bands=n_bands,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            min_cumulative_lag=cfg.min_cumulative_lag,
            max_lag_fraction=cfg.max_lag_fraction,
        )
        return DirectHorizonTWIG(
            graph_wno_block=graph_wno_block,
            modes=modes,
            eigenvalues=eigenvalues,
            fixed_pos_features=graph.node_static,
            architecture=architecture,
            temporal_encoder=encoder,
            forecast_horizon=cfg.forecast_horizon,
        )

    # Use K=4 as the capacity-reference SA model whenever it is included.
    reference_bands = 4 if 4 in normalized_bands else normalized_bands[0]
    reference_model = build_sa_model(reference_bands)
    reference_parameters = count_parameters(reference_model)
    del reference_model

    mesh_factory = lambda width: DirectMeshGraphNetBaseline(
        graph=graph,
        history=cfg.history,
        forecast_horizon=cfg.forecast_horizon,
        width=width,
        processor_steps=3,
    )
    mesh_width, mesh_parameters = _nearest_width(
        mesh_factory,
        reference_parameters,
        range(12, 129),
    )

    def sa_regularizer(model: nn.Module) -> torch.Tensor:
        return (
            cfg.scale_anchor_weight
            * model.temporal_encoder.scale_anchor_loss()
        )

    specs: dict[str, DirectModelSpec] = {}

    for n_bands in normalized_bands:
        name = direct_sa_name(n_bands)
        specs[name] = DirectModelSpec(
            name=name,
            factory=lambda n_bands=n_bands: build_sa_model(n_bands).to(
                device
            ),
            regularizer=sa_regularizer,
            metadata={
                "family": "SA-TWIG",
                "history": cfg.history,
                "forecast_horizon": cfg.forecast_horizon,
                "temporal_bands": n_bands,
                "spatial_wavelet_scales": cfg.wavelet_scales,
                "width": DIRECT14_TCW_BASE.width,
                "depth": DIRECT14_TCW_BASE.depth,
                "training_mode": "direct_14_to_14",
            },
        )

    mesh_name = direct_mesh_name()
    specs[mesh_name] = DirectModelSpec(
        name=mesh_name,
        factory=lambda: mesh_factory(mesh_width).to(device),
        metadata={
            "family": "MeshGraphNet",
            "history": cfg.history,
            "forecast_horizon": cfg.forecast_horizon,
            "width": mesh_width,
            "processor_steps": 3,
            "matched_to": direct_sa_name(reference_bands),
            "training_mode": "direct_14_to_14",
        },
    )

    matched = {
        "reference_sa_bands": reference_bands,
        "reference_sa_parameters": reference_parameters,
        "meshgraphnet_width": mesh_width,
        "meshgraphnet_parameters": mesh_parameters,
    }

    return specs, matched
