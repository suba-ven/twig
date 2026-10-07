"""
PFLOTRAN H=10 raw-history TWIG ablation models.

Drop this file in the same directory as `pflotran_h10_models.py` and
`pflotran_h10_training.py`.

Adds three model variants:
  1. SA TWIG + raw history, K=5, M=128
  2. SA TWIG + raw history, K=5, M=256
  3. Hybrid SA TWIG + raw history, K=5, M=128

The key architecture change relative to SATWIG3D in `pflotran_h10_models.py`
is that the input projection receives BOTH:
  - the full flattened raw history, shape H*C, and
  - the time-causal multiscale features, shape (K+2)*C,
plus the existing static node features.

The decoder's last linear layer is zero-initialized by default. Since these
models predict residuals added to the last observed state, this initializes the
model as a persistence forecaster and lets training learn corrections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .pflotran_h10_models import (
    GraphStatic3D,
    ModelSpec,
    StaticGraphModule,
    TimeCausalEncoder,
    GraphWNOBlock3D,
    GraphWaveletOperator,
    MGNBlock,
    build_mlp,
    count_parameters,
    add_residual_from_last,
)


# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------

def raw_sa_twig_name(modes: int, bands: int = 5) -> str:
    return f"raw_sa_twig_h10_k{int(bands)}_m{int(modes)}"


def hybrid_raw_sa_twig_name(modes: int, bands: int = 5) -> str:
    return f"hybrid_raw_sa_twig_h10_k{int(bands)}_m{int(modes)}"


def zero_init_last_linear(module: nn.Module) -> None:
    """Zero-initialize the final Linear layer inside a decoder module."""
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise ValueError("No nn.Linear layer found for zero initialization.")
    nn.init.zeros_(last_linear.weight)
    if last_linear.bias is not None:
        nn.init.zeros_(last_linear.bias)


# -----------------------------------------------------------------------------
# SA TWIG + raw-history skip
# -----------------------------------------------------------------------------

class RawHistorySATWIG3D(StaticGraphModule):
    """Scale-aware TWIG with a raw-history skip in the input features.

    Input features per node are:
        [flattened raw history | TC multiscale features | static node features]

    Output is residual-to-last-state block prediction, same as the existing
    PFLOTRAN H10 benchmark models.
    """

    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int,
        forecast_horizon: int,
        channels: int,
        width: int,
        bands: int = 5,
        depth: int = 10,
        n_scales: int = 4,
        n_modes: Optional[int] = None,
        dropout: float = 0.05,
        scale_aware: bool = True,
        zero_init_decoder: bool = True,
    ):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.width = int(width)
        self.depth = int(depth)
        self.bands = int(bands)

        self.tc = TimeCausalEncoder(history, channels, bands, scale_aware=scale_aware)

        K = min(int(n_modes or modes.shape[1]), int(modes.shape[1]))
        self.n_modes = K
        self.register_buffer("modes", modes[:, :K].float(), persistent=False)
        self.register_buffer("evals", evals[:K].float(), persistent=False)

        in_dim = history * channels + self.tc.out_channels + self.node_static.shape[1]
        self.input_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([
            GraphWNOBlock3D(width, K, n_scales, dropout)
            for _ in range(depth)
        ])
        self.decoder = build_mlp([width, width, forecast_horizon * channels], dropout=dropout)
        if zero_init_decoder:
            zero_init_last_linear(self.decoder)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape

        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        tc_feat = self.tc(context)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)

        x = torch.cat([hist, tc_feat, static], dim=-1)
        x = self.input_proj(x)

        U = self.modes.to(device=context.device, dtype=context.dtype)
        lam = self.evals.to(device=context.device, dtype=context.dtype)
        for block in self.blocks:
            x = block(x, U, lam)

        delta = self.decoder(x)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


# -----------------------------------------------------------------------------
# Hybrid SA TWIG + raw-history skip
# -----------------------------------------------------------------------------

class HybridRawTWIGBlock3D(nn.Module):
    """Wavelet residual + local MGN residual + pointwise FFN residual."""

    def __init__(
        self,
        width: int,
        n_modes: int,
        edge_dim: int,
        n_scales: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm_wav = nn.LayerNorm(width)
        self.wavelet = GraphWaveletOperator(width, n_modes, n_scales)
        self.local = MGNBlock(width, edge_dim, dropout)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * width, width),
        )

    def forward(
        self,
        x: torch.Tensor,
        U_modes: torch.Tensor,
        evals_modes: torch.Tensor,
        edge_index_batched: torch.Tensor,
        edge_attr_batched: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B,N,width)
        B, N, W = x.shape
        x = x + self.wavelet(self.norm_wav(x), U_modes, evals_modes)

        x_flat = x.reshape(B * N, W)
        x_flat = self.local(x_flat, edge_index_batched, edge_attr_batched)
        x = x_flat.reshape(B, N, W)

        x = x + self.ffn(self.norm_ffn(x))
        return x


class HybridRawHistorySATWIG3D(StaticGraphModule):
    """SA TWIG with raw-history input and local message passing in each block."""

    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int,
        forecast_horizon: int,
        channels: int,
        width: int,
        bands: int = 5,
        depth: int = 10,
        n_scales: int = 4,
        n_modes: Optional[int] = None,
        dropout: float = 0.05,
        scale_aware: bool = True,
        zero_init_decoder: bool = True,
    ):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.width = int(width)
        self.depth = int(depth)
        self.bands = int(bands)

        self.tc = TimeCausalEncoder(history, channels, bands, scale_aware=scale_aware)

        K = min(int(n_modes or modes.shape[1]), int(modes.shape[1]))
        self.n_modes = K
        self.register_buffer("modes", modes[:, :K].float(), persistent=False)
        self.register_buffer("evals", evals[:K].float(), persistent=False)

        in_dim = history * channels + self.tc.out_channels + self.node_static.shape[1]
        self.input_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([
            HybridRawTWIGBlock3D(
                width=width,
                n_modes=K,
                edge_dim=self.edge_attr_static.shape[1],
                n_scales=n_scales,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.decoder = build_mlp([width, width, forecast_horizon * channels], dropout=dropout)
        if zero_init_decoder:
            zero_init_last_linear(self.decoder)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape

        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        tc_feat = self.tc(context)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)

        x = torch.cat([hist, tc_feat, static], dim=-1)
        x = self.input_proj(x)

        U = self.modes.to(device=context.device, dtype=context.dtype)
        lam = self.evals.to(device=context.device, dtype=context.dtype)

        edge_index_batched, edge_attr_batched = self.batched_graph(B, edge_attr=True)
        edge_index_batched = edge_index_batched.to(context.device)
        edge_attr_batched = edge_attr_batched.to(context.device, dtype=context.dtype)

        for block in self.blocks:
            x = block(x, U, lam, edge_index_batched, edge_attr_batched)

        delta = self.decoder(x)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


# -----------------------------------------------------------------------------
# Spec builder for the three requested ablations
# -----------------------------------------------------------------------------

def _make_spec(name: str, label: str, factory) -> ModelSpec:
    model = factory()
    params = count_parameters(model)
    del model
    return ModelSpec(name=name, label=label, parameters=params, factory=factory)


def build_raw_history_ablation_specs(
    graph: GraphStatic3D,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    history: int = 10,
    forecast_horizon: int = 10,
    channels: int = 2,
    base_width: int,
    operator_depth: int = 10,
    bands: int = 5,
    dropout: float = 0.05,
    n_scales: int = 4,
    zero_init_decoder: bool = True,
) -> Tuple[Dict[str, ModelSpec], Dict[str, Dict[str, int]]]:
    """Build exactly the three requested model specs.

    `modes` and `eigenvalues` must contain at least 256 modes if you want the
    M=256 variant. Use `compute_laplacian_basis(graph, n_modes=256)` before
    calling this function.

    `base_width` should be copied from the already-matched SA TWIG K=5
    width in your existing benchmark so the 128/256-mode variants keep the same
    width/depth as your 64-mode run.
    """
    if modes.shape[1] < 256:
        raise ValueError(
            f"modes has only {modes.shape[1]} columns. Recompute the basis with n_modes=256."
        )
    if eigenvalues.shape[0] < 256:
        raise ValueError(
            f"eigenvalues has only {eigenvalues.shape[0]} entries. Recompute the basis with n_modes=256."
        )

    specs: Dict[str, ModelSpec] = {}
    matched: Dict[str, Dict[str, int]] = {}

    # 1. SA TWIG + raw history, K=5, M=128
    name = raw_sa_twig_name(128, bands=bands)
    def fac_raw_128():
        return RawHistorySATWIG3D(
            graph=graph,
            evals=eigenvalues,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            width=base_width,
            bands=bands,
            depth=operator_depth,
            n_scales=n_scales,
            n_modes=128,
            dropout=dropout,
            scale_aware=True,
            zero_init_decoder=zero_init_decoder,
        )
    specs[name] = _make_spec(name, f"SA TWIG + raw history (K={bands}, M=128)", fac_raw_128)
    matched[name] = {
        "width": int(base_width),
        "depth": int(operator_depth),
        "bands": int(bands),
        "modes": 128,
        "parameters": int(specs[name].parameters),
    }

    # 2. SA TWIG + raw history, K=5, M=256
    name = raw_sa_twig_name(256, bands=bands)
    def fac_raw_256():
        return RawHistorySATWIG3D(
            graph=graph,
            evals=eigenvalues,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            width=base_width,
            bands=bands,
            depth=operator_depth,
            n_scales=n_scales,
            n_modes=256,
            dropout=dropout,
            scale_aware=True,
            zero_init_decoder=zero_init_decoder,
        )
    specs[name] = _make_spec(name, f"SA TWIG + raw history (K={bands}, M=256)", fac_raw_256)
    matched[name] = {
        "width": int(base_width),
        "depth": int(operator_depth),
        "bands": int(bands),
        "modes": 256,
        "parameters": int(specs[name].parameters),
    }

    # 3. Hybrid SA TWIG + raw history, K=5, M=128
    name = hybrid_raw_sa_twig_name(128, bands=bands)
    def fac_hybrid_raw_128():
        return HybridRawHistorySATWIG3D(
            graph=graph,
            evals=eigenvalues,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            width=base_width,
            bands=bands,
            depth=operator_depth,
            n_scales=n_scales,
            n_modes=128,
            dropout=dropout,
            scale_aware=True,
            zero_init_decoder=zero_init_decoder,
        )
    specs[name] = _make_spec(name, f"Hybrid SA TWIG + raw history (K={bands}, M=128)", fac_hybrid_raw_128)
    matched[name] = {
        "width": int(base_width),
        "depth": int(operator_depth),
        "bands": int(bands),
        "modes": 128,
        "parameters": int(specs[name].parameters),
    }

    return specs, matched


def raw_history_ablation_order(bands: int = 5):
    return [
        raw_sa_twig_name(128, bands=bands),
        raw_sa_twig_name(256, bands=bands),
        hybrid_raw_sa_twig_name(128, bands=bands),
    ]
