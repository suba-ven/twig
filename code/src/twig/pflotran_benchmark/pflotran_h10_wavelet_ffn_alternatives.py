"""
PFLOTRAN H=10/F=10 TWIG block alternatives.

This module defines two raw-history, scale-aware TWIG variants that replace
(or strengthen) the standard pointwise FFN inside every graph-wavelet block:

A. Wavelet + SwiGLU
   x <- x + GraphWaveletOperator(LN(x))
   x <- x + SwiGLU(LN(x))

B. Wavelet + depthwise-separable Chebyshev convolution
   x <- x + GraphWaveletOperator(LN(x))
   x <- x + Pointwise(GELU(DepthwiseCheb_K(LN(x))))

Both models preserve the core temporal-causal and graph-wavelet components:

    raw history + fixed SA time-causal features + static node features
    -> input encoder
    -> repeated operator blocks
    -> residual block decoder

The Chebyshev branch uses the full sparse normalized graph rather than the
truncated eigenspace, so it adds local/full-spectrum polynomial filtering
without MGN/GAT-style feature-dependent message passing.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .pflotran_h10_models import (
    GraphStatic3D,
    GraphWaveletOperator,
    ModelSpec,
    StaticGraphModule,
    TimeCausalEncoder,
    add_residual_from_last,
    build_mlp,
    count_parameters,
)


# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------


def wavelet_swiglu_name(
    bands: int = 5,
    modes: int = 128,
    depth: int = 10,
) -> str:
    return (
        f"wavelet_swiglu_raw_sa_twig_h10_k{int(bands)}_"
        f"m{int(modes)}_d{int(depth)}"
    )


def wavelet_depthwise_cheb_name(
    bands: int = 5,
    modes: int = 128,
    depth: int = 10,
    cheb_order: int = 2,
) -> str:
    return (
        f"wavelet_dwcheb_raw_sa_twig_h10_k{int(bands)}_"
        f"m{int(modes)}_d{int(depth)}_q{int(cheb_order)}"
    )


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _make_input_encoder(
    in_dim: int,
    operator_width: int,
    encoder_width: int,
    encoder_depth: int,
    dropout: float,
    layer_norm: bool,
) -> nn.Sequential:
    """Build input_dim -> ... -> operator_width without changing block width."""
    in_dim = int(in_dim)
    operator_width = int(operator_width)
    encoder_width = int(encoder_width)
    encoder_depth = int(encoder_depth)

    if encoder_depth < 1:
        raise ValueError("encoder_depth must be >= 1")
    if encoder_depth == 1:
        return nn.Sequential(nn.Linear(in_dim, operator_width))

    dims = [in_dim] + [encoder_width] * (encoder_depth - 1) + [operator_width]
    return build_mlp(
        dims,
        dropout=float(dropout),
        layer_norm=bool(layer_norm),
    )


def _make_symmetric_normalized_edges(
    graph: GraphStatic3D,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return COO edges and weights for S=D^{-1/2}(A+A^T)/2 D^{-1/2}.

    This mirrors the graph symmetrization used by compute_laplacian_basis while
    retaining a sparse edge representation for Chebyshev recurrences.
    """
    edge_index = graph.edge_index.detach().cpu().long()
    edge_weight = graph.edge_weight.detach().cpu().float()
    n_nodes = int(graph.n_nodes)

    src, dst = edge_index
    sym_index = torch.cat(
        [
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0),
        ],
        dim=1,
    )
    sym_values = torch.cat([0.5 * edge_weight, 0.5 * edge_weight], dim=0)

    sparse = torch.sparse_coo_tensor(
        sym_index,
        sym_values,
        size=(n_nodes, n_nodes),
    ).coalesce()

    sym_index = sparse.indices().contiguous()
    sym_values = sparse.values().contiguous()
    src, dst = sym_index

    degree = torch.zeros(n_nodes, dtype=sym_values.dtype)
    degree.index_add_(0, dst, sym_values)
    inv_sqrt_degree = degree.clamp_min(1.0e-12).rsqrt()
    norm_values = sym_values * inv_sqrt_degree[src] * inv_sqrt_degree[dst]

    return sym_index, norm_values.contiguous()


def _normalized_adjacency_apply(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply symmetric normalized adjacency to x of shape (B,N,C)."""
    if x.dim() != 3:
        raise ValueError(f"Expected x with shape (B,N,C), got {tuple(x.shape)}")

    src, dst = edge_index
    messages = x.index_select(1, src)
    messages = messages * edge_weight.view(1, -1, 1)

    out = torch.zeros_like(x)
    out.index_add_(1, dst, messages)
    return out


class RawSATCWaveletBackbone3D(StaticGraphModule):
    """Shared raw-history + SA temporal feature backbone."""

    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int,
        forecast_horizon: int,
        channels: int,
        operator_width: int,
        encoder_width: int,
        encoder_depth: int,
        bands: int,
        operator_depth: int,
        n_modes: Optional[int],
        dropout: float,
        scale_aware: bool,
        encoder_layer_norm: bool,
    ):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.operator_width = int(operator_width)
        self.width = int(operator_width)
        self.encoder_width = int(encoder_width)
        self.encoder_depth = int(encoder_depth)
        self.depth = int(operator_depth)
        self.bands = int(bands)

        self.tc = TimeCausalEncoder(
            history=int(history),
            channels=int(channels),
            bands=int(bands),
            scale_aware=bool(scale_aware),
        )

        retained_modes = min(
            int(n_modes if n_modes is not None else modes.shape[1]),
            int(modes.shape[1]),
        )
        self.n_modes = int(retained_modes)
        self.register_buffer(
            "modes",
            modes[:, :retained_modes].detach().float(),
            persistent=False,
        )
        self.register_buffer(
            "evals",
            evals[:retained_modes].detach().float(),
            persistent=False,
        )

        input_dim = (
            int(history) * int(channels)
            + int(self.tc.out_channels)
            + int(self.node_static.shape[1])
        )
        self.input_dim = int(input_dim)
        self.input_encoder = _make_input_encoder(
            in_dim=input_dim,
            operator_width=operator_width,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            dropout=dropout,
            layer_norm=encoder_layer_norm,
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history, self.channels)
        batch_size, history, n_nodes, channels = context.shape

        raw_history = (
            context.permute(0, 2, 1, 3)
            .reshape(batch_size, n_nodes, history * channels)
        )
        tc_features = self.tc(context)
        static_features = (
            self.node_static
            .to(device=context.device, dtype=context.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        features = torch.cat(
            [raw_history, tc_features, static_features],
            dim=-1,
        )
        return self.input_encoder(features)

    def basis_for(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.modes.to(device=context.device, dtype=context.dtype),
            self.evals.to(device=context.device, dtype=context.dtype),
        )


# -----------------------------------------------------------------------------
# A: Wavelet + SwiGLU
# -----------------------------------------------------------------------------


class SwiGLUChannelMixer(nn.Module):
    """Pointwise gated channel mixer used instead of the standard GELU FFN."""

    def __init__(
        self,
        width: int,
        hidden_mult: float = 4.0 / 3.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        width = int(width)
        hidden = max(1, int(round(width * float(hidden_mult))))
        self.width = width
        self.hidden = hidden
        self.in_proj = nn.Linear(width, 2 * hidden)
        self.out_proj = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        mixed = value * torch.nn.functional.silu(gate)
        return self.out_proj(self.dropout(mixed))


class WaveletSwiGLUBlock3D(nn.Module):
    """Graph-wavelet residual followed by a pointwise SwiGLU residual."""

    def __init__(
        self,
        width: int,
        n_modes: int,
        n_scales: int = 4,
        swiglu_mult: float = 4.0 / 3.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm_wavelet = nn.LayerNorm(int(width))
        self.wavelet = GraphWaveletOperator(
            width=int(width),
            n_modes=int(n_modes),
            n_scales=int(n_scales),
        )
        self.norm_mixer = nn.LayerNorm(int(width))
        self.mixer = SwiGLUChannelMixer(
            width=int(width),
            hidden_mult=float(swiglu_mult),
            dropout=float(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        modes: torch.Tensor,
        evals: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.wavelet(self.norm_wavelet(x), modes, evals)
        return x + self.mixer(self.norm_mixer(x))


class RawSATCWaveletSwiGLU3D(RawSATCWaveletBackbone3D):
    """Raw-history SA TWIG with SwiGLU replacing every standard FFN."""

    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int = 10,
        forecast_horizon: int = 10,
        channels: int = 2,
        operator_width: int = 128,
        encoder_width: int = 512,
        encoder_depth: int = 2,
        bands: int = 5,
        operator_depth: int = 10,
        n_scales: int = 4,
        n_modes: Optional[int] = 128,
        swiglu_mult: float = 4.0 / 3.0,
        dropout: float = 0.05,
        scale_aware: bool = True,
        encoder_layer_norm: bool = True,
    ):
        super().__init__(
            graph=graph,
            evals=evals,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            operator_width=operator_width,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            bands=bands,
            operator_depth=operator_depth,
            n_modes=n_modes,
            dropout=dropout,
            scale_aware=scale_aware,
            encoder_layer_norm=encoder_layer_norm,
        )

        self.swiglu_mult = float(swiglu_mult)
        self.blocks = nn.ModuleList(
            [
                WaveletSwiGLUBlock3D(
                    width=operator_width,
                    n_modes=self.n_modes,
                    n_scales=n_scales,
                    swiglu_mult=swiglu_mult,
                    dropout=dropout,
                )
                for _ in range(operator_depth)
            ]
        )
        self.decoder = build_mlp(
            [operator_width, operator_width, forecast_horizon * channels],
            dropout=dropout,
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.encode_context(context)
        modes, evals = self.basis_for(context)
        for block in self.blocks:
            x = block(x, modes, evals)
        delta = self.decoder(x)
        return add_residual_from_last(
            context,
            delta,
            self.forecast_horizon,
            self.channels,
        )


# -----------------------------------------------------------------------------
# B: Wavelet + depthwise-separable Chebyshev convolution
# -----------------------------------------------------------------------------


class DepthwiseChebyshevFilter(nn.Module):
    """Channelwise Chebyshev polynomial filter on the full normalized graph.

    We use the normalized-Laplacian rescaling lambda_max=2. For
    L=I-S, where S=D^{-1/2}AD^{-1/2}, the scaled operator is
    L_tilde = 2L/2 - I = -S. The recurrence therefore uses sparse normalized
    adjacency applications and never forms a dense N x N matrix.
    """

    def __init__(self, width: int, order: int = 2):
        super().__init__()
        self.width = int(width)
        self.order = int(order)
        if self.order < 1:
            raise ValueError("Chebyshev order must be >= 1")

        self.coefficients = nn.Parameter(
            torch.empty(self.order + 1, self.width)
        )
        nn.init.normal_(self.coefficients, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        terms = [x]

        # T_1(L_tilde)x = -Sx.
        term_prev = x
        term_curr = -_normalized_adjacency_apply(
            x,
            edge_index,
            edge_weight,
        )
        terms.append(term_curr)

        # T_k(z)=2zT_{k-1}(z)-T_{k-2}(z), with z=-S.
        for _ in range(2, self.order + 1):
            term_next = (
                -2.0
                * _normalized_adjacency_apply(
                    term_curr,
                    edge_index,
                    edge_weight,
                )
                - term_prev
            )
            terms.append(term_next)
            term_prev, term_curr = term_curr, term_next

        stacked = torch.stack(terms, dim=0)  # (Q+1,B,N,C)
        coeff = self.coefficients.to(
            device=x.device,
            dtype=x.dtype,
        ).view(self.order + 1, 1, 1, self.width)
        return (stacked * coeff).sum(dim=0)


class DepthwiseSeparableChebMixer(nn.Module):
    """Depthwise graph polynomial followed by one pointwise channel map."""

    def __init__(
        self,
        width: int,
        order: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depthwise = DepthwiseChebyshevFilter(width=width, order=order)
        self.pointwise = nn.Linear(int(width), int(width))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        z = self.depthwise(x, edge_index, edge_weight)
        z = self.activation(z)
        z = self.dropout(z)
        return self.pointwise(z)


class WaveletDepthwiseChebBlock3D(nn.Module):
    """Global graph-wavelet residual + local/full-spectrum Chebyshev residual."""

    def __init__(
        self,
        width: int,
        n_modes: int,
        n_scales: int = 4,
        cheb_order: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm_wavelet = nn.LayerNorm(int(width))
        self.wavelet = GraphWaveletOperator(
            width=int(width),
            n_modes=int(n_modes),
            n_scales=int(n_scales),
        )
        self.norm_cheb = nn.LayerNorm(int(width))
        self.cheb = DepthwiseSeparableChebMixer(
            width=int(width),
            order=int(cheb_order),
            dropout=float(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        modes: torch.Tensor,
        evals: torch.Tensor,
        cheb_edge_index: torch.Tensor,
        cheb_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.wavelet(self.norm_wavelet(x), modes, evals)
        return x + self.cheb(
            self.norm_cheb(x),
            cheb_edge_index,
            cheb_edge_weight,
        )


class RawSATCWaveletDepthwiseCheb3D(RawSATCWaveletBackbone3D):
    """Raw-history SA TWIG with depthwise ChebConv replacing the FFN."""

    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int = 10,
        forecast_horizon: int = 10,
        channels: int = 2,
        operator_width: int = 128,
        encoder_width: int = 512,
        encoder_depth: int = 2,
        bands: int = 5,
        operator_depth: int = 10,
        n_scales: int = 4,
        n_modes: Optional[int] = 128,
        cheb_order: int = 2,
        dropout: float = 0.05,
        scale_aware: bool = True,
        encoder_layer_norm: bool = True,
    ):
        super().__init__(
            graph=graph,
            evals=evals,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            operator_width=operator_width,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            bands=bands,
            operator_depth=operator_depth,
            n_modes=n_modes,
            dropout=dropout,
            scale_aware=scale_aware,
            encoder_layer_norm=encoder_layer_norm,
        )

        self.cheb_order = int(cheb_order)
        cheb_edge_index, cheb_edge_weight = _make_symmetric_normalized_edges(graph)
        self.register_buffer(
            "cheb_edge_index",
            cheb_edge_index.long(),
            persistent=False,
        )
        self.register_buffer(
            "cheb_edge_weight",
            cheb_edge_weight.float(),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [
                WaveletDepthwiseChebBlock3D(
                    width=operator_width,
                    n_modes=self.n_modes,
                    n_scales=n_scales,
                    cheb_order=cheb_order,
                    dropout=dropout,
                )
                for _ in range(operator_depth)
            ]
        )
        self.decoder = build_mlp(
            [operator_width, operator_width, forecast_horizon * channels],
            dropout=dropout,
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.encode_context(context)
        modes, evals = self.basis_for(context)
        edge_index = self.cheb_edge_index.to(device=context.device)
        edge_weight = self.cheb_edge_weight.to(
            device=context.device,
            dtype=context.dtype,
        )

        for block in self.blocks:
            x = block(
                x,
                modes,
                evals,
                edge_index,
                edge_weight,
            )

        delta = self.decoder(x)
        return add_residual_from_last(
            context,
            delta,
            self.forecast_horizon,
            self.channels,
        )


# -----------------------------------------------------------------------------
# Parameter matching and ModelSpec builders
# -----------------------------------------------------------------------------


def _choose_width(
    factory_for_width: Callable[[int], nn.Module],
    target_parameters: int,
    width_candidates: Iterable[int],
) -> Tuple[int, int, Sequence[Tuple[int, int]]]:
    rows = []
    for width in width_candidates:
        model = factory_for_width(int(width))
        parameters = count_parameters(model)
        rows.append((int(width), int(parameters)))
        del model

    if not rows:
        raise ValueError("width_candidates is empty")

    best_width, best_parameters = min(
        rows,
        key=lambda item: abs(item[1] - int(target_parameters)),
    )
    return best_width, best_parameters, rows


def _make_spec(
    name: str,
    label: str,
    factory: Callable[[], nn.Module],
) -> ModelSpec:
    model = factory()
    parameters = count_parameters(model)
    del model
    return ModelSpec(
        name=name,
        label=label,
        parameters=int(parameters),
        factory=factory,
    )


def build_wavelet_swiglu_spec(
    graph: GraphStatic3D,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    history: int = 10,
    forecast_horizon: int = 10,
    channels: int = 2,
    bands: int = 5,
    n_modes: int = 128,
    operator_depth: int = 10,
    encoder_width: int = 512,
    encoder_depth: int = 2,
    swiglu_mult: float = 4.0 / 3.0,
    n_scales: int = 4,
    dropout: float = 0.05,
    target_parameters: int = 1_000_000,
    width_candidates: Iterable[int] = range(48, 225, 4),
) -> Tuple[ModelSpec, Dict[str, object]]:
    if modes.shape[1] < int(n_modes):
        raise ValueError(
            f"modes has {modes.shape[1]} columns, but n_modes={n_modes}. "
            "Recompute the basis with at least that many modes."
        )

    def factory_for_width(width: int) -> nn.Module:
        return RawSATCWaveletSwiGLU3D(
            graph=graph,
            evals=eigenvalues,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            operator_width=width,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            bands=bands,
            operator_depth=operator_depth,
            n_scales=n_scales,
            n_modes=n_modes,
            swiglu_mult=swiglu_mult,
            dropout=dropout,
        )

    width, parameters, search = _choose_width(
        factory_for_width,
        target_parameters=target_parameters,
        width_candidates=width_candidates,
    )
    name = wavelet_swiglu_name(bands, n_modes, operator_depth)
    label = (
        "SA TWIG + raw history: wavelet + SwiGLU "
        f"(K={bands}, M={n_modes}, d={width})"
    )

    def factory() -> nn.Module:
        return factory_for_width(width)

    spec = _make_spec(name, label, factory)
    config = {
        "variant": "wavelet_swiglu",
        "operator_width": int(width),
        "operator_depth": int(operator_depth),
        "encoder_width": int(encoder_width),
        "encoder_depth": int(encoder_depth),
        "bands": int(bands),
        "n_modes": int(n_modes),
        "n_scales": int(n_scales),
        "swiglu_mult": float(swiglu_mult),
        "parameters": int(parameters),
        "target_parameters": int(target_parameters),
        "width_search": list(search),
    }
    return spec, config


def build_wavelet_depthwise_cheb_spec(
    graph: GraphStatic3D,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    history: int = 10,
    forecast_horizon: int = 10,
    channels: int = 2,
    bands: int = 5,
    n_modes: int = 128,
    operator_depth: int = 10,
    encoder_width: int = 512,
    encoder_depth: int = 2,
    cheb_order: int = 2,
    n_scales: int = 4,
    dropout: float = 0.05,
    target_parameters: int = 1_000_000,
    width_candidates: Iterable[int] = range(48, 257, 4),
) -> Tuple[ModelSpec, Dict[str, object]]:
    if modes.shape[1] < int(n_modes):
        raise ValueError(
            f"modes has {modes.shape[1]} columns, but n_modes={n_modes}. "
            "Recompute the basis with at least that many modes."
        )

    def factory_for_width(width: int) -> nn.Module:
        return RawSATCWaveletDepthwiseCheb3D(
            graph=graph,
            evals=eigenvalues,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            operator_width=width,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            bands=bands,
            operator_depth=operator_depth,
            n_scales=n_scales,
            n_modes=n_modes,
            cheb_order=cheb_order,
            dropout=dropout,
        )

    width, parameters, search = _choose_width(
        factory_for_width,
        target_parameters=target_parameters,
        width_candidates=width_candidates,
    )
    name = wavelet_depthwise_cheb_name(
        bands,
        n_modes,
        operator_depth,
        cheb_order,
    )
    label = (
        "SA TWIG + raw history: wavelet + depthwise ChebConv "
        f"(K={bands}, M={n_modes}, Q={cheb_order}, d={width})"
    )

    def factory() -> nn.Module:
        return factory_for_width(width)

    spec = _make_spec(name, label, factory)
    config = {
        "variant": "wavelet_depthwise_cheb",
        "operator_width": int(width),
        "operator_depth": int(operator_depth),
        "encoder_width": int(encoder_width),
        "encoder_depth": int(encoder_depth),
        "bands": int(bands),
        "n_modes": int(n_modes),
        "n_scales": int(n_scales),
        "cheb_order": int(cheb_order),
        "parameters": int(parameters),
        "target_parameters": int(target_parameters),
        "width_search": list(search),
    }
    return spec, config
# -----------------------------------------------------------------------------
# C: Wavelet + full order-specific Chebyshev convolution
# -----------------------------------------------------------------------------


def wavelet_full_cheb_name(
    bands: int = 5,
    modes: int = 128,
    depth: int = 10,
    cheb_order: int = 2,
) -> str:
    return (
        f"wavelet_fullcheb_raw_sa_twig_h10_k{int(bands)}_"
        f"m{int(modes)}_d{int(depth)}_q{int(cheb_order)}"
    )


class FullChebyshevMixer(nn.Module):
    """Full order-specific Chebyshev convolution.

    For x with shape (B,N,d),

        z = sum_q T_q(L_tilde) x Theta_q
        y = W_out Dropout(GELU(z)).

    Every polynomial order has its own full d -> h projection. For Q=2 and
    h=d, the dominant parameter count is 3d^2+d^2=4d^2, comparable to the
    original d -> 2d -> d FFN.
    """

    def __init__(
        self,
        width: int,
        order: int = 2,
        hidden_mult: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = int(width)
        self.order = int(order)
        if self.order < 1:
            raise ValueError("Chebyshev order must be >= 1")

        self.hidden = max(1, int(round(self.width * float(hidden_mult))))
        self.order_projections = nn.ModuleList(
            [
                nn.Linear(self.width, self.hidden, bias=False)
                for _ in range(self.order + 1)
            ]
        )
        self.activation = nn.GELU()
        self.dropout = (
            nn.Dropout(float(dropout))
            if float(dropout) > 0.0
            else nn.Identity()
        )
        self.out_proj = nn.Linear(self.hidden, self.width, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected x with shape (B,N,C), got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.width:
            raise ValueError(
                f"Expected width={self.width}, got {x.shape[-1]}"
            )

        # T_0(L_tilde)x = x.
        term_prev = x
        mixed = self.order_projections[0](term_prev)

        # T_1(L_tilde)x = -Sx, because L_tilde=-S for lambda_max=2.
        term_curr = -_normalized_adjacency_apply(
            x,
            edge_index,
            edge_weight,
        )
        mixed = mixed + self.order_projections[1](term_curr)

        # T_q(z)=2zT_{q-1}(z)-T_{q-2}(z), z=-S.
        for q in range(2, self.order + 1):
            term_next = (
                -2.0
                * _normalized_adjacency_apply(
                    term_curr,
                    edge_index,
                    edge_weight,
                )
                - term_prev
            )
            mixed = mixed + self.order_projections[q](term_next)
            term_prev, term_curr = term_curr, term_next

        return self.out_proj(
            self.dropout(
                self.activation(mixed)
            )
        )


class WaveletFullChebBlock3D(nn.Module):
    """Wavelet residual followed by a full-ChebConv residual."""

    def __init__(
        self,
        width: int,
        n_modes: int,
        n_scales: int = 4,
        cheb_order: int = 2,
        cheb_hidden_mult: float = 1.0,
        initial_cheb_scale: float = 0.05,
        learnable_cheb_scale: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm_wavelet = nn.LayerNorm(int(width))
        self.wavelet = GraphWaveletOperator(
            width=int(width),
            n_modes=int(n_modes),
            n_scales=int(n_scales),
        )

        self.norm_cheb = nn.LayerNorm(int(width))
        self.cheb = FullChebyshevMixer(
            width=int(width),
            order=int(cheb_order),
            hidden_mult=float(cheb_hidden_mult),
            dropout=float(dropout),
        )

        initial_scale = torch.tensor(float(initial_cheb_scale))
        if bool(learnable_cheb_scale):
            self.cheb_scale = nn.Parameter(initial_scale)
        else:
            self.register_buffer(
                "cheb_scale",
                initial_scale,
                persistent=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        modes: torch.Tensor,
        evals: torch.Tensor,
        cheb_edge_index: torch.Tensor,
        cheb_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.wavelet(self.norm_wavelet(x), modes, evals)

        cheb_update = self.cheb(
            self.norm_cheb(x),
            cheb_edge_index,
            cheb_edge_weight,
        )
        scale = self.cheb_scale.to(device=x.device, dtype=x.dtype)
        return x + scale * cheb_update


class RawSATCWaveletFullCheb3D(RawSATCWaveletBackbone3D):
    """Raw-history SA TWIG with full ChebConv replacing each FFN."""

    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int = 10,
        forecast_horizon: int = 10,
        channels: int = 2,
        operator_width: int = 96,
        encoder_width: int = 512,
        encoder_depth: int = 2,
        bands: int = 5,
        operator_depth: int = 10,
        n_scales: int = 4,
        n_modes: Optional[int] = 128,
        cheb_order: int = 2,
        cheb_hidden_mult: float = 1.0,
        initial_cheb_scale: float = 0.05,
        learnable_cheb_scale: bool = True,
        dropout: float = 0.05,
        scale_aware: bool = True,
        encoder_layer_norm: bool = True,
    ):
        super().__init__(
            graph=graph,
            evals=evals,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            operator_width=operator_width,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            bands=bands,
            operator_depth=operator_depth,
            n_modes=n_modes,
            dropout=dropout,
            scale_aware=scale_aware,
            encoder_layer_norm=encoder_layer_norm,
        )

        self.cheb_order = int(cheb_order)
        self.cheb_hidden_mult = float(cheb_hidden_mult)

        cheb_edge_index, cheb_edge_weight = _make_symmetric_normalized_edges(
            graph
        )
        self.register_buffer(
            "cheb_edge_index",
            cheb_edge_index.long(),
            persistent=False,
        )
        self.register_buffer(
            "cheb_edge_weight",
            cheb_edge_weight.float(),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [
                WaveletFullChebBlock3D(
                    width=operator_width,
                    n_modes=self.n_modes,
                    n_scales=n_scales,
                    cheb_order=cheb_order,
                    cheb_hidden_mult=cheb_hidden_mult,
                    initial_cheb_scale=initial_cheb_scale,
                    learnable_cheb_scale=learnable_cheb_scale,
                    dropout=dropout,
                )
                for _ in range(operator_depth)
            ]
        )
        self.decoder = build_mlp(
            [operator_width, operator_width, forecast_horizon * channels],
            dropout=dropout,
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.encode_context(context)
        modes, evals = self.basis_for(context)
        edge_index = self.cheb_edge_index.to(device=context.device)
        edge_weight = self.cheb_edge_weight.to(
            device=context.device,
            dtype=context.dtype,
        )

        for block in self.blocks:
            x = block(
                x,
                modes,
                evals,
                edge_index,
                edge_weight,
            )

        delta = self.decoder(x)
        return add_residual_from_last(
            context,
            delta,
            self.forecast_horizon,
            self.channels,
        )

    def current_cheb_scales(self) -> Sequence[float]:
        return [
            float(block.cheb_scale.detach().cpu())
            for block in self.blocks
        ]


def build_wavelet_full_cheb_spec(
    graph: GraphStatic3D,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    history: int = 10,
    forecast_horizon: int = 10,
    channels: int = 2,
    bands: int = 5,
    n_modes: int = 128,
    operator_depth: int = 10,
    encoder_width: int = 512,
    encoder_depth: int = 2,
    cheb_order: int = 2,
    cheb_hidden_mult: float = 1.0,
    initial_cheb_scale: float = 0.05,
    learnable_cheb_scale: bool = True,
    n_scales: int = 4,
    dropout: float = 0.05,
    target_parameters: int = 1_000_000,
    width_candidates: Iterable[int] = range(48, 193, 4),
) -> Tuple[ModelSpec, Dict[str, object]]:
    if modes.shape[1] < int(n_modes):
        raise ValueError(
            f"modes has {modes.shape[1]} columns, but n_modes={n_modes}. "
            "Recompute the basis with at least that many modes."
        )

    def factory_for_width(width: int) -> nn.Module:
        return RawSATCWaveletFullCheb3D(
            graph=graph,
            evals=eigenvalues,
            modes=modes,
            history=history,
            forecast_horizon=forecast_horizon,
            channels=channels,
            operator_width=width,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            bands=bands,
            operator_depth=operator_depth,
            n_scales=n_scales,
            n_modes=n_modes,
            cheb_order=cheb_order,
            cheb_hidden_mult=cheb_hidden_mult,
            initial_cheb_scale=initial_cheb_scale,
            learnable_cheb_scale=learnable_cheb_scale,
            dropout=dropout,
        )

    width, parameters, search = _choose_width(
        factory_for_width,
        target_parameters=target_parameters,
        width_candidates=width_candidates,
    )

    name = wavelet_full_cheb_name(
        bands=bands,
        modes=n_modes,
        depth=operator_depth,
        cheb_order=cheb_order,
    )
    label = (
        "SA TWIG + raw history: wavelet + full ChebConv "
        f"(K={bands}, M={n_modes}, Q={cheb_order}, d={width})"
    )

    def factory() -> nn.Module:
        return factory_for_width(width)

    spec = _make_spec(name, label, factory)
    config = {
        "variant": "wavelet_full_cheb",
        "operator_width": int(width),
        "operator_depth": int(operator_depth),
        "encoder_width": int(encoder_width),
        "encoder_depth": int(encoder_depth),
        "bands": int(bands),
        "n_modes": int(n_modes),
        "n_scales": int(n_scales),
        "cheb_order": int(cheb_order),
        "cheb_hidden_mult": float(cheb_hidden_mult),
        "initial_cheb_scale": float(initial_cheb_scale),
        "learnable_cheb_scale": bool(learnable_cheb_scale),
        "parameters": int(parameters),
        "target_parameters": int(target_parameters),
        "width_search": list(search),
    }
    return spec, config


__all__ = [
    "RawSATCWaveletSwiGLU3D",
    "RawSATCWaveletDepthwiseCheb3D",
    "RawSATCWaveletFullCheb3D",
    "WaveletSwiGLUBlock3D",
    "WaveletDepthwiseChebBlock3D",
    "WaveletFullChebBlock3D",
    "SwiGLUChannelMixer",
    "DepthwiseChebyshevFilter",
    "DepthwiseSeparableChebMixer",
    "FullChebyshevMixer",
    "build_wavelet_swiglu_spec",
    "build_wavelet_depthwise_cheb_spec",
    "build_wavelet_full_cheb_spec",
    "wavelet_swiglu_name",
    "wavelet_depthwise_cheb_name",
    "wavelet_full_cheb_name",
]
