from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv, GATv2Conv, GINEConv
except ModuleNotFoundError:  # lets non-PyG models still be imported
    GATConv = None
    GATv2Conv = None
    GINEConv = None


# -----------------------------------------------------------------------------
# Names/specs
# -----------------------------------------------------------------------------

def pflotran_sa_name(k: int) -> str:
    return f"sa_tc_wgno_h10_k{int(k)}"


def pflotran_plain_gwno_name() -> str:
    return "plain_graph_wno_h10"


def pflotran_graph_fno_name() -> str:
    return "graph_fno_h10"


def pflotran_mesh_name() -> str:
    return "meshgraphnet_h10"


def pflotran_gat_name() -> str:
    return "gat_h10"


def pflotran_gatv2_name() -> str:
    return "gatv2_h10"


def pflotran_gps_name() -> str:
    return "gps_transformer_h10"


def pflotran_rnn_name() -> str:
    return "rnn_h10"


def pflotran_rnn_gnn_fusion_name() -> str:
    return "rnn_gnn_fusion_h10"


@dataclass
class ModelSpec:
    name: str
    label: str
    parameters: int
    factory: Callable[[], nn.Module]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Graph utilities
# -----------------------------------------------------------------------------

def canonicalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.dim() != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.shape[0] == 2:
        return edge_index.contiguous().long()
    if edge_index.shape[1] == 2:
        return edge_index.t().contiguous().long()
    raise ValueError(f"edge_index must have shape (2,E) or (E,2), got {tuple(edge_index.shape)}")


def make_batched_edge_index(edge_index: torch.Tensor, num_nodes: int, batch_size: int, device=None) -> torch.Tensor:
    ei = canonicalize_edge_index(edge_index)
    if device is None:
        device = ei.device
    ei = ei.to(device)
    E = ei.shape[1]
    offsets = (torch.arange(batch_size, device=device, dtype=ei.dtype) * num_nodes).view(batch_size, 1, 1)
    base = ei.t().unsqueeze(0)
    return (base + offsets).reshape(batch_size * E, 2).t().contiguous()


def segment_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_zeros(n, values.shape[-1])
    out.index_add_(0, index, values)
    return out


def build_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    ei = canonicalize_edge_index(edge_index)
    src, dst = ei
    rel = pos[dst] - pos[src]
    dist = torch.norm(rel, dim=-1, keepdim=True)
    inv = 1.0 / dist.clamp_min(1e-8)
    loginv = torch.log1p(inv)
    return torch.cat([rel, dist, loginv], dim=-1)  # 3 + 1 + 1 = 5


def build_edge_weight(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    ei = canonicalize_edge_index(edge_index)
    src, dst = ei
    dist = torch.norm(pos[dst] - pos[src], dim=-1)
    w = 1.0 / dist.clamp_min(1e-8)
    return w / w.mean().clamp_min(1e-8)


def make_node_static_features(pos: torch.Tensor, num_scales: int = 4) -> torch.Tensor:
    """Small fixed 3D positional feature map for non-operator baselines.

    The input `pos` must already be normalized by the dataset builder.
    """
    if pos.dim() != 2 or pos.shape[-1] != 3:
        raise ValueError(f"Expected normalized pos shape (N,3), got {tuple(pos.shape)}")
    feats = [pos, torch.norm(pos, dim=-1, keepdim=True)]
    scales = torch.tensor([1.0, 2.0, 4.0, 8.0][:num_scales], device=pos.device, dtype=pos.dtype)
    x = 2.0 * torch.pi * pos.unsqueeze(-1) * scales.view(1, 1, -1)
    feats.append(torch.sin(x).reshape(pos.shape[0], -1))
    feats.append(torch.cos(x).reshape(pos.shape[0], -1))
    return torch.cat(feats, dim=-1)


@dataclass
class GraphStatic3D:
    node_pos: torch.Tensor           # normalized (N,3)
    edge_index: torch.Tensor         # (2,E)
    node_static: torch.Tensor        # (N,S)
    edge_attr: torch.Tensor          # (E,5)
    edge_weight: torch.Tensor        # (E,)

    @property
    def n_nodes(self) -> int:
        return int(self.node_pos.shape[0])


def make_graph_static_3d(node_pos: torch.Tensor, edge_index: torch.Tensor) -> GraphStatic3D:
    node_pos = node_pos.detach().float().cpu()
    edge_index = canonicalize_edge_index(edge_index.detach().cpu())
    node_static = make_node_static_features(node_pos)
    edge_attr = build_edge_features(node_pos, edge_index)
    edge_weight = build_edge_weight(node_pos, edge_index)
    return GraphStatic3D(
        node_pos=node_pos,
        edge_index=edge_index,
        node_static=node_static,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
    )


def compute_laplacian_basis(
    graph: GraphStatic3D,
    n_modes: int = 96,
    symmetrize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a normalized graph-Laplacian eigensystem on the normalized geometry."""
    N = graph.n_nodes
    ei = graph.edge_index
    src, dst = ei
    w = graph.edge_weight.float()

    A = torch.zeros(N, N, dtype=torch.float32)
    A.index_put_((src, dst), w, accumulate=True)
    if symmetrize:
        A = 0.5 * (A + A.t())

    deg = A.sum(dim=1).clamp_min(1e-12)
    inv_sqrt = deg.rsqrt()
    L = torch.eye(N, dtype=torch.float32) - inv_sqrt[:, None] * A * inv_sqrt[None, :]
    evals, evecs = torch.linalg.eigh(L)
    K = min(int(n_modes), N)
    return evals[:K].contiguous(), evecs[:, :K].contiguous()


# -----------------------------------------------------------------------------
# Common modules
# -----------------------------------------------------------------------------

def build_mlp(dims: Sequence[int], *, activation=nn.GELU, dropout: float = 0.0, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
        if i < len(dims) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(int(dims[i + 1])))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


class StaticGraphModule(nn.Module):
    def __init__(self, graph: GraphStatic3D):
        super().__init__()
        self.register_buffer("node_pos_static", graph.node_pos.float(), persistent=False)
        self.register_buffer("edge_index_static", graph.edge_index.long(), persistent=False)
        self.register_buffer("node_static", graph.node_static.float(), persistent=False)
        self.register_buffer("edge_attr_static", graph.edge_attr.float(), persistent=False)
        self.register_buffer("edge_weight_static", graph.edge_weight.float(), persistent=False)
        self.n_nodes = graph.n_nodes

    def validate_context(self, context: torch.Tensor, history: int, channels: int) -> None:
        if context.dim() != 4:
            raise ValueError(f"context must have shape (B,H,N,C), got {tuple(context.shape)}")
        if context.shape[1] != int(history):
            raise ValueError(f"Expected history={history}, got {context.shape[1]}")
        if context.shape[2] != self.n_nodes:
            raise ValueError(f"Expected N={self.n_nodes}, got {context.shape[2]}")
        if context.shape[3] != int(channels):
            raise ValueError(f"Expected channels={channels}, got {context.shape[3]}")

    def batched_graph(self, batch_size: int, edge_attr: bool = False):
        ei = make_batched_edge_index(self.edge_index_static, self.n_nodes, batch_size, device=self.edge_index_static.device)
        if not edge_attr:
            return ei, None
        attr = self.edge_attr_static.repeat(batch_size, 1)
        return ei, attr


class TimeCausalEncoder(nn.Module):
    def __init__(self, history: int, channels: int, bands: int, scale_aware: bool = True):
        super().__init__()
        self.history = int(history)
        self.channels = int(channels)
        self.bands = int(bands)
        self.scale_aware = bool(scale_aware)
        alphas = self._make_alphas()
        self.register_buffer("alphas", alphas, persistent=True)

    @property
    def out_channels(self) -> int:
        return (self.bands + 2) * self.channels

    def _make_alphas(self) -> torch.Tensor:
        if self.bands < 1:
            raise ValueError("bands must be >= 1")
        if self.scale_aware:
            # e-folding memory spans from one step to the full available history.
            if self.bands == 1:
                taus = torch.tensor([float(self.history)])
            else:
                taus = torch.logspace(0.0, float(np.log10(self.history)), steps=self.bands)
            return torch.exp(-1.0 / taus).float()
        # Fixed non-scale-aware default for optional ablations.
        return torch.linspace(0.35, 0.95, steps=self.bands).float()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B,H,N,C)
        B, H, N, C = context.shape
        if H != self.history or C != self.channels:
            raise ValueError(f"Expected H={self.history}, C={self.channels}; got H={H}, C={C}")
        current = context[:, -1]  # (B,N,C)
        residuals = []
        memories = []
        for alpha in self.alphas.to(device=context.device, dtype=context.dtype):
            m = context[:, 0]
            for t in range(1, H):
                m = alpha * m + (1.0 - alpha) * context[:, t]
            memories.append(m)
            residuals.append(current - m)
        return torch.cat([current] + residuals + [memories[-1]], dim=-1)


def add_residual_from_last(context: torch.Tensor, delta_flat: torch.Tensor, forecast_horizon: int, channels: int) -> torch.Tensor:
    B, _, N, C = context.shape
    out = delta_flat.reshape(B, N, forecast_horizon, channels).permute(0, 2, 1, 3).contiguous()
    return context[:, -1:].expand_as(out) + out


# -----------------------------------------------------------------------------
# Graph wavelet operator family
# -----------------------------------------------------------------------------

class GraphWaveletOperator(nn.Module):
    def __init__(self, width: int, n_modes: int, n_scales: int = 4):
        super().__init__()
        self.width = int(width)
        self.n_modes = int(n_modes)
        self.n_scales = int(n_scales)
        self.spec_linear = nn.Linear(width, width)
        self.scale_logits = nn.Parameter(torch.linspace(-1.0, 1.0, steps=n_scales))
        self.scale_mix = nn.Parameter(torch.ones(n_scales, width))
        self.lowpass_gate = nn.Parameter(torch.ones(n_modes, width))

    def _wavelet_bank(self, evals: torch.Tensor) -> torch.Tensor:
        scales = torch.exp(self.scale_logits).to(device=evals.device, dtype=evals.dtype)
        lam = evals.view(1, -1)
        return lam * torch.exp(-scales.view(-1, 1) * lam)

    def forward(self, x: torch.Tensor, U_modes: torch.Tensor, evals_modes: torch.Tensor) -> torch.Tensor:
        coeff = torch.einsum("bnc,nk->bkc", x, U_modes)
        coeff = self.spec_linear(coeff)
        bank = self._wavelet_bank(evals_modes)
        wavelet_out = 0.0
        for s in range(self.n_scales):
            filt = bank[s].unsqueeze(-1) * self.scale_mix[s].unsqueeze(0)
            wavelet_out = wavelet_out + coeff * filt.unsqueeze(0)
        lowpass_out = coeff * self.lowpass_gate.unsqueeze(0)
        return torch.einsum("bkc,nk->bnc", wavelet_out + lowpass_out, U_modes)


class GraphWNOBlock3D(nn.Module):
    def __init__(self, width: int, n_modes: int, n_scales: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.wavelet = GraphWaveletOperator(width, n_modes, n_scales)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * width, width),
        )

    def forward(self, x: torch.Tensor, U_modes: torch.Tensor, evals_modes: torch.Tensor) -> torch.Tensor:
        x = x + self.wavelet(self.norm1(x), U_modes, evals_modes)
        return x + self.ffn(self.norm2(x))


class SATCWGNO3D(StaticGraphModule):
    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int,
        forecast_horizon: int,
        channels: int,
        width: int,
        bands: int,
        depth: int = 10,
        n_scales: int = 4,
        n_modes: Optional[int] = None,
        dropout: float = 0.05,
        scale_aware: bool = True,
    ):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.width = int(width)
        self.tc = TimeCausalEncoder(history, channels, bands, scale_aware=scale_aware)
        K = min(int(n_modes or modes.shape[1]), int(modes.shape[1]))
        self.n_modes = K
        self.register_buffer("modes", modes[:, :K].float(), persistent=False)
        self.register_buffer("evals", evals[:K].float(), persistent=False)
        self.input_proj = nn.Linear(self.tc.out_channels + self.node_static.shape[1], width)
        self.blocks = nn.ModuleList([GraphWNOBlock3D(width, K, n_scales, dropout) for _ in range(depth)])
        self.decoder = build_mlp([width, width, forecast_horizon * channels], dropout=dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history, self.channels)
        B = context.shape[0]
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([self.tc(context), static], dim=-1)
        x = self.input_proj(x)
        U = self.modes.to(device=context.device, dtype=context.dtype)
        lam = self.evals.to(device=context.device, dtype=context.dtype)
        for block in self.blocks:
            x = block(x, U, lam)
        delta = self.decoder(x)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


class PlainGWNO3D(StaticGraphModule):
    def __init__(
        self,
        graph: GraphStatic3D,
        evals: torch.Tensor,
        modes: torch.Tensor,
        history: int,
        forecast_horizon: int,
        channels: int,
        width: int,
        depth: int = 10,
        n_scales: int = 4,
        n_modes: Optional[int] = None,
        dropout: float = 0.05,
    ):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        K = min(int(n_modes or modes.shape[1]), int(modes.shape[1]))
        self.register_buffer("modes", modes[:, :K].float(), persistent=False)
        self.register_buffer("evals", evals[:K].float(), persistent=False)
        self.input_proj = nn.Linear(history * channels + self.node_static.shape[1], width)
        self.blocks = nn.ModuleList([GraphWNOBlock3D(width, K, n_scales, dropout) for _ in range(depth)])
        self.decoder = build_mlp([width, width, forecast_horizon * channels], dropout=dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape
        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)
        x = self.input_proj(torch.cat([hist, static], dim=-1))
        U = self.modes.to(device=context.device, dtype=context.dtype)
        lam = self.evals.to(device=context.device, dtype=context.dtype)
        for block in self.blocks:
            x = block(x, U, lam)
        delta = self.decoder(x)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


# -----------------------------------------------------------------------------
# Graph FNO
# -----------------------------------------------------------------------------

class GraphFNOBlock3D(nn.Module):
    def __init__(self, width: int, n_modes: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.spectral_weight = nn.Parameter(torch.randn(n_modes, width, width) * (1.0 / np.sqrt(width)))
        self.local = nn.Linear(width, width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * width, width),
        )

    def forward(self, x: torch.Tensor, U_modes: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        coeff = torch.einsum("bnc,nk->bkc", z, U_modes)
        mixed = torch.einsum("bki,kio->bko", coeff, self.spectral_weight)
        spec = torch.einsum("bkc,nk->bnc", mixed, U_modes)
        x = x + spec + self.local(z)
        return x + self.ffn(self.norm2(x))


class GraphFNO3D(StaticGraphModule):
    def __init__(
        self,
        graph: GraphStatic3D,
        modes: torch.Tensor,
        history: int,
        forecast_horizon: int,
        channels: int,
        width: int,
        depth: int = 10,
        n_modes: Optional[int] = None,
        dropout: float = 0.05,
    ):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        K = min(int(n_modes or modes.shape[1]), int(modes.shape[1]))
        self.register_buffer("modes", modes[:, :K].float(), persistent=False)
        self.input_proj = nn.Linear(history * channels + self.node_static.shape[1], width)
        self.blocks = nn.ModuleList([GraphFNOBlock3D(width, K, dropout) for _ in range(depth)])
        self.decoder = build_mlp([width, width, forecast_horizon * channels], dropout=dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape
        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)
        x = self.input_proj(torch.cat([hist, static], dim=-1))
        U = self.modes.to(device=context.device, dtype=context.dtype)
        for block in self.blocks:
            x = block(x, U)
        delta = self.decoder(x)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


# -----------------------------------------------------------------------------
# MeshGraphNet
# -----------------------------------------------------------------------------

class EdgeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = build_mlp([2 * node_dim + edge_dim, hidden_dim, hidden_dim], dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return self.mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))


class NodeModel(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = build_mlp([node_dim + hidden_dim, hidden_dim, hidden_dim], dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_msg: torch.Tensor) -> torch.Tensor:
        _, dst = edge_index
        agg = segment_sum(edge_msg, dst, x.shape[0])
        return self.mlp(torch.cat([x, agg], dim=-1))


class MGNBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.edge_model = EdgeModel(hidden_dim, edge_dim, hidden_dim, dropout)
        self.node_model = NodeModel(hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        msg = self.edge_model(z, edge_index, edge_attr)
        return x + self.node_model(z, edge_index, msg)


class MeshGraphNetDirect3D(StaticGraphModule):
    def __init__(self, graph: GraphStatic3D, history: int, forecast_horizon: int, channels: int, width: int, depth: int = 10, dropout: float = 0.05):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.encoder = build_mlp([history * channels + self.node_static.shape[1], width, width], dropout=dropout, layer_norm=True)
        self.blocks = nn.ModuleList([MGNBlock(width, self.edge_attr_static.shape[1], dropout) for _ in range(depth)])
        self.decoder = build_mlp([width, width, forecast_horizon * channels], dropout=dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape
        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)
        x = self.encoder(torch.cat([hist, static], dim=-1).reshape(B * N, -1))
        ei, edge_attr = self.batched_graph(B, edge_attr=True)
        ei = ei.to(context.device)
        edge_attr = edge_attr.to(context.device, dtype=context.dtype)
        for block in self.blocks:
            x = block(x, ei, edge_attr)
        delta = self.decoder(x).reshape(B, N, -1)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


# -----------------------------------------------------------------------------
# GAT / GATv2 / GPS
# -----------------------------------------------------------------------------

def _require_pyg(name: str):
    if name == "GAT" and GATConv is None:
        raise ImportError("torch_geometric is required for GAT.")
    if name == "GATv2" and GATv2Conv is None:
        raise ImportError("torch_geometric is required for GATv2.")
    if name == "GPS" and GINEConv is None:
        raise ImportError("torch_geometric is required for GPS Transformer.")


class GATBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        _require_pyg("GAT")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.norm1 = nn.LayerNorm(width)
        self.attn = GATConv(width, width // heads, heads=heads, concat=True, add_self_loops=False, dropout=0.0)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = build_mlp([width, 2 * width, width])

    def forward(self, x, edge_index):
        x = x + self.attn(self.norm1(x), edge_index)
        return x + self.ffn(self.norm2(x))


class GATDirect3D(StaticGraphModule):
    def __init__(self, graph: GraphStatic3D, history: int, forecast_horizon: int, channels: int, width: int, depth: int = 3, heads: int = 4):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.encoder = build_mlp([history * channels + self.node_static.shape[1], width, width], layer_norm=True)
        self.blocks = nn.ModuleList([GATBlock(width, heads) for _ in range(depth)])
        self.decoder = build_mlp([width, width, forecast_horizon * channels])

    def forward(self, context):
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape
        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)
        x = self.encoder(torch.cat([hist, static], dim=-1).reshape(B * N, -1))
        ei, _ = self.batched_graph(B, edge_attr=False)
        ei = ei.to(context.device)
        for block in self.blocks:
            x = block(x, ei)
        delta = self.decoder(x).reshape(B, N, -1)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


class GATv2Block(nn.Module):
    def __init__(self, width: int, edge_dim: int, heads: int):
        super().__init__()
        _require_pyg("GATv2")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.norm1 = nn.LayerNorm(width)
        self.attn = GATv2Conv(width, width // heads, heads=heads, concat=True, edge_dim=edge_dim, add_self_loops=False, dropout=0.0, share_weights=False)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = build_mlp([width, 2 * width, width])

    def forward(self, x, edge_index, edge_attr):
        x = x + self.attn(self.norm1(x), edge_index, edge_attr=edge_attr)
        return x + self.ffn(self.norm2(x))


class GATv2Direct3D(StaticGraphModule):
    def __init__(self, graph: GraphStatic3D, history: int, forecast_horizon: int, channels: int, width: int, depth: int = 3, heads: int = 4):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.encoder = build_mlp([history * channels + self.node_static.shape[1], width, width], layer_norm=True)
        self.blocks = nn.ModuleList([GATv2Block(width, self.edge_attr_static.shape[1], heads) for _ in range(depth)])
        self.decoder = build_mlp([width, width, forecast_horizon * channels])

    def forward(self, context):
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape
        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)
        x = self.encoder(torch.cat([hist, static], dim=-1).reshape(B * N, -1))
        ei, edge_attr = self.batched_graph(B, edge_attr=True)
        ei = ei.to(context.device)
        edge_attr = edge_attr.to(context.device, dtype=context.dtype)
        for block in self.blocks:
            x = block(x, ei, edge_attr)
        delta = self.decoder(x).reshape(B, N, -1)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


class GPSLayer(nn.Module):
    def __init__(self, width: int, edge_dim: int, heads: int):
        super().__init__()
        _require_pyg("GPS")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.local_norm = nn.LayerNorm(width)
        self.local = GINEConv(
            nn=nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)),
            edge_dim=edge_dim,
            train_eps=True,
        )
        self.global_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = build_mlp([width, 2 * width, width])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        B, N, W = x.shape
        flat = x.reshape(B * N, W)
        local = self.local(self.local_norm(flat), edge_index, edge_attr=edge_attr).reshape(B, N, W)
        x = x + local
        z = self.global_norm(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x))


class GPSDirect3D(StaticGraphModule):
    def __init__(self, graph: GraphStatic3D, history: int, forecast_horizon: int, channels: int, width: int, depth: int = 3, heads: int = 4):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.encoder = build_mlp([history * channels + self.node_static.shape[1], width, width], layer_norm=True)
        self.blocks = nn.ModuleList([GPSLayer(width, self.edge_attr_static.shape[1], heads) for _ in range(depth)])
        self.decoder = build_mlp([width, width, forecast_horizon * channels])

    def forward(self, context):
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape
        hist = context.permute(0, 2, 1, 3).reshape(B, N, H * C)
        static = self.node_static.to(device=context.device, dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)
        x = self.encoder(torch.cat([hist, static], dim=-1))
        ei, edge_attr = self.batched_graph(B, edge_attr=True)
        ei = ei.to(context.device)
        edge_attr = edge_attr.to(context.device, dtype=context.dtype)
        for block in self.blocks:
            x = block(x, ei, edge_attr)
        delta = self.decoder(x.reshape(B * N, -1)).reshape(B, N, -1)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


# -----------------------------------------------------------------------------
# Recurrent baselines
# -----------------------------------------------------------------------------

class RNNDirect3D(nn.Module):
    def __init__(self, history: int, forecast_horizon: int, channels: int, hidden_size: int, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.rnn = nn.GRU(channels, hidden_size, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.decoder = build_mlp([hidden_size, hidden_size, forecast_horizon * channels])

    def forward(self, context):
        if context.dim() != 4:
            raise ValueError(f"context must have shape (B,H,N,C), got {tuple(context.shape)}")
        B, H, N, C = context.shape
        x = context.permute(0, 2, 1, 3).reshape(B * N, H, C)
        _, h = self.rnn(x)
        delta = self.decoder(h[-1]).reshape(B, N, -1)
        return add_residual_from_last(context, delta, self.forecast_horizon, self.channels)


class RNNGNNFusionDirect3D(StaticGraphModule):
    def __init__(self, graph: GraphStatic3D, history: int, forecast_horizon: int, channels: int, hidden_size: int, dropout: float = 0.2):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.channels = int(channels)
        self.rnn = RNNDirect3D(history, forecast_horizon, channels, hidden_size, layers=3, dropout=dropout)
        graph_hidden = max(16, hidden_size // 2)
        self.message_net = nn.Sequential(
            nn.Linear(2 * history * channels, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, graph_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(graph_hidden, forecast_horizon * channels),
        )
        self.out_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, context):
        self.validate_context(context, self.history, self.channels)
        B, H, N, C = context.shape
        rnn_pred = self.rnn(context)
        hist = context.permute(0, 2, 1, 3).reshape(B * N, H * C)
        ei, _ = self.batched_graph(B, edge_attr=False)
        ei = ei.to(context.device)
        src, dst = ei
        msg_in = torch.cat([hist[src], hist[src] - hist[dst]], dim=-1)
        weights = self.edge_weight_static.to(context.device, dtype=context.dtype).repeat(B).unsqueeze(-1)
        messages = self.message_net(msg_in) * weights
        agg = segment_sum(messages, dst, B * N)
        denom = hist.new_zeros(B * N, 1)
        denom.index_add_(0, dst, weights)
        graph_delta = agg / denom.clamp_min(1e-8)
        graph_pred = add_residual_from_last(context, graph_delta.reshape(B, N, -1), self.forecast_horizon, self.channels)
        gate = torch.sigmoid(self.out_param)
        return gate * rnn_pred + (1.0 - gate) * graph_pred


# -----------------------------------------------------------------------------
# Width matching / spec construction
# -----------------------------------------------------------------------------

def _nearest_width(factory: Callable[[int], nn.Module], target_parameters: int, candidates: Sequence[int]) -> Tuple[int, int]:
    choices = []
    for width in candidates:
        try:
            model = factory(int(width))
            params = count_parameters(model)
            choices.append((abs(params - target_parameters), int(width), int(params)))
            del model
        except Exception:
            # Some candidates violate head divisibility or optional package availability.
            continue
    if not choices:
        raise RuntimeError("No valid width candidates for factory.")
    _, width, params = min(choices)
    return width, params


def _make_spec(name: str, label: str, factory: Callable[[], nn.Module]) -> ModelSpec:
    model = factory()
    params = count_parameters(model)
    del model
    return ModelSpec(name=name, label=label, parameters=params, factory=factory)


def build_pflotran_h10_model_specs(
    graph: GraphStatic3D,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    history: int = 10,
    forecast_horizon: int = 10,
    channels: int = 2,
    target_parameters: int = 1_000_000,
    sa_band_counts: Tuple[int, ...] = (3, 5),
    operator_depth: int = 10,
    message_depth: int = 10,
    attention_depth: int = 3,
    n_modes: int = 96,
    dropout: float = 0.05,
) -> Tuple[Dict[str, ModelSpec], Dict[str, Dict[str, int]]]:
    specs: Dict[str, ModelSpec] = {}
    matched: Dict[str, Dict[str, int]] = {}

    # SA TC-WGNO variants.
    for bands in sa_band_counts:
        name = pflotran_sa_name(bands)
        def fac(width, bands=bands):
            return SATCWGNO3D(graph, eigenvalues, modes, history, forecast_horizon, channels, width, bands,
                              depth=operator_depth, n_scales=4, n_modes=n_modes, dropout=dropout, scale_aware=True)
        width, params = _nearest_width(fac, target_parameters, range(64, 257, 4))
        matched[name] = {"width": width, "parameters": params, "bands": int(bands)}
        specs[name] = _make_spec(name, f"SA TC-WGNO (K={bands})", lambda width=width, bands=bands: fac(width, bands))

    # Operator controls.
    name = pflotran_plain_gwno_name()
    def fac_gwno(width):
        return PlainGWNO3D(graph, eigenvalues, modes, history, forecast_horizon, channels, width,
                           depth=operator_depth, n_scales=4, n_modes=n_modes, dropout=dropout)
    width, params = _nearest_width(fac_gwno, target_parameters, range(64, 257, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = _make_spec(name, "Plain Graph WNO", lambda width=width: fac_gwno(width))

    name = pflotran_graph_fno_name()
    def fac_fno(width):
        return GraphFNO3D(graph, modes, history, forecast_horizon, channels, width,
                          depth=operator_depth, n_modes=n_modes, dropout=dropout)
    width, params = _nearest_width(fac_fno, target_parameters, range(16, 129, 2))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = _make_spec(name, "Graph FNO", lambda width=width: fac_fno(width))

    # Graph baselines.
    name = pflotran_mesh_name()
    def fac_mesh(width):
        return MeshGraphNetDirect3D(graph, history, forecast_horizon, channels, width, depth=message_depth, dropout=dropout)
    width, params = _nearest_width(fac_mesh, target_parameters, range(64, 257, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = _make_spec(name, "MeshGraphNet", lambda width=width: fac_mesh(width))

    name = pflotran_gat_name()
    def fac_gat(width):
        return GATDirect3D(graph, history, forecast_horizon, channels, width, depth=attention_depth, heads=4)
    width, params = _nearest_width(fac_gat, target_parameters, range(64, 513, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = _make_spec(name, "GAT", lambda width=width: fac_gat(width))

    name = pflotran_gatv2_name()
    def fac_gatv2(width):
        return GATv2Direct3D(graph, history, forecast_horizon, channels, width, depth=attention_depth, heads=4)
    width, params = _nearest_width(fac_gatv2, target_parameters, range(64, 513, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = _make_spec(name, "GATv2", lambda width=width: fac_gatv2(width))

    name = pflotran_gps_name()
    def fac_gps(width):
        return GPSDirect3D(graph, history, forecast_horizon, channels, width, depth=attention_depth, heads=4)
    width, params = _nearest_width(fac_gps, target_parameters, range(64, 513, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = _make_spec(name, "GPS Transformer", lambda width=width: fac_gps(width))

    # Recurrent baselines.
    name = pflotran_rnn_name()
    def fac_rnn(width):
        return RNNDirect3D(history, forecast_horizon, channels, width, layers=3, dropout=0.1)
    width, params = _nearest_width(fac_rnn, target_parameters, range(64, 769, 8))
    matched[name] = {"hidden_size": width, "parameters": params}
    specs[name] = _make_spec(name, "RNN", lambda width=width: fac_rnn(width))

    name = pflotran_rnn_gnn_fusion_name()
    def fac_fusion(width):
        return RNNGNNFusionDirect3D(graph, history, forecast_horizon, channels, width, dropout=0.2)
    width, params = _nearest_width(fac_fusion, target_parameters, range(64, 769, 8))
    matched[name] = {"hidden_size": width, "parameters": params}
    specs[name] = _make_spec(name, "RNN-GNN Fusion", lambda width=width: fac_fusion(width))

    return specs, matched


def default_model_order(sa_band_counts: Tuple[int, ...] = (3, 5)) -> Sequence[str]:
    return [
        *(pflotran_sa_name(k) for k in sa_band_counts),
        pflotran_graph_fno_name(),
        pflotran_plain_gwno_name(),
        pflotran_mesh_name(),
        pflotran_gat_name(),
        pflotran_gatv2_name(),
        pflotran_gps_name(),
        pflotran_rnn_gnn_fusion_name(),
        pflotran_rnn_name(),
    ]
