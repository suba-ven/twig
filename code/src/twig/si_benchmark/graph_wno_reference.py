from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNOConfig:
    in_channels: int
    out_channels: int
    width: int = 128
    depth: int = 10
    edge_dim: int = 4
    mlp_depth: int = 2
    dropout: float = 0.0


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
    layers = []
    d = in_dim
    for _ in range(max(depth - 1, 1)):
        layers += [nn.Linear(d, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def canonicalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.dim() != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.shape[0] == 2:
        ei = edge_index
    elif edge_index.shape[1] == 2:
        ei = edge_index.t()
    else:
        raise ValueError(f"edge_index must have shape (2,E) or (E,2), got {tuple(edge_index.shape)}")
    return ei.contiguous().long()


def make_batched_edge_index(edge_index: torch.Tensor, num_nodes: int, batch_size: int, device=None):
    ei = canonicalize_edge_index(edge_index)
    if device is None:
        device = ei.device
    ei = ei.to(device)
    E = ei.shape[1]
    offsets = (torch.arange(batch_size, device=device, dtype=ei.dtype) * num_nodes).view(batch_size, 1, 1)
    base = ei.t().unsqueeze(0)
    batched = base + offsets
    batched = batched.reshape(batch_size * E, 2).t().contiguous()
    return batched


def build_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    ei = canonicalize_edge_index(edge_index)
    src, dst = ei
    rel = pos[dst] - pos[src]
    dist = torch.norm(rel, dim=-1, keepdim=True)
    edge_attr = torch.cat([rel, dist], dim=-1)
    return edge_attr


class EdgeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, mlp_depth: int, dropout: float):
        super().__init__()
        self.mlp = build_mlp(2 * node_dim + edge_dim, hidden_dim, hidden_dim, mlp_depth, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        inp = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.mlp(inp)


class NodeModel(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, mlp_depth: int, dropout: float):
        super().__init__()
        self.mlp = build_mlp(node_dim + hidden_dim, hidden_dim, hidden_dim, mlp_depth, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_msg: torch.Tensor) -> torch.Tensor:
        _, dst = edge_index
        num_nodes = x.shape[0]
        hidden_dim = edge_msg.shape[-1]

        agg = torch.zeros(
            num_nodes,
            hidden_dim,
            device=edge_msg.device,
            dtype=edge_msg.dtype,
        )
        agg.index_add_(0, dst, edge_msg)

        x_in = x.to(agg.dtype) if x.dtype != agg.dtype else x
        inp = torch.cat([x_in, agg], dim=-1)
        out = self.mlp(inp)

        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out


class MGNBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, mlp_depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.edge_model = EdgeModel(hidden_dim, edge_dim, hidden_dim, mlp_depth, dropout)
        self.node_model = NodeModel(hidden_dim, hidden_dim, mlp_depth, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        ei = canonicalize_edge_index(edge_index)
        edge_msg = self.edge_model(x, ei, edge_attr)
        dx = self.node_model(x, ei, edge_msg)
        return x + dx


class MeshGraphNet3D(nn.Module):
    def __init__(self, cfg: GNOConfig):
        super().__init__()
        self.cfg = cfg

        self.node_encoder = build_mlp(
            in_dim=cfg.in_channels,
            hidden_dim=cfg.width,
            out_dim=cfg.width,
            depth=cfg.mlp_depth,
            dropout=cfg.dropout,
        )

        self.blocks = nn.ModuleList([
            MGNBlock(cfg.width, cfg.edge_dim, cfg.mlp_depth, cfg.dropout)
            for _ in range(cfg.depth)
        ])

        self.node_decoder = build_mlp(
            in_dim=cfg.width,
            hidden_dim=cfg.width,
            out_dim=cfg.out_channels,
            depth=cfg.mlp_depth,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        unbatched = (x.dim() == 2)
        if unbatched:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(f"x must have shape (B,N,C) or (N,C), got {tuple(x.shape)}")

        B, N, _ = x.shape

        if pos.dim() == 2:
            if pos.shape != (N, 3):
                raise ValueError(f"Expected pos shape (N,3), got {tuple(pos.shape)}")
            pos_batched = pos.unsqueeze(0).expand(B, N, 3)
        elif pos.dim() == 3:
            if pos.shape[0] != B or pos.shape[1] != N or pos.shape[2] != 3:
                raise ValueError(f"Expected pos shape (B,N,3), got {tuple(pos.shape)}")
            pos_batched = pos
        else:
            raise ValueError(f"pos must have shape (N,3) or (B,N,3), got {tuple(pos.shape)}")

        x = x.reshape(B * N, -1)
        pos_flat = pos_batched.reshape(B * N, 3)

        batched_edge_index = make_batched_edge_index(edge_index, N, B, device=x.device)
        edge_attr = build_edge_features(pos_flat, batched_edge_index)

        x = self.node_encoder(x)
        for block in self.blocks:
            x = block(x, batched_edge_index, edge_attr)
        x = self.node_decoder(x)
        x = x.reshape(B, N, -1)

        if unbatched:
            x = x.squeeze(0)

        return x


class GraphWaveletPositionalEncoding3D(nn.Module):
    """
    Coordinate wavelet-style positional encoding.
    Uses localized bands instead of plain Fourier sin/cos bands.
    """
    def __init__(
        self,
        num_scales: int = 6,
        include_input: bool = True,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.include_input = bool(include_input)
        self.max_freq = float(max_freq)

        if self.num_scales < 1:
            raise ValueError("num_scales must be >= 1")

        scales = torch.logspace(
            start=0.0,
            end=torch.log10(torch.tensor(self.max_freq)).item(),
            steps=self.num_scales,
        )
        self.register_buffer("scales", scales, persistent=False)

    @property
    def out_dim(self) -> int:
        base = 3 if self.include_input else 0
        # for each coordinate and each scale: sin, cos, gaussian envelope
        return base + 3 * self.num_scales * 3

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if pos.dim() != 3 or pos.shape[-1] != 3:
            raise ValueError(f"Expected pos shape (B,N,3) or (N,3), got {tuple(pos.shape)}")

        B, N, _ = pos.shape
        scales = self.scales.to(device=pos.device, dtype=pos.dtype)

        x = pos.unsqueeze(-1) * scales.view(1, 1, 1, -1) * (2.0 * torch.pi)
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        gauss_x = torch.exp(-0.5 * x.pow(2))

        feats = []
        if self.include_input:
            feats.append(pos)
        feats.append(sin_x.reshape(B, N, -1))
        feats.append(cos_x.reshape(B, N, -1))
        feats.append(gauss_x.reshape(B, N, -1))
        return torch.cat(feats, dim=-1)


class _GraphWNOBase3D(nn.Module):
    def __init__(
        self,
        cfg: GNOConfig,
        evals: torch.Tensor,
        U: torch.Tensor,
        n_modes: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg

        if U.dim() != 2:
            raise ValueError(f"U must have shape (N,K), got {tuple(U.shape)}")
        if evals.dim() != 1:
            raise ValueError(f"evals must have shape (K,), got {tuple(evals.shape)}")
        if U.shape[1] != evals.shape[0]:
            raise ValueError(f"U has K={U.shape[1]} but evals has K={evals.shape[0]}")

        N, K = U.shape
        self.N_graph = int(N)
        self.K_total = int(K)
        self.n_modes = int(min(n_modes if n_modes is not None else K, K))

        self.register_buffer("U_full", U.float(), persistent=False)
        self.register_buffer("evals_full", evals.float(), persistent=False)

    def _get_wavelet_basis(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if pos.dim() == 2:
            N = pos.shape[0]
        elif pos.dim() == 3:
            N = pos.shape[1]
        else:
            raise ValueError(f"pos must have shape (N,3) or (B,N,3), got {tuple(pos.shape)}")

        if N != self.N_graph:
            raise ValueError(f"Basis was built for N={self.N_graph} nodes, but input has N={N}")

        return self.U_full[:, :self.n_modes], self.evals_full[:self.n_modes]


class GraphWaveletOperator(nn.Module):
    """
    Spectral graph wavelet operator.
    Uses learned combinations of wavelet responses across scales.
    """
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
        # evals: (K,)
        device = evals.device
        dtype = evals.dtype
        scales = torch.exp(self.scale_logits).to(device=device, dtype=dtype)  # (S,)
        lam = evals.view(1, -1)  # (1,K)

        # Mexican-hat / heat-derivative style graph wavelets
        wavelets = lam * torch.exp(-scales.view(-1, 1) * lam)  # (S,K)

        # add one low-pass branch implicitly through lowpass_gate later
        return wavelets

    def forward(self, x: torch.Tensor, U_modes: torch.Tensor, evals_modes: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C), U_modes: (N,K), evals_modes: (K,)
        coeff = torch.einsum("bnc,nk->bkc", x, U_modes)  # (B,K,C)
        coeff = self.spec_linear(coeff)

        bank = self._wavelet_bank(evals_modes)  # (S,K)

        wavelet_out = 0.0
        for s in range(self.n_scales):
            filt = bank[s].unsqueeze(-1) * self.scale_mix[s].unsqueeze(0)  # (K,C)
            wavelet_out = wavelet_out + coeff * filt.unsqueeze(0)

        lowpass_out = coeff * self.lowpass_gate.unsqueeze(0)

        coeff_out = wavelet_out + lowpass_out
        out = torch.einsum("bkc,nk->bnc", coeff_out, U_modes)
        return out


class GraphWNOBlock3D(nn.Module):
    def __init__(self, width: int, n_modes: int, n_scales: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.wavelet = GraphWaveletOperator(width=width, n_modes=n_modes, n_scales=n_scales)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * width, width),
        )

    def forward(self, x: torch.Tensor, U_modes: torch.Tensor, evals_modes: torch.Tensor) -> torch.Tensor:
        x = x + self.wavelet(self.norm1(x), U_modes, evals_modes)
        x = x + self.ffn(self.norm2(x))
        return x


class HybridGraphWNOBlock3D(nn.Module):
    def __init__(
        self,
        width: int,
        n_modes: int,
        n_scales: int = 4,
        edge_dim: int = 4,
        mlp_depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

        self.wavelet = GraphWaveletOperator(width=width, n_modes=n_modes, n_scales=n_scales)
        self.graph_block = MGNBlock(
            hidden_dim=width,
            edge_dim=edge_dim,
            mlp_depth=mlp_depth,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * width, width),
        )

    def forward(
        self,
        x_flat: torch.Tensor,
        x_batched: torch.Tensor,
        U_modes: torch.Tensor,
        evals_modes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        B: int,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_batched = x_batched + self.wavelet(self.norm1(x_batched), U_modes, evals_modes)

        x_flat = x_batched.reshape(B * N, -1)
        x_flat = self.graph_block(self.norm2(x_flat), edge_index, edge_attr)

        x_batched = x_flat.reshape(B, N, -1)
        x_batched = x_batched + self.ffn(self.norm3(x_batched))
        x_flat = x_batched.reshape(B * N, -1)
        return x_flat, x_batched


class BaseGraphWNO3D(_GraphWNOBase3D):
    def __init__(
        self,
        cfg: GNOConfig,
        evals: torch.Tensor,
        U: torch.Tensor,
        n_modes: Optional[int] = None,
        n_scales: int = 4,
    ):
        super().__init__(cfg=cfg, evals=evals, U=U, n_modes=n_modes)

        self.input_proj = nn.Linear(cfg.in_channels, cfg.width)
        self.blocks = nn.ModuleList([
            GraphWNOBlock3D(
                width=cfg.width,
                n_modes=self.n_modes,
                n_scales=n_scales,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.width, cfg.width),
            nn.GELU(),
            nn.Linear(cfg.width, cfg.out_channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        unbatched = (x.dim() == 2)
        if unbatched:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(f"x must have shape (B,N,C) or (N,C), got {tuple(x.shape)}")

        U_modes, evals_modes = self._get_wavelet_basis(pos)
        U_modes = U_modes.to(device=x.device, dtype=x.dtype)
        evals_modes = evals_modes.to(device=x.device, dtype=x.dtype)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, U_modes, evals_modes)
        x = self.output_proj(x)

        if unbatched:
            x = x.squeeze(0)
        return x


class PositionalGraphWNO3D(_GraphWNOBase3D):
    def __init__(
        self,
        cfg: GNOConfig,
        evals: torch.Tensor,
        U: torch.Tensor,
        n_modes: Optional[int] = None,
        n_scales: int = 4,
        num_pos_scales: int = 6,
        include_input_pos: bool = True,
        max_pos_freq: float = 10.0,
    ):
        super().__init__(cfg=cfg, evals=evals, U=U, n_modes=n_modes)

        self.pos_encoder = GraphWaveletPositionalEncoding3D(
            num_scales=num_pos_scales,
            include_input=include_input_pos,
            max_freq=max_pos_freq,
        )

        self.input_proj = nn.Linear(cfg.in_channels + self.pos_encoder.out_dim, cfg.width)
        self.blocks = nn.ModuleList([
            GraphWNOBlock3D(
                width=cfg.width,
                n_modes=self.n_modes,
                n_scales=n_scales,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.width, cfg.width),
            nn.GELU(),
            nn.Linear(cfg.width, cfg.out_channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        unbatched = (x.dim() == 2)
        if unbatched:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(f"x must have shape (B,N,C) or (N,C), got {tuple(x.shape)}")

        B, N, _ = x.shape

        if pos.dim() == 2:
            if pos.shape != (N, 3):
                raise ValueError(f"Expected pos shape (N,3), got {tuple(pos.shape)}")
            pos_batched = pos.unsqueeze(0).expand(B, N, 3)
        elif pos.dim() == 3:
            if pos.shape[1] != N or pos.shape[2] != 3:
                raise ValueError(f"Expected pos shape (B,N,3), got {tuple(pos.shape)}")
            pos_batched = pos
        else:
            raise ValueError(f"pos must have shape (N,3) or (B,N,3), got {tuple(pos.shape)}")

        U_modes, evals_modes = self._get_wavelet_basis(pos_batched)
        U_modes = U_modes.to(device=x.device, dtype=x.dtype)
        evals_modes = evals_modes.to(device=x.device, dtype=x.dtype)

        pos_feat = self.pos_encoder(pos_batched).to(dtype=x.dtype)
        x = torch.cat([x, pos_feat], dim=-1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, U_modes, evals_modes)

        x = self.output_proj(x)

        if unbatched:
            x = x.squeeze(0)
        return x


class HybridGraphWNO3D(_GraphWNOBase3D):
    def __init__(
        self,
        cfg: GNOConfig,
        evals: torch.Tensor,
        U: torch.Tensor,
        n_modes: Optional[int] = None,
        n_scales: int = 4,
        num_pos_scales: int = 6,
        include_input_pos: bool = True,
        max_pos_freq: float = 10.0,
    ):
        super().__init__(cfg=cfg, evals=evals, U=U, n_modes=n_modes)

        self.pos_encoder = GraphWaveletPositionalEncoding3D(
            num_scales=num_pos_scales,
            include_input=include_input_pos,
            max_freq=max_pos_freq,
        )

        self.input_proj = nn.Linear(cfg.in_channels + self.pos_encoder.out_dim, cfg.width)

        self.blocks = nn.ModuleList([
            HybridGraphWNOBlock3D(
                width=cfg.width,
                n_modes=self.n_modes,
                n_scales=n_scales,
                edge_dim=cfg.edge_dim,
                mlp_depth=cfg.mlp_depth,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(cfg.width, cfg.width),
            nn.GELU(),
            nn.Linear(cfg.width, cfg.out_channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        unbatched = (x.dim() == 2)
        if unbatched:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(f"x must have shape (B,N,C) or (N,C), got {tuple(x.shape)}")

        B, N, _ = x.shape

        if pos.dim() == 2:
            if pos.shape != (N, 3):
                raise ValueError(f"Expected pos shape (N,3), got {tuple(pos.shape)}")
            pos_batched = pos.unsqueeze(0).expand(B, N, 3)
        elif pos.dim() == 3:
            if pos.shape[0] != B or pos.shape[1] != N or pos.shape[2] != 3:
                raise ValueError(f"Expected pos shape (B,N,3), got {tuple(pos.shape)}")
            pos_batched = pos
        else:
            raise ValueError(f"pos must have shape (N,3) or (B,N,3), got {tuple(pos.shape)}")

        U_modes, evals_modes = self._get_wavelet_basis(pos_batched)
        U_modes = U_modes.to(device=x.device, dtype=x.dtype)
        evals_modes = evals_modes.to(device=x.device, dtype=x.dtype)

        pos_feat = self.pos_encoder(pos_batched).to(dtype=x.dtype)

        x_batched = torch.cat([x, pos_feat], dim=-1)
        x_batched = self.input_proj(x_batched)

        x_flat = x_batched.reshape(B * N, -1)
        pos_flat = pos_batched.reshape(B * N, 3)

        batched_edge_index = make_batched_edge_index(edge_index, N, B, device=x.device)
        edge_attr = build_edge_features(pos_flat, batched_edge_index)

        for block in self.blocks:
            x_flat, x_batched = block(
                x_flat=x_flat,
                x_batched=x_batched,
                U_modes=U_modes,
                evals_modes=evals_modes,
                edge_index=batched_edge_index,
                edge_attr=edge_attr,
                B=B,
                N=N,
            )

        x = self.output_proj(x_batched)

        if unbatched:
            x = x.squeeze(0)
        return x