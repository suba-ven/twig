from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv, GATv2Conv, GINEConv
except ModuleNotFoundError as exc:
    raise ImportError(
        "The direct GAT/GATv2/GPS baselines require torch_geometric. "
        "Use the same PyG environment as the earlier benchmark."
    ) from exc

from .direct14_config import Direct14Config
from .direct14_models import (
    MLP,
    StaticGraphModule,
)
from .direct14_registry import (
    DirectModelSpec,
    build_direct14_model_specs,
    count_parameters,
    direct_mesh_name,
    direct_sa_name,
)
from .graph import GraphStatic, segment_sum


def direct_rnn_name() -> str:
    return "RNN-D14"


def direct_rnn_gnn_fusion_name() -> str:
    return "RNN-GNN-Fusion-D14"


def direct_gat_name() -> str:
    return "GAT-D14"


def direct_gatv2_name() -> str:
    return "GATv2-D14"


def direct_gps_name() -> str:
    return "GPS-Transformer-D14"


class DirectRNNBaseline(nn.Module):
    """Node-wise GRU with a direct 14-frame output head."""

    def __init__(
        self,
        history: int,
        forecast_horizon: int,
        hidden_size: int,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.hidden_size = int(hidden_size)
        self.layers = int(layers)

        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=float(dropout) if self.layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, self.forecast_horizon),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 4 or context.shape[1] != self.history:
            raise ValueError(
                f"Expected [B,{self.history},N,1], got "
                f"{tuple(context.shape)}."
            )

        if context.shape[-1] != 1:
            raise ValueError("DirectRNNBaseline requires one scalar channel.")

        batch, _, nodes, _ = context.shape

        sequence = (
            context.permute(0, 2, 1, 3)
            .reshape(batch * nodes, self.history, 1)
            .contiguous()
        )

        _, hidden = self.rnn(sequence)
        output = self.head(hidden[-1])

        return (
            output.reshape(batch, nodes, self.forecast_horizon)
            .permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )


class DirectRNNGNNFusionBaseline(StaticGraphModule):
    """Direct RNN output fused with distance-weighted graph propagation."""

    def __init__(
        self,
        graph: GraphStatic,
        history: int,
        forecast_horizon: int,
        hidden_size: int,
        dropout: float = 0.2,
    ):
        super().__init__(graph)

        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)

        self.rnn = DirectRNNBaseline(
            history=self.history,
            forecast_horizon=self.forecast_horizon,
            hidden_size=int(hidden_size),
            layers=3,
            dropout=dropout,
        )

        graph_hidden = max(8, int(hidden_size) // 2)

        self.message_net = nn.Sequential(
            nn.Linear(2 * self.history, int(hidden_size)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), graph_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(graph_hidden, self.forecast_horizon),
        )

        self.out_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history)

        batch = context.shape[0]
        rnn_prediction = self.rnn(context)

        history = context.squeeze(-1).permute(0, 2, 1).reshape(
            batch * self.n_nodes,
            self.history,
        )

        edge_index, _ = self.batched_graph(
            batch,
            edge_attr=False,
        )
        src, dst = edge_index

        message_input = torch.cat(
            [
                history[src],
                history[src] - history[dst],
            ],
            dim=-1,
        )

        weights = self.edge_weight_static.repeat(batch).unsqueeze(-1)

        messages = self.message_net(message_input) * weights
        aggregate = segment_sum(
            messages,
            dst,
            batch * self.n_nodes,
        )

        normalizer = history.new_zeros(batch * self.n_nodes, 1)
        normalizer.index_add_(0, dst, weights)

        graph_prediction = aggregate / normalizer.clamp_min(1e-8)

        graph_prediction = (
            graph_prediction.reshape(
                batch,
                self.n_nodes,
                self.forecast_horizon,
            )
            .permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )

        gate = torch.sigmoid(self.out_param)

        return (
            gate * rnn_prediction
            + (1.0 - gate) * graph_prediction
        )


class DirectGATBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()

        if width % heads:
            raise ValueError("width must be divisible by heads.")

        self.pre_norm = nn.LayerNorm(width)

        self.attention = GATConv(
            in_channels=width,
            out_channels=width // heads,
            heads=heads,
            concat=True,
            dropout=0.0,
            add_self_loops=False,
        )

        self.post_norm = nn.LayerNorm(width)
        self.ffn = MLP([width, 2 * width, width])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.pre_norm(x),
            edge_index,
        )
        return x + self.ffn(self.post_norm(x))


class DirectGATBaseline(StaticGraphModule):
    def __init__(
        self,
        graph: GraphStatic,
        history: int,
        forecast_horizon: int,
        width: int,
        layers: int = 3,
        heads: int = 4,
    ):
        super().__init__(graph)

        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)

        self.encoder = MLP(
            [
                self.history + self.node_static.shape[1],
                width,
                width,
            ],
            layer_norm=True,
        )

        self.blocks = nn.ModuleList(
            [
                DirectGATBlock(width, heads)
                for _ in range(int(layers))
            ]
        )

        self.decoder = MLP(
            [width, width, self.forecast_horizon]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history)

        batch = context.shape[0]
        history = context.squeeze(-1).permute(0, 2, 1)

        static = self.node_static.unsqueeze(0).expand(
            batch,
            -1,
            -1,
        )

        x = self.encoder(
            torch.cat([history, static], dim=-1).reshape(
                batch * self.n_nodes,
                -1,
            )
        )

        edge_index, _ = self.batched_graph(
            batch,
            edge_attr=False,
        )

        for block in self.blocks:
            x = block(x, edge_index)

        delta = self.decoder(x).reshape(
            batch,
            self.n_nodes,
            self.forecast_horizon,
        )

        delta = (
            delta.permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )

        return context[:, -1:].expand_as(delta) + delta


class DirectGATv2Block(nn.Module):
    def __init__(
        self,
        width: int,
        edge_dim: int,
        heads: int,
    ):
        super().__init__()

        if width % heads:
            raise ValueError("width must be divisible by heads.")

        self.pre_norm = nn.LayerNorm(width)

        self.attention = GATv2Conv(
            in_channels=width,
            out_channels=width // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False,
            dropout=0.0,
            share_weights=False,
        )

        self.post_norm = nn.LayerNorm(width)
        self.ffn = MLP([width, 2 * width, width])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.pre_norm(x),
            edge_index,
            edge_attr=edge_attr,
        )
        return x + self.ffn(self.post_norm(x))


class DirectGATv2Baseline(StaticGraphModule):
    def __init__(
        self,
        graph: GraphStatic,
        history: int,
        forecast_horizon: int,
        width: int,
        layers: int = 3,
        heads: int = 4,
    ):
        super().__init__(graph)

        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)

        self.encoder = MLP(
            [
                self.history + self.node_static.shape[1],
                width,
                width,
            ],
            layer_norm=True,
        )

        self.blocks = nn.ModuleList(
            [
                DirectGATv2Block(
                    width,
                    self.edge_attr_static.shape[1],
                    heads,
                )
                for _ in range(int(layers))
            ]
        )

        self.decoder = MLP(
            [width, width, self.forecast_horizon]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history)

        batch = context.shape[0]
        history = context.squeeze(-1).permute(0, 2, 1)

        static = self.node_static.unsqueeze(0).expand(
            batch,
            -1,
            -1,
        )

        x = self.encoder(
            torch.cat([history, static], dim=-1).reshape(
                batch * self.n_nodes,
                -1,
            )
        )

        edge_index, edge_attr = self.batched_graph(
            batch,
            edge_attr=True,
        )

        if edge_attr is None:
            raise RuntimeError(
                "DirectGATv2Baseline requires edge attributes."
            )

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        delta = self.decoder(x).reshape(
            batch,
            self.n_nodes,
            self.forecast_horizon,
        )

        delta = (
            delta.permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )

        return context[:, -1:].expand_as(delta) + delta


class DirectGPSLayer(nn.Module):
    def __init__(
        self,
        width: int,
        edge_dim: int,
        heads: int,
    ):
        super().__init__()

        if width % heads:
            raise ValueError("width must be divisible by heads.")

        self.local_norm = nn.LayerNorm(width)

        # Must use an nn.Sequential beginning with nn.Linear so that
        # torch_geometric.nn.GINEConv can infer in_channels=width.
        self.local = GINEConv(
            nn=nn.Sequential(
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
            ),
            edge_dim=edge_dim,
            train_eps=True,
        )

        self.global_norm = nn.LayerNorm(width)

        self.global_attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = MLP([width, 2 * width, width])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, width = x.shape

        flat = x.reshape(batch * nodes, width)

        local = self.local(
            self.local_norm(flat),
            edge_index,
            edge_attr=edge_attr,
        )

        x = (flat + local).reshape(batch, nodes, width)

        attention_input = self.global_norm(x)

        attention_output, _ = self.global_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )

        x = x + attention_output

        return x + self.ffn(self.ffn_norm(x))


class DirectGPSGraphTransformerBaseline(StaticGraphModule):
    def __init__(
        self,
        graph: GraphStatic,
        history: int,
        forecast_horizon: int,
        width: int,
        layers: int = 3,
        heads: int = 4,
    ):
        super().__init__(graph)

        if width % heads:
            raise ValueError("width must be divisible by heads.")

        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)

        self.encoder = MLP(
            [
                self.history + self.node_static.shape[1],
                width,
                width,
            ],
            layer_norm=True,
        )

        self.layers = nn.ModuleList(
            [
                DirectGPSLayer(
                    width=width,
                    edge_dim=self.edge_attr_static.shape[1],
                    heads=heads,
                )
                for _ in range(int(layers))
            ]
        )

        self.decoder = MLP(
            [width, width, self.forecast_horizon]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history)

        batch = context.shape[0]
        history = context.squeeze(-1).permute(0, 2, 1)

        static = self.node_static.unsqueeze(0).expand(
            batch,
            -1,
            -1,
        )

        x = self.encoder(
            torch.cat([history, static], dim=-1)
        )

        edge_index, edge_attr = self.batched_graph(
            batch,
            edge_attr=True,
        )

        if edge_attr is None:
            raise RuntimeError(
                "DirectGPSGraphTransformerBaseline requires edge attributes."
            )

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        delta = self.decoder(
            x.reshape(batch * self.n_nodes, -1)
        ).reshape(
            batch,
            self.n_nodes,
            self.forecast_horizon,
        )

        delta = (
            delta.permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )

        return context[:, -1:].expand_as(delta) + delta


def _nearest_width(
    factory: Callable[[int], nn.Module],
    target_parameters: int,
    candidates: range,
) -> tuple[int, int]:
    choices = []

    for width in candidates:
        model = factory(int(width))
        parameters = count_parameters(model)

        choices.append(
            (
                abs(parameters - target_parameters),
                int(width),
                int(parameters),
            )
        )

        del model

    _, width, parameters = min(choices)

    return width, parameters


def build_all_direct14_model_specs(
    cfg: Direct14Config,
    graph: GraphStatic,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    graph_wno_block,
    device: torch.device,
    *,
    sa_band_counts: tuple[int, ...] = (2, 6),
) -> tuple[dict[str, DirectModelSpec], dict[str, dict[str, int]]]:
    """Build SA variants plus every prior benchmark family for direct 14→14."""

    if cfg.history != 14 or cfg.forecast_horizon != 14:
        raise ValueError(
            "This experiment requires history=forecast_horizon=14."
        )

    specs, matched = build_direct14_model_specs(
        cfg=cfg,
        graph=graph,
        modes=modes,
        eigenvalues=eigenvalues,
        graph_wno_block=graph_wno_block,
        device=device,
        sa_band_counts=sa_band_counts,
    )

    reference_bands = 2 if 2 in sa_band_counts else int(
        sa_band_counts[0]
    )

    reference_name = direct_sa_name(reference_bands)

    reference_model = specs[reference_name].factory()
    target_parameters = count_parameters(reference_model)

    del reference_model

    factories = {
        direct_rnn_name(): lambda width: DirectRNNBaseline(
            history=cfg.history,
            forecast_horizon=cfg.forecast_horizon,
            hidden_size=width,
            layers=3,
            dropout=0.1,
        ),
        direct_rnn_gnn_fusion_name(): lambda width: (
            DirectRNNGNNFusionBaseline(
                graph=graph,
                history=cfg.history,
                forecast_horizon=cfg.forecast_horizon,
                hidden_size=width,
                dropout=0.2,
            )
        ),
        direct_gat_name(): lambda width: DirectGATBaseline(
            graph=graph,
            history=cfg.history,
            forecast_horizon=cfg.forecast_horizon,
            width=width,
            layers=3,
            heads=4,
        ),
        direct_gatv2_name(): lambda width: DirectGATv2Baseline(
            graph=graph,
            history=cfg.history,
            forecast_horizon=cfg.forecast_horizon,
            width=width,
            layers=3,
            heads=4,
        ),
        direct_gps_name(): lambda width: DirectGPSGraphTransformerBaseline(
            graph=graph,
            history=cfg.history,
            forecast_horizon=cfg.forecast_horizon,
            width=width,
            layers=3,
            heads=4,
        ),
    }

    matched_baselines = {
        "reference_sa": {
            "temporal_bands": int(reference_bands),
            "parameters": int(target_parameters),
        },
        "meshgraphnet": {
            "width": int(matched["meshgraphnet_width"]),
            "parameters": int(matched["meshgraphnet_parameters"]),
        },
    }

    for name, factory in factories.items():
        width, parameters = _nearest_width(
            factory,
            target_parameters,
            range(16, 129, 4),
        )

        matched_baselines[name] = {
            "width": int(width),
            "parameters": int(parameters),
        }

        specs[name] = DirectModelSpec(
            name=name,
            factory=lambda factory=factory, width=width: (
                factory(width).to(device)
            ),
            metadata={
                "family": name.replace("-D14", ""),
                "history": cfg.history,
                "forecast_horizon": cfg.forecast_horizon,
                "width": int(width),
                "matched_to": reference_name,
                "training_mode": "direct_14_to_14",
            },
        )

    return specs, matched_baselines