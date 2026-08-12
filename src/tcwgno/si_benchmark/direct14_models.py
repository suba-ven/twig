from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .graph import BlockGraphCache, GraphStatic, segment_sum


# These compact local copies avoid importing the package-wide `models`
# namespace, which is unnecessary for the direct SA TC-WGNO / MeshGraphNet
# experiment and can otherwise pull in optional PyG baseline dependencies.
class MLP(nn.Module):
    def __init__(self, dims: list[int], layer_norm: bool = False):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must include an input and output width.")

        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dims[-1]) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class StaticGraphModule(nn.Module):
    def __init__(self, graph: GraphStatic):
        super().__init__()
        self.register_buffer("node_static", graph.node_static)
        self.register_buffer("edge_index_static", graph.edge_index)
        self.register_buffer("edge_weight_static", graph.edge_weight)
        self.register_buffer("edge_attr_static", graph.edge_attr)
        self.block_graphs = BlockGraphCache()

    @property
    def n_nodes(self) -> int:
        return int(self.node_static.shape[0])

    def validate_context(
        self,
        context: torch.Tensor,
        history: int,
    ) -> None:
        expected = (history, self.n_nodes, 1)
        if context.ndim != 4 or tuple(context.shape[1:]) != expected:
            raise ValueError(
                f"Expected (B,{history},{self.n_nodes},1), got "
                f"{tuple(context.shape)}."
            )

    def batched_graph(
        self,
        batch_size: int,
        *,
        edge_attr: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.block_graphs.get(
            self.edge_index_static,
            self.edge_attr_static if edge_attr else None,
            self.n_nodes,
            batch_size,
        )


class MeshGraphNetBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.edge_model = MLP(
            [3 * width, width, width, width],
            layer_norm=True,
        )
        self.node_model = MLP(
            [2 * width, width, width, width],
            layer_norm=True,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_delta = self.edge_model(
            torch.cat([nodes[src], nodes[dst], edges], dim=-1)
        )
        edges = edges + edge_delta
        aggregate = segment_sum(edges, dst, nodes.shape[0])
        nodes = nodes + self.node_model(
            torch.cat([nodes, aggregate], dim=-1)
        )
        return nodes, edges


class TimeCausalHistoryEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        n_bands: int,
        alpha_min: float = 0.15,
        alpha_max: float = 0.85,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_bands = int(n_bands)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_logits = nn.Parameter(
            torch.linspace(-1.0, 1.0, self.n_bands)
        )

    @property
    def out_channels(self) -> int:
        return self.channels * (self.n_bands + 2)

    def alphas(self) -> torch.Tensor:
        return self.alpha_min + (
            self.alpha_max - self.alpha_min
        ) * torch.sigmoid(self.alpha_logits)

    @staticmethod
    def causal_ema(
        sequence: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        states = [sequence[:, :, 0]]
        running = sequence[:, :, 0]

        for index in range(1, sequence.shape[2]):
            running = (
                alpha * running
                + (1.0 - alpha) * sequence[:, :, index]
            )
            states.append(running)

        return torch.stack(states, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.permute(0, 2, 1, 3).contiguous()
        current = sequence[:, :, -1]
        level = sequence
        bands = []

        for alpha in self.alphas():
            smooth = self.causal_ema(level, alpha)
            bands.append(level[:, :, -1] - smooth[:, :, -1])
            level = smooth

        return torch.cat(
            [current, *bands, level[:, :, -1]],
            dim=-1,
        )


class HistoryMatchedEncoder(TimeCausalHistoryEncoder):
    def __init__(
        self,
        channels: int,
        history: int,
        n_bands: int,
        alpha_min: float,
        alpha_max: float,
        min_cumulative_lag: float,
        max_lag_fraction: float,
    ):
        super().__init__(channels, n_bands, alpha_min, alpha_max)

        max_lag = max(
            min_cumulative_lag + 1e-3,
            max_lag_fraction * (history - 1),
        )
        target = torch.exp(
            torch.linspace(
                math.log(min_cumulative_lag),
                math.log(max_lag),
                n_bands,
            )
        )
        incremental = torch.diff(
            torch.cat([torch.zeros(1), target])
        )
        alpha = (
            incremental / (1.0 + incremental)
        ).clamp(alpha_min + 1e-4, alpha_max - 1e-4)
        normalized = (
            (alpha - alpha_min) / (alpha_max - alpha_min)
        ).clamp(1e-6, 1.0 - 1e-6)

        self.alpha_logits = nn.Parameter(torch.logit(normalized))
        self.register_buffer("target_cumulative_lags", target)

    def cumulative_lags(self) -> torch.Tensor:
        alpha = self.alphas()
        return torch.cumsum(alpha / (1.0 - alpha), dim=0)

    def scale_anchor_loss(self) -> torch.Tensor:
        return (
            torch.log(self.cumulative_lags().clamp_min(1e-6))
            - torch.log(
                self.target_cumulative_lags.clamp_min(1e-6)
            )
        ).square().mean()


@dataclass(frozen=True)
class TCWGNOArchitecture:
    width: int
    depth: int
    temporal_bands: int
    input_channels: int
    output_channels: int
    wavelet_scales: int
    residual_output: bool = False


class StaticTCWGNO(nn.Module):
    def __init__(
        self,
        graph_wno_block,
        modes: torch.Tensor,
        eigenvalues: torch.Tensor,
        fixed_pos_features: torch.Tensor,
        architecture: TCWGNOArchitecture,
        temporal_encoder: nn.Module,
    ):
        super().__init__()
        self.output_channels = architecture.output_channels
        self.residual_output = architecture.residual_output
        self.temporal_encoder = temporal_encoder

        self.register_buffer(
            "U_modes",
            modes.float(),
            persistent=False,
        )
        self.register_buffer(
            "evals_modes",
            eigenvalues.float(),
            persistent=False,
        )
        self.register_buffer(
            "fixed_pos_features",
            fixed_pos_features.float(),
            persistent=False,
        )

        input_dim = (
            temporal_encoder.out_channels
            + fixed_pos_features.shape[1]
        )
        self.input_proj = nn.Linear(input_dim, architecture.width)
        self.blocks = nn.ModuleList(
            [
                graph_wno_block(
                    width=architecture.width,
                    n_modes=modes.shape[1],
                    n_scales=architecture.wavelet_scales,
                    dropout=0.0,
                )
                for _ in range(architecture.depth)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(architecture.width, architecture.width),
            nn.GELU(),
            nn.Linear(
                architecture.width,
                architecture.output_channels,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        temporal = self.temporal_encoder(x)
        positional = self.fixed_pos_features.unsqueeze(0).expand(
            batch,
            -1,
            -1,
        )
        hidden = self.input_proj(
            torch.cat([temporal, positional], dim=-1)
        )

        for block in self.blocks:
            hidden = block(hidden, self.U_modes, self.evals_modes)

        output = self.output_proj(hidden)
        if self.residual_output:
            output = x[:, -1, :, : self.output_channels] + output

        return output


class DirectHorizonTCWGNO(nn.Module):
    """Direct multi-frame head around the static TC-WGNO trunk."""

    def __init__(
        self,
        graph_wno_block,
        modes: torch.Tensor,
        eigenvalues: torch.Tensor,
        fixed_pos_features: torch.Tensor,
        architecture: TCWGNOArchitecture,
        temporal_encoder: nn.Module,
        forecast_horizon: int,
    ):
        super().__init__()

        if architecture.output_channels != int(forecast_horizon):
            raise ValueError(
                "architecture.output_channels must equal forecast_horizon; "
                f"got {architecture.output_channels} and "
                f"{forecast_horizon}."
            )
        if architecture.residual_output:
            raise ValueError(
                "Direct multi-output TC-WGNO requires residual_output=False."
            )

        self.forecast_horizon = int(forecast_horizon)
        self.trunk = StaticTCWGNO(
            graph_wno_block=graph_wno_block,
            modes=modes,
            eigenvalues=eigenvalues,
            fixed_pos_features=fixed_pos_features,
            architecture=architecture,
            temporal_encoder=temporal_encoder,
        )

    @property
    def temporal_encoder(self) -> nn.Module:
        return self.trunk.temporal_encoder

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        direct_output = self.trunk(context)

        if (
            direct_output.ndim != 3
            or direct_output.shape[-1] != self.forecast_horizon
        ):
            raise RuntimeError(
                "DirectHorizonTCWGNO received unexpected trunk output "
                f"{tuple(direct_output.shape)}."
            )

        return (
            direct_output
            .permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )


class DirectMeshGraphNetBaseline(StaticGraphModule):
    """MeshGraphNet with one direct output channel per future frame."""

    def __init__(
        self,
        graph: GraphStatic,
        history: int,
        forecast_horizon: int,
        width: int,
        processor_steps: int = 3,
    ):
        super().__init__(graph)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)

        self.node_encoder = MLP(
            [
                self.history + self.node_static.shape[1],
                width,
                width,
                width,
            ],
            layer_norm=True,
        )
        self.edge_encoder = MLP(
            [
                self.edge_attr_static.shape[1],
                width,
                width,
                width,
            ],
            layer_norm=True,
        )
        self.processors = nn.ModuleList(
            [
                MeshGraphNetBlock(width)
                for _ in range(int(processor_steps))
            ]
        )
        self.decoder = MLP(
            [width, width, width, self.forecast_horizon]
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
        nodes = self.node_encoder(
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
                "DirectMeshGraphNet requires edge attributes."
            )
        edges = self.edge_encoder(edge_attr)

        for processor in self.processors:
            nodes, edges = processor(nodes, edges, edge_index)

        deltas = self.decoder(nodes).reshape(
            batch,
            self.n_nodes,
            self.forecast_horizon,
        )
        deltas = (
            deltas
            .permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )

        baseline = context[:, -1:].expand(
            -1,
            self.forecast_horizon,
            -1,
            -1,
        )
        return baseline + deltas
