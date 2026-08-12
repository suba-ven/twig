from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphStatic:
    node_static: torch.Tensor
    node_xy: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_attr: torch.Tensor


def make_edge_attributes(
    node_xy: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    displacement = node_xy[dst] - node_xy[src]
    distance = displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat(
        [displacement, distance, torch.log1p(edge_weight[:, None].clamp_min(0.0))],
        dim=-1,
    )


def make_graph_static(
    node_static: torch.Tensor,
    node_xy: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> GraphStatic:
    xy = (node_xy - node_xy.mean(dim=0, keepdim=True)) / node_xy.std(
        dim=0, keepdim=True
    ).clamp_min(1e-8)
    return GraphStatic(
        node_static=node_static.detach().cpu().float().contiguous(),
        node_xy=xy.detach().cpu().float().contiguous(),
        edge_index=edge_index.detach().cpu().long().contiguous(),
        edge_weight=edge_weight.detach().cpu().float().contiguous(),
        edge_attr=make_edge_attributes(xy, edge_index, edge_weight).detach().cpu().float().contiguous(),
    )


class BlockGraphCache:
    def __init__(self):
        self._cache = {}

    def get(self, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, n_nodes: int, batch_size: int):
        key = (edge_index.data_ptr(), 0 if edge_attr is None else edge_attr.data_ptr(), n_nodes, batch_size, str(edge_index.device))
        if key not in self._cache:
            offsets = torch.arange(batch_size, device=edge_index.device).view(batch_size, 1, 1) * n_nodes
            indices = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            attrs = None if edge_attr is None else edge_attr.repeat(batch_size, 1)
            self._cache[key] = (indices, attrs)
        return self._cache[key]


def segment_sum(messages: torch.Tensor, destination: torch.Tensor, n_nodes: int) -> torch.Tensor:
    output = messages.new_zeros(n_nodes, messages.shape[-1])
    output.index_add_(0, destination, messages)
    return output


def segment_mean(messages: torch.Tensor, destination: torch.Tensor, n_nodes: int) -> torch.Tensor:
    output = segment_sum(messages, destination, n_nodes)
    counts = messages.new_zeros(n_nodes, 1)
    counts.index_add_(0, destination, torch.ones_like(messages[:, :1]))
    return output / counts.clamp_min(1.0)
