from twig.models.twig import (
    GraphStatic3D, canonicalize_edge_index, make_batched_edge_index,
    make_graph_static_3d, make_node_static_features,
)

__all__ = ["GraphStatic3D", "canonicalize_edge_index", "make_batched_edge_index",
           "make_graph_static_3d", "make_node_static_features"]


def knn_edge_index(positions, k=6):
    """Directed k-nearest-neighbour edges without optional graph packages."""
    import torch
    pos = torch.as_tensor(positions, dtype=torch.float32)
    distance = torch.cdist(pos, pos)
    neighbours = distance.topk(min(k + 1, len(pos)), largest=False).indices[:, 1:]
    source = torch.arange(len(pos)).repeat_interleave(neighbours.shape[1])
    return torch.stack((source, neighbours.reshape(-1)))
