from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import models
from .config import CONFIG


def load_graph_and_basis():
    data = np.load(CONFIG.artifact_dir / "airflow200_static.npz")
    graph = models.make_graph_static_3d(
        torch.from_numpy(data["pos"]), torch.from_numpy(data["edge_index"])
    )
    # Preserve each benchmark architecture while giving every model the static
    # boundary information required by Airfoil.
    node_type = torch.from_numpy(data["node_type"].reshape(-1)).long()
    classes = int(node_type.max().item()) + 1
    graph.node_static = torch.cat([graph.node_static, F.one_hot(node_type, classes).float()], dim=-1)
    return graph, torch.from_numpy(data["modes"]), torch.from_numpy(data["evals"])


def _nearest(factory: Callable[[int], torch.nn.Module], candidates) -> tuple[int, int]:
    candidates = list(candidates)
    cache = {}
    def measure(index: int):
        width = int(candidates[index])
        if width not in cache:
            model = factory(width)
            cache[width] = int(models.count_parameters(model))
            del model
        return cache[width]
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if measure(mid) < CONFIG.target_parameters:
            lo = mid + 1
        else:
            hi = mid
    indices = {max(0, lo - 1), lo, min(len(candidates) - 1, lo + 1)}
    choices = [(abs(measure(i) - CONFIG.target_parameters), int(candidates[i]), measure(i)) for i in indices]
    _, width, params = min(choices)
    return width, params


def build_model_specs() -> tuple[dict[str, models.ModelSpec], dict[str, dict[str, int]]]:
    graph, modes, evals = load_graph_and_basis()
    h, f, c, k = CONFIG.history, CONFIG.forecast_horizon, CONFIG.channels, CONFIG.n_modes
    specs = {}
    matched = {}

    for bands in (5, 7):
        name = models.pflotran_sa_name(bands)
        fac = lambda width, bands=bands: models.SATWIG3D(
            graph, evals, modes, h, f, c, width, bands, depth=10,
            n_scales=4, n_modes=k, dropout=0.05, scale_aware=True)
        width, params = _nearest(fac, range(64, 701, 4))
        matched[name] = {"width": width, "parameters": params, "bands": bands}
        specs[name] = models.ModelSpec(name, f"SA TWIG (K={bands})", params,
                                       lambda width=width, fac=fac: fac(width))

    name = models.pflotran_plain_gwno_name()
    fac = lambda width: models.PlainGWNO3D(graph, evals, modes, h, f, c, width,
                                            depth=10, n_scales=4, n_modes=k, dropout=0.05)
    width, params = _nearest(fac, range(64, 701, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "Plain Graph WNO", params,
                                   lambda fac=fac, width=width: fac(width))

    name = models.pflotran_graph_fno_name()
    fac = lambda width: models.GraphFNO3D(graph, modes, h, f, c, width,
                                          depth=10, n_modes=k, dropout=0.05)
    width, params = _nearest(fac, range(32, 181, 2))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "Graph FNO", params,
                                   lambda fac=fac, width=width: fac(width))

    name = models.pflotran_mesh_name()
    fac = lambda width: models.MeshGraphNetDirect3D(graph, h, f, c, width, depth=10, dropout=0.05)
    width, params = _nearest(fac, range(64, 901, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "MeshGraphNet", params,
                                   lambda fac=fac, width=width: fac(width))

    name = models.pflotran_gat_name()
    fac = lambda width: models.GATDirect3D(graph, h, f, c, width, depth=3, heads=4)
    width, params = _nearest(fac, range(64, 1401, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "GAT", params,
                                   lambda fac=fac, width=width: fac(width))

    name = models.pflotran_gatv2_name()
    fac = lambda width: models.GATv2Direct3D(graph, h, f, c, width, depth=3, heads=4)
    width, params = _nearest(fac, range(64, 1401, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "GATv2", params,
                                   lambda fac=fac, width=width: fac(width))

    name = models.pflotran_gps_name()
    fac = lambda width: models.GPSDirect3D(graph, h, f, c, width, depth=3, heads=4)
    width, params = _nearest(fac, range(64, 1401, 4))
    matched[name] = {"width": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "GPS Transformer", params,
                                   lambda fac=fac, width=width: fac(width))

    name = models.pflotran_rnn_name()
    fac = lambda width: models.RNNDirect3D(h, f, c, width, layers=3, dropout=0.1)
    width, params = _nearest(fac, range(64, 1801, 8))
    matched[name] = {"hidden_size": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "RNN", params,
                                   lambda fac=fac, width=width: fac(width))

    name = models.pflotran_rnn_gnn_fusion_name()
    fac = lambda width: models.RNNGNNFusionDirect3D(graph, h, f, c, width, dropout=0.2)
    width, params = _nearest(fac, range(64, 1801, 8))
    matched[name] = {"hidden_size": width, "parameters": params}
    specs[name] = models.ModelSpec(name, "RNN-GNN Fusion", params,
                                   lambda fac=fac, width=width: fac(width))
    return specs, matched
