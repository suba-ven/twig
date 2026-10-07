"""Frozen Airfoil architectures from the plotted run records."""
import numpy as np
import torch
from . import models, registry


def build_paper_specs(cfg, selections):
    graph, modes, evals = registry.load_graph_and_basis()
    if modes.shape[1] < 128:
        raise ValueError('Airfoil requires the original 128-mode basis')
    modes, evals = modes[:, :128], evals[:128]
    specs, matched = {}, {}
    for label, entry in selections.items():
        name, arch = entry['model_key'], entry['architecture']
        width = arch.get('width', arch.get('hidden_size'))
        h, f, c = cfg.history, cfg.forecast_horizon, cfg.channels
        if label == 'TWIG':
            factory = lambda w=width: models.PaperTWIG3D(graph, evals, modes, h, f, c, w, bands=5, depth=10, n_scales=12, n_modes=128, dropout=0.05, enhanced_input=False)
        elif label == 'Graph FNO':
            basis = np.load(cfg.artifact_dir / 'airflow200_modes169.npz')
            fno_modes = torch.from_numpy(np.asarray(basis['modes'], dtype=np.float32))
            if fno_modes.shape[1] != 169:
                raise ValueError('Selected Graph FNO requires exactly 169 modes')
            factory = lambda w=width: models.GraphFNO3D(graph, fno_modes, h, f, c, w, depth=14, n_modes=169, dropout=0.05)
        elif label == 'Graph WNO':
            factory = lambda w=width: models.PlainGWNO3D(graph, evals, modes, h, f, c, w, depth=10, n_scales=4, n_modes=128, dropout=0.05)
        elif label == 'MeshGraphNet':
            factory = lambda w=width: models.MeshGraphNetDirect3D(graph, h, f, c, w, depth=10, dropout=0.05)
        elif label in ('GAT', 'GATv2', 'GPS Transformer'):
            cls = {'GAT': models.GATDirect3D, 'GATv2': models.GATv2Direct3D, 'GPS Transformer': models.GPSDirect3D}[label]
            depth, heads = (6, 6) if label == 'GPS Transformer' else (3, 4)
            factory = lambda w=width, cls=cls, depth=depth, heads=heads: cls(graph, h, f, c, w, depth=depth, heads=heads)
        elif label == 'RNN':
            factory = lambda w=width: models.RNNDirect3D(h, f, c, w, layers=3, dropout=0.1)
        elif label == 'RNN-GNN Fusion':
            factory = lambda w=width: models.RNNGNNFusionDirect3D(graph, h, f, c, w, dropout=0.2)
        else:
            raise ValueError(label)
        probe = factory()
        parameters = models.count_parameters(probe)
        del probe
        if parameters != entry['parameters']:
            raise ValueError(f'{label}: expected {entry["parameters"]} parameters, got {parameters}; check original Airfoil geometry')
        specs[name] = models.ModelSpec(name, label, parameters, factory)
        matched[name] = arch
    return specs, matched
