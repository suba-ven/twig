#!/usr/bin/env python
"""Train the final PFLOTRAN nine-model comparison (H10/F10, 1M, step 60)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from tcwgno.pflotran_benchmark.pflotran_block_dataset_3d_fixed import build_block_dataloaders_3d
from tcwgno.pflotran_benchmark.pflotran_h10_models import compute_laplacian_basis, make_graph_static_3d
from tcwgno.pflotran_benchmark.pflotran_h10_training import TrainConfig, run_model_family
from tcwgno.pflotran_benchmark.paper import build_paper_specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+')
    p.add_argument('--model', help='Model key; omitted means all nine selected models')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--output', type=Path, default=Path('results/pflotran/selected'))
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, val, test, info = build_block_dataloaders_3d(
        a.inputs, history=10, forecast_horizon=10, batch_size=8, seed=0,
        train_frac=0.8, val_frac=0.1, q_low=1, q_high=99, normalize=True,
        split_stats_path=a.output/'dataset_info.json')
    sample = next(iter(train))
    graph = make_graph_static_3d(sample[4], sample[5])
    eigenvalues, modes = compute_laplacian_basis(graph, n_modes=128)
    specs, matched = build_paper_specs(graph, modes, eigenvalues)
    if a.model and a.model not in specs:
        p.error('unknown model; choose from ' + ', '.join(specs))
    cfg = TrainConfig(epochs=a.epochs, rollout_steps=60, lr=5e-4,
                      weight_decay=1e-5, early_stopping_patience=10, min_lr=1e-6)
    (a.output/'config.json').write_text(json.dumps({**asdict(cfg), 'seeds': a.seeds, 'split_seed': 0}, indent=2))
    (a.output/'capacity_match.json').write_text(json.dumps(matched, indent=2))
    for name in ([a.model] if a.model else specs):
        run_model_family(specs[name], a.seeds, train, val, test, info, cfg, device, a.output, force=a.force)


if __name__ == '__main__':
    main()
