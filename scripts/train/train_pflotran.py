#!/usr/bin/env python
import argparse, json
from pathlib import Path
import torch

from tcwgno.data.pflotran import build_block_dataloaders_3d
from tcwgno.models.tc_wgno import compute_laplacian_basis, make_graph_static_3d, build_pflotran_h10_model_specs
from tcwgno.training.trainer import TrainConfig, run_model_family


def main():
    p = argparse.ArgumentParser(description="Train one PFLOTRAN model family")
    p.add_argument("inputs", nargs="+"); p.add_argument("--model", required=True); p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--output", default="results/pflotran"); p.add_argument("--force", action="store_true")
    a = p.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_block_dataloaders_3d(a.inputs, history=10, forecast_horizon=10, batch_size=8, seed=42,
        split_stats_path=Path(a.output)/"dataset_info.json")
    train, val, test, info = loaders; sample = next(iter(train)); graph = make_graph_static_3d(sample[4], sample[5])
    eigenvalues, modes = compute_laplacian_basis(graph, n_modes=128)
    specs, matched = build_pflotran_h10_model_specs(graph, modes, eigenvalues, target_parameters=500_000, sa_band_counts=(5,7), n_modes=128)
    if a.model not in specs: p.error(f"unknown model {a.model}; choose from {', '.join(specs)}")
    cfg = TrainConfig(epochs=a.epochs); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out/"capacity_match.json").write_text(json.dumps(matched, indent=2))
    run_model_family(specs[a.model], a.seeds, train, val, test, info, cfg, device, out, force=a.force)


if __name__ == "__main__": main()
