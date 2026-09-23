#!/usr/bin/env python
"""Reproduce the paper's capacity-matched SI direct-14 benchmark."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from tcwgno.si_benchmark.data_direct14 import prepare_direct14_data
from tcwgno.si_benchmark.direct14_all_models import build_all_direct14_model_specs
from tcwgno.si_benchmark.direct14_config import Direct14Config, Direct14Paths
from tcwgno.si_benchmark.direct14_operator_baselines import build_direct14_operator_baseline_specs
from tcwgno.si_benchmark.direct14_registry import count_parameters, import_graph_wno_block
from tcwgno.si_benchmark.graph import make_graph_static
from tcwgno.si_benchmark.swiglu import SWIGLU_NAME, build_swiglu_spec
from tcwgno.si_benchmark.training_direct14 import aggregate_direct_table, run_direct_family


def build_specs(cfg, paths, device):
    data, modes, eigenvalues = prepare_direct14_data(paths, cfg)
    graph = make_graph_static(data.fixed_pos_features, data.node_xy, data.edge_index, data.edge_weight)
    block = import_graph_wno_block(paths.models_3d_dir)
    base, matched = build_all_direct14_model_specs(
        cfg, graph, modes, eigenvalues, block, device, sa_band_counts=(2, 5)
    )
    reference = count_parameters(base["SA-TC-WGNO-D14-K2"].factory())
    operators, operator_match = build_direct14_operator_baseline_specs(
        cfg, graph, modes, eigenvalues, block, device, target_parameters=reference
    )
    specs = {**base, **operators}
    specs[SWIGLU_NAME] = build_swiglu_spec(base)
    return data, specs, {"base": matched, "operators": operator_match, "target_parameters": reference}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/si_diffusion/si_diffusion_data.pt",
                   help="First-100 .pt package or full SI_equation_dataset.npy")
    p.add_argument("--edges", default="data/si_diffusion/si_diffusion_graph_edges.pt")
    p.add_argument("--coordinates", default="data/si_diffusion/si_diffusion_node_coordinates.csv")
    p.add_argument("--output", default="results/si_diffusion/direct14_paper")
    p.add_argument("--models", nargs="*", help="Defaults to every paper-table model")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--allow-short-rollout", action="store_true", help="Explicitly allow shortened data; results will not match the 100-step paper protocol")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--force", action="store_true")
    args = p.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    cfg = replace(Direct14Config(), epochs=args.epochs, early_stopping_patience=10)
    paths = Direct14Paths(data=Path(args.data), edges=Path(args.edges), output=output,
        models_3d_dir=Path(__file__).resolve().parents[2] / "src/tcwgno/si_benchmark",
        coordinate_cache=output / "nuts3_true_coordinates.pt", coordinate_csv=Path(args.coordinates))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, specs, capacity = build_specs(cfg, paths, device)
    order = (SWIGLU_NAME,
             "GraphFNO-D14", "GWNO-D14", "GPS-Transformer-D14", "GATv2-D14", "GAT-D14",
             "MeshGraphNet-D14", "RNN-GNN-Fusion-D14", "RNN-D14")
    selected = tuple(args.models) if args.models else order
    unknown = set(selected) - set(specs)
    if unknown: p.error("unknown model(s): " + ", ".join(sorted(unknown)))
    effective_rollout = data.effective_rollout_steps(cfg.rollout_steps, cfg.history)
    if effective_rollout != cfg.rollout_steps and not args.allow_short_rollout:
        p.error("The paper requires 100 forecast steps after 14 context steps. Supply the full SI array, or explicitly use --allow-short-rollout.")
    (output / "config.json").write_text(json.dumps({
        **asdict(cfg),
        "seeds_by_model": {name: args.seeds or ([42, 43, 45] if name == SWIGLU_NAME else [42, 43, 44]) for name in selected},
        "available_timesteps": data.available_timesteps,
        "effective_rollout_steps": effective_rollout,
    }, indent=2))
    (output / "capacity_match.json").write_text(json.dumps(capacity, indent=2))
    families = [run_direct_family(specs[name], tuple(args.seeds or ([42, 43, 45] if name == SWIGLU_NAME else [42, 43, 44])), data, cfg, device, output, force=args.force) for name in selected]
    aggregate_direct_table(families).to_csv(output / "metrics.csv", index=False)


if __name__ == "__main__": main()
