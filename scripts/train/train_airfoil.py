#!/usr/bin/env python
"""Train the frozen original 200-state Airfoil selections."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from tcwgno.airfoil_benchmark import config, registry, training
from tcwgno.airfoil_benchmark.paper import build_paper_specs
from tcwgno.airfoil_benchmark.data import load_meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/airfoil/raw'))
    p.add_argument('--artifacts', type=Path, default=Path('data/airfoil/artifacts'))
    p.add_argument('--output', type=Path, default=Path('results/airfoil/selected'))
    p.add_argument('--models', nargs='+', help='Quoted display names, e.g. "TC-WGNO" "GPS Transformer"')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[2]/'configs/paper_selections.json').read_text())
    selected = manifest['datasets']['airfoil']['models']
    if a.models:
        if set(a.models) - selected.keys(): p.error('unknown model name')
        selected = {name: selected[name] for name in a.models}
    load_meta(a.data)
    cfg = replace(config.CONFIG, data_dir=a.data, artifact_dir=a.artifacts,
                  output_dir=a.output, epochs=a.epochs)
    config.CONFIG = registry.CONFIG = training.CONFIG = cfg
    specs, matched = build_paper_specs(cfg, selected)
    training.build_model_specs = lambda: (specs, matched)
    for entry in selected.values():
        for run, seed in zip(entry['runs'], entry['seeds']):
            if cfg.seed + run - 1 != seed: raise ValueError('Run/seed mismatch')
            protocol = entry['training_by_run'][str(run)]
            run_cfg = replace(cfg, **{k: v for k, v in protocol.items() if k not in ('epochs', 'n_modes')})
            config.CONFIG = registry.CONFIG = training.CONFIG = run_cfg
            training.train_model(entry['model_key'], run=run, force=a.force)


if __name__ == '__main__':
    main()
