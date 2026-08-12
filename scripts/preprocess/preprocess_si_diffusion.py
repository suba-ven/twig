#!/usr/bin/env python
"""Validate and record inputs for the paper's SI direct-14 benchmark."""
import argparse, json
from dataclasses import asdict
from pathlib import Path
from tcwgno.si_benchmark.data_direct14 import prepare_direct14_data
from tcwgno.si_benchmark.direct14_config import Direct14Config, Direct14Paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/si_diffusion/si_diffusion_data.pt")
    p.add_argument("--edges", default="data/si_diffusion/si_diffusion_graph_edges.pt")
    p.add_argument("--coordinates", default="data/si_diffusion/si_diffusion_node_coordinates.csv")
    p.add_argument("--output-dir", default="data/si_diffusion")
    a = p.parse_args(); output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    cfg = Direct14Config(); paths = Direct14Paths(Path(a.data), Path(a.edges), output,
        Path(__file__).resolve().parents[2] / "src/tcwgno/si_benchmark",
        output / "nuts3_true_coordinates.pt", Path(a.coordinates))
    prepared, modes, eigenvalues = prepare_direct14_data(paths, cfg)
    manifest = {"protocol": "direct_14_to_14", "config": asdict(cfg),
        "data": str(Path(a.data).resolve()), "edges": str(Path(a.edges).resolve()),
        "coordinates": str(Path(a.coordinates).resolve()),
        "splits": {"train": prepared.train_scenario_ids, "validation": prepared.val_scenario_ids,
                   "test": prepared.test_scenario_ids}, "modes_shape": list(modes.shape),
        "available_timesteps": prepared.available_timesteps,
        "effective_rollout_steps": prepared.effective_rollout_steps(cfg.rollout_steps, cfg.history),
        "target_parameters": 70224}
    (output / "dataset_info.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "dataset_info.json")


if __name__ == "__main__": main()
