#!/usr/bin/env python
import argparse
from tcwgno.data.pflotran import build_block_dataloaders_3d


def main():
    p = argparse.ArgumentParser(description="Create leakage-free PFLOTRAN split/statistics manifest")
    p.add_argument("inputs", nargs="+", help="HDF5 files, directory, glob, or zip")
    p.add_argument("--output", default="data/pflotran/dataset_info.json")
    p.add_argument("--history", type=int, default=10); p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    build_block_dataloaders_3d(a.inputs, history=a.history, forecast_horizon=a.horizon,
        batch_size=a.batch_size, seed=a.seed, split_stats_path=a.output)
    print(a.output)


if __name__ == "__main__": main()
