import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("curves", nargs="+"); parser.add_argument("--output", default="figures/si_diffusion/rollout.pdf")
    args = parser.parse_args()
    for item in args.curves:
        values = np.load(item); values = values[values.files[0]] if hasattr(values, "files") else values
        plt.plot(np.asarray(values).mean(axis=0), label=Path(item).stem)
    plt.xlabel("Forecast step"); plt.ylabel("RMSE"); plt.legend(); Path(args.output).parent.mkdir(parents=True, exist_ok=True); plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__": main()
