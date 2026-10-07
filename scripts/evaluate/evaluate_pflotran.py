#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np
from twig.evaluation.metrics import rmse


def main():
    p = argparse.ArgumentParser(); p.add_argument("predictions"); p.add_argument("--output", default=None)
    a = p.parse_args(); data = np.load(a.predictions)
    metrics = {"rmse": float(rmse(data["predictions"], data["targets"]))}
    output = Path(a.output) if a.output else Path(a.predictions).with_name("metrics.json")
    output.write_text(json.dumps(metrics, indent=2) + "\n"); print(json.dumps(metrics))


if __name__ == "__main__": main()
