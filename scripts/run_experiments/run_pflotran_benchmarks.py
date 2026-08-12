#!/usr/bin/env python
import argparse, subprocess, sys
from tcwgno.utils.model_registry import PFLOTRAN_MODELS


def main():
    p = argparse.ArgumentParser(); p.add_argument("inputs", nargs="+"); p.add_argument("--models", nargs="+", default=PFLOTRAN_MODELS); p.add_argument("--epochs", type=int, default=30); p.add_argument("--force", action="store_true")
    a = p.parse_args()
    for model in a.models:
        command = [sys.executable, "scripts/train/train_pflotran.py", *a.inputs, "--model", model, "--epochs", str(a.epochs)]
        if a.force: command.append("--force")
        subprocess.run(command, check=True)


if __name__ == "__main__": main()
