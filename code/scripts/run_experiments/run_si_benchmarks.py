#!/usr/bin/env python
"""Convenience alias for the exact SI paper benchmark runner."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "train/train_si_diffusion.py"), run_name="__main__")
