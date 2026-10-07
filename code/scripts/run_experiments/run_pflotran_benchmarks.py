#!/usr/bin/env python
"""Convenience alias for the final PFLOTRAN paper runner."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'train/train_pflotran.py'), run_name='__main__')
