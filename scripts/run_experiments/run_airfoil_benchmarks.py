#!/usr/bin/env python
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'train/train_airfoil.py'), run_name='__main__')
