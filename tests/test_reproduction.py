"""Offline integration and integrity checks for the Code Ocean entry point.

Can also run without pytest: python tests/test_reproduction.py
"""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vis.reproduce_paper import FIGURES, validate_inputs


class ReproductionTests(unittest.TestCase):
    def test_frozen_input_integrity(self):
        self.assertEqual(len(validate_inputs()), 8)
        with tempfile.TemporaryDirectory() as folder:
            clone = Path(folder)
            shutil.copytree(ROOT/'configs', clone/'configs')
            shutil.copytree(ROOT/'results/paper', clone/'results/paper')
            with (clone/'results/paper/si_diffusion_curves.npz').open('ab') as f:
                f.write(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                validate_inputs(clone)

    def test_run_from_other_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'results with spaces'
            env = dict(os.environ, RESULTS_DIR=str(output), PYTHON=sys.executable)
            subprocess.run([str(ROOT/'run')], cwd=folder, env=env, check=True)
            for filename in FIGURES:
                self.assertGreater((output/'figures'/filename).stat().st_size, 1000)
            with (output/'tables/manuscript_summary.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 27)
            for row in rows:
                self.assertEqual(row['sd_ddof'], '0' if row['dataset']=='si_diffusion' else '1')
            report=json.loads((output/'reproduction.json').read_text())
            self.assertEqual(set(report['figures']), set(FIGURES))
            self.assertIn('tables/manuscript_summary.tex', report['outputs_sha256'])


if __name__ == '__main__':
    unittest.main()
